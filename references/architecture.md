# Arquitetura e decisões de design

Este documento explica o *porquê* das escolhas em `scripts/setup_env.py` e
`scripts/transcribe.py`. Leia antes de alterar o comportamento — várias
decisões aqui resolvem um problema específico que não é óbvio à primeira
vista.

## Por que PyAV em vez de ffmpeg de linha de comando

O `openai-whisper` normalmente decodifica áudio chamando o binário `ffmpeg`
via subprocess. Isso cria uma dependência externa que o usuário precisa
instalar manualmente, e quebra silenciosamente se o `ffmpeg` não estiver no
PATH (era exatamente o caso da máquina onde esta skill foi criada — sem
GPU e sem ffmpeg instalado).

A biblioteca `av` (PyAV) resolve isso porque seus wheels pré-compilados já
embutem as bibliotecas do FFmpeg estaticamente — `pip install av` funciona
sem precisar de nada instalado no sistema. Por isso o `transcribe.py`
decodifica o áudio inteiramente via PyAV e passa o resultado como array
numpy diretamente para `faster_whisper`, nunca deixando o motor tentar
chamar `ffmpeg` por conta própria (`faster-whisper` aceita um array numpy
já decodificado em vez de um caminho de arquivo — quando recebe um array,
pula a etapa de decodificação interna).

## Por que faster-whisper também na CPU (não só na GPU)

A primeira versão desta skill usava `faster-whisper` na GPU mas o pacote
`openai-whisper` clássico na CPU, com modelo `small` por padrão. Depois de
um usuário reclamar da qualidade da transcrição (termos do negócio saindo
errados — "boleto" virando "bolete", "imobiliária" virando "imoblera",
"lote" virando "Lotte"), a causa raiz identificada foi o modelo pequeno
demais para o vocabulário do áudio, não um bug de código.

A correção óbvia seria só trocar o modelo padrão de `small` para `medium`
mantendo `openai-whisper` — mas isso teria deixado a CPU ainda mais lenta
sem necessidade. `faster-whisper` (baseado em `ctranslate2`) já é
significativamente mais rápido que `openai-whisper` (PyTorch) para o mesmo
tamanho de modelo, mesmo em CPU, com quantização `int8`. Unificar os dois
casos (GPU e CPU) no mesmo motor elimina um code path inteiro do
`transcribe.py` (menos lugar pra bug se esconder) e "paga" o modelo maior
com a eficiência que se ganha trocando de motor — na prática, rodar
`medium` via `faster-whisper`/CPU/int8 saiu com tempo comparável a rodar
`small` via `openai-whisper`/CPU antes, mas com qualidade sensivelmente
melhor.

## Vocabulário de domínio (`--vocab` / `initial_prompt`)

Whisper (e por extensão `faster-whisper`) aceita um `initial_prompt`: um
texto usado como contexto para enviesar a transcrição, sem ele aparecer no
resultado. Passar os termos técnicos, nomes próprios ou jargão esperado do
áudio como `initial_prompt` ajuda bastante o modelo a "ouvir" a grafia certa
desses termos — é uma técnica leve (não muda o modelo nem o tempo de
execução de forma perceptível) que resolve exatamente o tipo de erro que
motivou a mudança de motor acima. A skill expõe isso como `--vocab`, uma
lista de termos separados por vírgula que vira o `initial_prompt`
diretamente. Não é um vocabulário fixo/hardcoded na skill de propósito: o
assunto de cada áudio muda (reunião de imobiliária, aula de programação,
entrevista médica...), então quem sabe o vocabulário certo para aquele
áudio especificamente é quem está pedindo a transcrição.

## Por que decodificar em janelas (`decode_window`) em vez do áudio inteiro

Carregar um áudio de 3 horas inteiro em memória como float32 mono a 16kHz
já daria ~690MB só de PCM — sem contar overhead do próprio motor de
transcrição. Para áudios ainda mais longos ou várias transcrições
simultâneas, isso escala mal. `decode_window` reabre o container e decodifica
apenas o intervalo `[start_s, end_s)` pedido, então o uso de memória fica
proporcional ao tamanho do chunk (alguns minutos), não ao áudio inteiro,
independente de o arquivo ter 10 minutos ou 10 horas.

O custo de reabrir o arquivo a cada chunk é irrelevante perto do tempo que a
própria transcrição leva (segundos de seek/decode contra minutos de
inferência do modelo) — não vale a pena complicar o código para manter o
container aberto entre chamadas.

## Por que cortar o chunk no silêncio (`find_silence_cut`) em vez de um corte fixo

Cortar um chunk exatamente em `chunk_start + chunk_seconds` frequentemente
cai no meio de uma palavra, o que degrada a transcrição bem na borda de
cada chunk (a última palavra de um chunk e a primeira do próximo tendem a
sair truncadas ou erradas). Duas alternativas foram consideradas:

1. **Sobreposição entre chunks + deduplicação de texto** — usado por várias
   implementações por aí, mas exige comparar texto transcrito de regiões
   sobrepostas para descartar duplicatas, o que é frágil (comparação de
   string não é confiável quando o modelo transcreve a mesma fala de forma
   ligeiramente diferente nas duas passagens).
2. **Corte no ponto de silêncio mais próximo do alvo** — decodifica uma
   janela pequena (±3s) ao redor do tempo alvo, calcula energia em frames
   de 50ms e escolhe o ponto de menor energia como corte real. Não precisa
   de nenhum modelo de VAD (voice activity detection) baseado em ML — só
   energia do sinal, que já é suficiente para achar uma pausa entre frases.

A opção 2 foi escolhida por ser mais simples de raciocinar e sem risco de
duplicar ou perder texto. O trade-off aceito: em fala contínua sem nenhuma
pausa perto do alvo, o corte cai no tempo exato mesmo assim (sem crash, só
sem o refinamento) — isso é raro o suficiente para não justificar mais
complexidade.

## Por que a detecção de GPU inclui um "smoke test" real

`nvidia-smi` só confirma que existe uma placa NVIDIA e um driver básico —
não garante que CUDA e cuDNN estão corretamente instalados e compatíveis
com o `faster-whisper` (via `ctranslate2`). É comum em máquinas Windows ter
a placa mas não o toolkit CUDA completo. Por isso `setup_env.py` não confia
apenas em `nvidia-smi`: depois de instalar `faster-whisper`, ele tenta de
fato carregar um modelo `tiny` na GPU e transcrever 1 segundo de silêncio.
Só marca o device como `cuda` se isso funcionar de ponta a ponta.
Se falhar em qualquer etapa, cai para CPU (`faster-whisper` com `int8`) automaticamente — é
o "fallback seguro em caso de falha na detecção de GPU" pedido na spec
original desta skill.

## Por que existe um ambiente virtual dentro da própria skill

`faster-whisper` puxa dependências pesadas (`ctranslate2`,
`onnxruntime`, runtime CUDA na GPU etc.) que não deveriam poluir o Python global do usuário nem
colidir com versões usadas por outros projetos. Um `.venv` isolado dentro
de `<skill_dir>` resolve isso e também torna a skill auto-contida: apagar a
pasta da skill remove tudo que ela instalou, sem deixar rastro no sistema.

## Retomada após interrupção

A cada chunk concluído, `transcribe.py` grava um `<saida>.srt.progress.json`
com a impressão digital do áudio de origem (mtime + tamanho do arquivo),
os parâmetros usados (idioma/modelo/tamanho de chunk) e o ponto onde parar
de retomar. Ao rodar de novo com o mesmo áudio e os mesmos parâmetros, o
script abre o `.srt` em modo *append* e continua do último chunk salvo em
vez de recomeçar. Mudar qualquer parâmetro (ex: trocar o modelo) invalida
o progresso salvo de propósito — misturar transcrições de modelos
diferentes no mesmo arquivo geraria um `.srt` inconsistente.

## Resiliência a chunk individual corrompido

Se um chunk específico falhar ao decodificar ou transcrever (trecho de
áudio corrompido, erro pontual do motor), o script não aborta a
transcrição inteira: grava uma entrada de texto
`[trecho nao pode ser transcrito]` cobrindo aquele intervalo de tempo e
segue para o próximo chunk. Preferir "degradar graciosamente" a "falhar
tudo por causa de uma parte" é o comportamento correto para áudios longos,
onde refazer do zero por causa de um trecho ruim é caro.
