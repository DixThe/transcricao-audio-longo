---
name: transcricao-audio-longo
description: Transcreve áudios e vídeos longos (reuniões, calls, aulas, entrevistas, gravações de campo) em arquivos .srt com timestamps precisos, processando em chunks para não perder contexto nem travar em arquivos de 1h30 ou mais. Detecta automaticamente se há GPU disponível e usa faster-whisper em ambos os casos (GPU com float16, CPU com int8), com fallback seguro se a GPU falhar. Aceita um vocabulário de domínio (nomes, jargão técnico) para melhorar a precisão em áudios com termos específicos. Salva o .srt incrementalmente à medida que cada trecho é transcrito, então se o processo cair no meio o progresso não se perde e a transcrição pode ser retomada de onde parou. Use esta skill sempre que o usuário pedir para transcrever um áudio, gerar legenda, converter uma gravação/reunião/call em texto ou .srt, ou mencionar arquivos como .m4a, .mp3, .wav, .ogg, .mp4 de reuniões, aulas, calls ou entrevistas — mesmo que ele não diga a palavra "transcrição" explicitamente (ex: "passa esse áudio pra texto", "preciso da legenda dessa call de ontem", "converte essa gravação em srt").
compatibility: Requer Python 3.10+ e acesso à internet na primeira execução (para instalar dependências e baixar o modelo). Cria um ambiente virtual isolado dentro da própria skill — não interfere no Python global do usuário. Não depende de ffmpeg instalado no sistema.
---

# Transcrição de Áudio Longo

> Autor: [DixThe](https://github.com/DixThe) · repositório oficial: https://github.com/DixThe/transcricao-audio-longo

Transforma um arquivo de áudio (mesmo de várias horas) em uma legenda `.srt` bem formada, sem estourar memória e sem perder o trabalho já feito se algo falhar no meio do caminho.

## Por que isso não é "só rodar o Whisper"

Rodar `whisper.transcribe()` direto num áudio de 1h30 tem três problemas na prática: o processo fica horas sem dar sinal de vida, uma queda de energia ou um erro no meio faz perder tudo, e carregar o áudio inteiro na memória de uma vez é desnecessário e arriscado. Esta skill resolve isso decodificando e transcrevendo em pedaços (chunks) de poucos minutos por vez, salvando o `.srt` a cada pedaço concluído. Isso também é o motivo de existir um script pronto em vez de pedir para você escrever o código na hora: a lógica de chunking, corte no silêncio, retomada e fallback de GPU já está resolvida e testada — reaproveite.

## Fluxo de uso

### 1. Preparar o ambiente (só precisa rodar uma vez por máquina)

```bash
python "<skill_dir>/scripts/setup_env.py"
```

Substitua `<skill_dir>` pelo caminho real desta skill (o diretório que contém este `SKILL.md`). Esse script:
- Cria um ambiente virtual isolado em `<skill_dir>/.venv` (não mexe no Python global do usuário).
- Detecta se há GPU NVIDIA disponível (`nvidia-smi`) e faz um teste real de carregar e transcrever um trecho mínimo com `faster-whisper` antes de confiar na GPU.
- O motor é sempre `faster-whisper` — na GPU com `device=cuda, compute_type=float16` (modelo padrão `large-v3`), na CPU com `device=cpu, compute_type=int8` (modelo padrão `medium`). Não usa mais o pacote `whisper` clássico em nenhum dos dois casos: `faster-whisper` é mais rápido também em CPU, o que compensa usar um modelo maior (melhor qualidade) no mesmo orçamento de tempo. Ver `references/architecture.md` para a comparação que motivou essa troca.
- Se a GPU existir mas o teste falhar (driver CUDA/cuDNN ausente ou incompatível — comum em máquinas Windows sem toolkit instalado), cai automaticamente para CPU. Isso é o fallback seguro pedido pela spec: nunca trava a instalação por causa da GPU.
- Se já tiver rodado antes nesta máquina, é rápido (não reinstala nada) — só reexecute quando quiser forçar uma nova detecção (`--force`).

Ele termina imprimindo o comando exato para o passo 2, com o caminho do Python do ambiente virtual já resolvido.

### 2. Rodar a transcrição

```bash
"<skill_dir>/.venv/Scripts/python.exe" "<skill_dir>/scripts/transcribe.py" "<caminho_do_audio>"
```
(em Linux/macOS o Python fica em `<skill_dir>/.venv/bin/python`)

Isso já basta para o caso comum. Parâmetros opcionais:

| Flag | Default | Quando mudar |
|---|---|---|
| `--output <caminho.srt>` | (nenhum — usa `--output-dir`) | Quando o usuário pedir um nome/local específico para o `.srt` |
| `--output-dir <pasta>` | `~/Documents/Transcricoes` | Quando o usuário pedir para salvar em outro lugar |
| `--language pt` | `pt` | Áudio em outro idioma: passe o código (`en`, `es`...) ou `auto` para detecção automática |
| `--model auto` | `auto` (large-v3 na GPU, medium na CPU) | Se o usuário pedir explicitamente mais velocidade (`tiny`/`base`/`small`) ou mais precisão (`large-v3`, mesmo na CPU — só vai ser mais lento) |
| `--chunk-seconds 300` | 300 (5 min) | Áudios muito longos (3h+) podem se beneficiar de chunks maiores para reduzir overhead |
| `--vocab "termo1, termo2, ..."` | (nenhum) | Quando o áudio tem nomes próprios, jargão técnico ou termos do negócio que o modelo tende a errar (ex: `--vocab "lote, loteamento, imobiliária, boleto, corretor"` para uma reunião de imobiliária) — melhora bastante a grafia desses termos sem precisar trocar de modelo. Ver `references/architecture.md` para o porquê disso funcionar |

**Importante sobre tempo de execução:** transcrever 1h30 de áudio pode levar de poucos minutos (com GPU) a mais de uma hora (CPU, modelo `medium`/`large-v3`). Rode o comando em background (`run_in_background: true` na ferramenta Bash) em vez de esperar de forma bloqueante, e acompanhe o progresso lendo o próprio arquivo `.srt` de saída (ele cresce a cada chunk) ou o console — o script imprime uma linha de progresso (`"X.Xs / Y.Ys (Z%)"`) a cada chunk concluído. Se a prioridade for velocidade em vez de qualidade, `--model small` reduz bastante o tempo de CPU às custas de mais erros de transcrição.

### 3. Interrupções e retomada

Se o processo for interrompido (erro, fechamento do terminal, queda de energia), **rode exatamente o mesmo comando de novo**. O script guarda um arquivo de progresso ao lado do `.srt` (`<nome>.progress.json`) e detecta que já existe trabalho feito para aquele áudio com aqueles mesmos parâmetros, retomando do último chunk concluído em vez de começar do zero. Se algum trecho específico falhar (áudio corrompido, erro do motor de transcrição), o script não aborta o restante: grava uma entrada marcando `[trecho não pôde ser transcrito]` naquele intervalo e continua para o próximo chunk.

## Formato de saída

`.srt` padrão, com índice sequencial, timestamps `HH:MM:SS,mmm` e o texto do trecho:

```
1
00:00:00,000 --> 00:00:04,320
Bom dia, pessoal, vamos começar a reunião.

2
00:00:04,320 --> 00:00:09,150
Hoje o pauta principal é o fechamento do trimestre.
```

## Detalhes de design (leia se for mexer no código ou investigar um comportamento inesperado)

A decodificação de áudio, o corte de chunk no ponto de silêncio mais próximo (em vez de um corte seco no meio de uma palavra), a lógica de streaming com memória limitada e as decisões sobre GPU/CUDA estão documentadas em `references/architecture.md`. Vale ler antes de alterar `scripts/transcribe.py` ou `scripts/setup_env.py` para entender o porquê das escolhas, não só o quê.

## Troubleshooting rápido

- **"nvidia-smi não encontrado" mas a máquina tem GPU**: normal em algumas configurações (driver não expõe `nvidia-smi` no PATH). O script simplesmente usa CPU — não é um erro, é o fallback funcionando. Se o usuário insistir em usar a GPU, verifique os drivers NVIDIA fora desta skill antes de tentar de novo.
- **Instalação muito lenta na primeira vez**: esperado — `torch`/`faster-whisper` e o modelo de IA somam de algumas centenas de MB a alguns GB. Avise o usuário antes de rodar `setup_env.py` se a conexão for limitada.
- **Erro "arquivo não encontrado"**: confirme que o caminho do áudio está entre aspas se tiver espaços (comum em pastas como "Gravações de som").
