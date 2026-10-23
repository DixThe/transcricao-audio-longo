# transcricao-audio-longo

> por [DixThe](https://github.com/DixThe) · [github.com/DixThe/transcricao-audio-longo](https://github.com/DixThe/transcricao-audio-longo)

Transcreve áudios e vídeos longos (reuniões, calls, aulas, entrevistas) para `.srt`, em pedaços, sem estourar memória e sem perder o que já foi feito se o processo cair no meio.

Funciona como script Python comum ou como **skill do Claude Code**.

## O que ele faz de diferente de "só rodar o Whisper"

- **Processa em chunks** de 5 minutos (configurável), decodificando só o trecho da vez. A memória usada é a de um chunk, não a do arquivo inteiro, então um áudio de 3h roda igual a um de 10 min.
- **Corta no silêncio.** Em vez de cortar seco no minuto 5 (no meio de uma palavra), procura a pausa mais próxima numa janela de ±3s.
- **Salva o `.srt` a cada chunk** e guarda um `.progress.json`. Caiu a energia? Roda o mesmo comando de novo e ele continua de onde parou.
- **Não precisa de ffmpeg instalado.** A decodificação usa PyAV, que já vem com o FFmpeg embutido.
- **GPU quando dá, CPU quando não dá.** O setup testa a GPU de verdade (carrega um modelo e transcreve 1s) antes de confiar nela. Se o CUDA/cuDNN não estiver certo, cai para CPU sem travar.
- **Vocabulário de domínio** (`--vocab`): passe nomes próprios e jargão do assunto e o modelo acerta a grafia ("boleto" em vez de "bolete").
- **Um trecho ruim não derruba tudo.** Se um chunk falhar, ele marca `[trecho nao pode ser transcrito]` naquele intervalo e segue.

Motor: [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Na GPU usa `float16` com `large-v3`; na CPU usa `int8` com `medium`.

## Requisitos

- Python 3.10+
- Internet na primeira execução (instala as dependências e baixa o modelo)
- Opcional: GPU NVIDIA

## Uso

### 1. Preparar o ambiente (uma vez por máquina)

```bash
python scripts/setup_env.py
```

Cria um `.venv` isolado dentro da própria pasta, instala `av`, `numpy` e `faster-whisper`, detecta a GPU e grava a escolha em `.venv/engine.json`. Use `--force` para refazer a detecção (por exemplo, depois de instalar drivers novos).

### 2. Transcrever

```bash
# Windows
.venv\Scripts\python.exe scripts\transcribe.py "C:\caminho\reuniao.m4a"

# Linux / macOS
.venv/bin/python scripts/transcribe.py ~/gravacoes/reuniao.m4a
```

Por padrão o `.srt` vai para `~/Documents/Transcricoes/<nome-do-audio>.srt`.

| Flag | Padrão | Para quê |
|---|---|---|
| `--output arquivo.srt` | — | Caminho exato da saída |
| `--output-dir pasta` | `~/Documents/Transcricoes` | Outra pasta de saída |
| `--language pt` | `pt` | Outro idioma (`en`, `es`...) ou `auto` |
| `--model auto` | `large-v3` na GPU, `medium` na CPU | `small`/`base` para ir mais rápido, `large-v3` para mais precisão |
| `--chunk-seconds 300` | `300` | Tamanho alvo de cada pedaço |
| `--vocab "termo1, termo2"` | — | Nomes e jargão que o modelo costuma errar |

Exemplo com vocabulário:

```bash
.venv/bin/python scripts/transcribe.py reuniao.m4a --vocab "loteamento, imobiliária, boleto, corretor"
```

### Tempo de execução

1h30 de áudio leva de poucos minutos (GPU) a mais de uma hora (CPU com `medium`). O script imprime o progresso a cada chunk, e o `.srt` vai crescendo enquanto roda.

## Como skill do Claude Code

Copie a pasta para `~/.claude/skills/transcricao-audio-longo/`. O `SKILL.md` ensina o Claude a rodar o setup, disparar a transcrição em background e retomar se cair. Depois é só pedir "transcreve esse áudio" com o caminho do arquivo.

## Por dentro

As decisões de design (por que PyAV, por que corte por energia em vez de sobreposição com deduplicação, por que o smoke test de GPU, como funciona a retomada) estão em [`references/architecture.md`](references/architecture.md).

## Autor

Criado por **[DixThe](https://github.com/DixThe)**. Se usar, der fork ou adaptar, mantenha o crédito (a licença MIT exige manter o aviso de copyright). Uma ⭐ no repo ajuda.

## Licença

[MIT](LICENSE) © [DixThe](https://github.com/DixThe)
