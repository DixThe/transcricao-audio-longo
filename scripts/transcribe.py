"""
Transcreve um arquivo de audio longo para .srt, processando em chunks,
salvando incrementalmente e retomando de onde parou se for interrompido.

Precisa rodar com o Python do venv criado por setup_env.py (ele tem as
dependencias av / numpy / faster-whisper instaladas). O motor e sempre
faster-whisper (GPU com float16 quando disponivel, CPU com int8 caso
contrario) — ver references/architecture.md para o porque de nao usar
mais o pacote "whisper" classico em nenhum dos dois casos.

Uso:
    python transcribe.py <caminho_do_audio> [--output ARQ.srt]
                          [--output-dir PASTA] [--language pt]
                          [--model auto] [--chunk-seconds 300]
                          [--vocab "termo1, termo2, ..."]

--vocab aceita uma lista de palavras/expressoes especificas do assunto do
audio (nomes proprios, jargao tecnico, girias do negocio) para ajudar o
modelo a acertar a grafia de termos que ele erraria por falta de contexto
— ver references/architecture.md para exemplos e o porque disso funcionar.

Ver references/architecture.md (na raiz da skill) para o raciocinio por
tras do corte de chunk no silencio, do streaming com memoria limitada e
do mecanismo de retomada.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np

SKILL_DIR = Path(__file__).resolve().parent.parent
ENGINE_CONFIG_PATH = SKILL_DIR / ".venv" / "engine.json"

TARGET_SR = 16000

AUTHOR_BANNER = "transcricao-audio-longo por DixThe | https://github.com/DixThe/transcricao-audio-longo"


# ---------------------------------------------------------------------------
# Decodificacao de audio (streaming, sem depender de ffmpeg do sistema)
# ---------------------------------------------------------------------------

def get_audio_duration_seconds(audio_path: Path) -> float:
    container = av.open(str(audio_path))
    try:
        stream = container.streams.audio[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        raise RuntimeError("Nao foi possivel determinar a duracao do audio.")
    finally:
        container.close()


def get_stream_start_offset_seconds(audio_path: Path) -> float:
    """Alguns arquivos (tipicamente clipes recortados de uma gravacao maior
    com stream-copy, ou certos formatos de gravador/celular) tem o relogio
    interno do stream comecando em um valor diferente de zero
    (`stream.start_time`), mesmo que o conteudo de audio comece no inicio
    do arquivo. Sem compensar isso, comparar o tempo absoluto de cada frame
    contra uma janela de chunk 0-based (0s, 90s, 300s...) nunca bate, e o
    decode_window silenciosamente nao coleta nenhuma amostra. Ver
    references/architecture.md para o caso real que motivou isso."""
    container = av.open(str(audio_path))
    try:
        stream = container.streams.audio[0]
        if stream.start_time is not None:
            return float(stream.start_time * stream.time_base)
        return 0.0
    finally:
        container.close()


def decode_window(audio_path: Path, start_s: float, end_s: float, start_offset_s: float = 0.0, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decodifica somente o intervalo [start_s, end_s) do audio (em segundos
    relativos ao inicio real do audio), mixado para mono e reamostrado para
    target_sr. `start_offset_s` compensa arquivos cujo relogio interno nao
    comeca em zero (ver get_stream_start_offset_seconds). Reabre o container
    a cada chamada de proposito: mantem o uso de memoria limitado ao tamanho
    do proprio intervalo, nao ao arquivo inteiro (importante para audios de
    horas)."""
    if end_s <= start_s:
        return np.zeros(0, dtype=np.float32)

    abs_start = start_offset_s + start_s
    abs_end = start_offset_s + end_s

    container = av.open(str(audio_path))
    try:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)

        seek_target = max(0.0, abs_start - 1.0)
        container.seek(int(seek_target / stream.time_base), stream=stream, backward=True, any_frame=False)

        chunks = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time < abs_start:
                continue
            if frame_time >= abs_end:
                break
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.size:
                    chunks.append(arr)

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        pcm = np.concatenate(chunks, axis=1).flatten()
        return (pcm.astype(np.float32)) / 32768.0
    finally:
        container.close()


def find_silence_cut(audio_path: Path, target_time_s: float, duration_s: float, start_offset_s: float = 0.0, radius_s: float = 3.0) -> float:
    """Procura o ponto de menor energia (mais proximo do silencio) numa
    janela em torno de target_time_s, para nao cortar um chunk no meio de
    uma palavra. Se nao houver janela valida (ex: perto do fim do audio),
    apenas usa o tempo alvo sem ajuste."""
    lo = max(0.0, target_time_s - radius_s)
    hi = min(duration_s, target_time_s + radius_s)
    if hi <= lo:
        return min(target_time_s, duration_s)

    audio = decode_window(audio_path, lo, hi, start_offset_s=start_offset_s)
    frame_len = int(TARGET_SR * 0.05)  # janelas de 50ms para medir energia
    if audio.size < frame_len:
        return min(target_time_s, duration_s)

    n_frames = audio.size // frame_len
    energies = np.array([
        np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2) for i in range(n_frames)
    ])
    quietest_frame = int(np.argmin(energies))
    return lo + (quietest_frame * frame_len) / TARGET_SR


# ---------------------------------------------------------------------------
# Motor de transcricao
# ---------------------------------------------------------------------------

def load_engine_config() -> dict:
    if not ENGINE_CONFIG_PATH.exists():
        print(
            "ERRO: ambiente nao configurado. Rode primeiro:\n"
            f"  python \"{SKILL_DIR / 'scripts' / 'setup_env.py'}\"",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(ENGINE_CONFIG_PATH.read_text(encoding="utf-8"))


def load_model(device: str, compute_type: str, model_name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_chunk(model, audio_array: np.ndarray, language: str | None, initial_prompt: str | None):
    """Retorna lista de (start_relativo, end_relativo, texto) em segundos
    relativos ao inicio do chunk. `initial_prompt` funciona como uma dica
    de vocabulario: o modelo usa esse texto como contexto para enviesar a
    grafia de termos ambiguos (nomes proprios, jargao) na direcao certa,
    sem afetar o conteudo transcrito em si."""
    if audio_array.size == 0:
        return []

    segments, _info = model.transcribe(
        audio_array,
        language=language,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )
    return [(seg.start, seg.end, seg.text) for seg in segments]


# ---------------------------------------------------------------------------
# Escrita incremental do .srt
# ---------------------------------------------------------------------------

def format_srt_timestamp(total_seconds: float) -> str:
    total_seconds = max(0.0, total_seconds)
    ms = int(round(total_seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_entries(srt_file, entries, start_index: int) -> int:
    index = start_index
    for start, end, text in entries:
        text = text.strip()
        if not text:
            continue
        srt_file.write(f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}\n\n")
        index += 1
    srt_file.flush()
    return index


# ---------------------------------------------------------------------------
# Progresso / retomada
# ---------------------------------------------------------------------------

def audio_fingerprint(audio_path: Path) -> str:
    stat = audio_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def params_fingerprint(args, device: str, compute_type: str, model_name: str) -> str:
    """Usa o modelo/device/compute_type ja resolvidos (nao o '--model auto'
    literal) para que uma troca no motor padrao (ex: engine.json atualizado
    apos um `setup_env.py --force`) invalide corretamente um progresso salvo
    com o motor antigo, em vez de tentar retomar misturando transcricoes de
    modelos diferentes no mesmo .srt."""
    return json.dumps(
        {
            "language": args.language,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "chunk_seconds": args.chunk_seconds,
            "vocab": args.vocab,
        },
        sort_keys=True,
    )


def load_progress(progress_path: Path):
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_progress(progress_path: Path, next_chunk_start_s: float, next_srt_index: int, fingerprint: str, params: str):
    progress_path.write_text(
        json.dumps(
            {
                "audio_fingerprint": fingerprint,
                "params": params,
                "next_chunk_start_s": next_chunk_start_s,
                "next_srt_index": next_srt_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Orquestracao principal
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Transcreve audio longo para .srt")
    parser.add_argument("audio", help="Caminho do arquivo de audio/video de entrada")
    parser.add_argument("--output", default=None, help="Caminho completo do .srt de saida")
    parser.add_argument("--output-dir", default=None, help="Pasta de saida (default: ~/Documents/Transcricoes)")
    parser.add_argument("--language", default="pt", help="Codigo do idioma (default: pt) ou 'auto' para deteccao automatica")
    parser.add_argument("--model", default="auto", help="Tamanho do modelo Whisper, ou 'auto' para usar o padrao do motor detectado")
    parser.add_argument("--chunk-seconds", type=float, default=300.0, help="Duracao alvo de cada chunk, em segundos (default: 300)")
    parser.add_argument("--vocab", default=None, help="Lista de termos/nomes especificos do assunto do audio, separados por virgula, para melhorar a precisao (ex: \"lote, loteamento, imobiliaria, boleto\")")
    return parser.parse_args()


def default_output_path(audio_path: Path, output_dir: str | None) -> Path:
    out_dir = Path(output_dir) if output_dir else Path.home() / "Documents" / "Transcricoes"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (audio_path.stem + ".srt")


def main():
    print(AUTHOR_BANNER)
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"ERRO: arquivo nao encontrado: {audio_path}", file=sys.stderr)
        sys.exit(1)

    engine_cfg = load_engine_config()
    device = engine_cfg["device"]
    compute_type = engine_cfg["compute_type"]
    model_name = engine_cfg["default_model"] if args.model == "auto" else args.model
    language = None if args.language == "auto" else args.language
    initial_prompt = args.vocab.strip() if args.vocab else None

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(audio_path, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")

    fingerprint = audio_fingerprint(audio_path)
    params = params_fingerprint(args, device, compute_type, model_name)
    progress = load_progress(progress_path)

    if progress and progress.get("audio_fingerprint") == fingerprint and progress.get("params") == params:
        chunk_start = progress["next_chunk_start_s"]
        srt_index = progress["next_srt_index"]
        srt_file = open(output_path, "a", encoding="utf-8")
        print(f"Progresso anterior encontrado. Retomando a partir de {chunk_start:.1f}s.")
    else:
        chunk_start = 0.0
        srt_index = 1
        srt_file = open(output_path, "w", encoding="utf-8")

    print(f"Motor: faster-whisper | device: {device} ({compute_type}) | modelo: {model_name} | idioma: {language or 'auto-detectar'}")
    if initial_prompt:
        print(f"Vocabulario de apoio: {initial_prompt}")
    print("Carregando modelo (pode demorar na primeira vez)...")
    try:
        model = load_model(device, compute_type, model_name)
    except Exception as e:
        if device == "cuda":
            print(f"AVISO: falha ao carregar modelo '{model_name}' na GPU ({e}). Tentando fallback para CPU...")
            device, compute_type = "cpu", "int8"
            model = load_model(device, compute_type, model_name)
        else:
            raise

    duration = get_audio_duration_seconds(audio_path)
    start_offset_s = get_stream_start_offset_seconds(audio_path)
    if start_offset_s > 0.01:
        print(
            f"Aviso: o arquivo tem um deslocamento interno de {start_offset_s:.2f}s "
            "(comum em trechos recortados de uma gravacao maior) — compensando automaticamente."
        )
    print(f"Duracao total: {duration / 60:.1f} min. Salvando em: {output_path}")

    chunk_len = args.chunk_seconds
    started_at = time.time()

    while chunk_start < duration - 0.05:
        target_end = min(chunk_start + chunk_len, duration)
        actual_end = find_silence_cut(audio_path, target_end, duration, start_offset_s=start_offset_s) if target_end < duration else duration
        if actual_end <= chunk_start:
            actual_end = target_end

        try:
            audio_array = decode_window(audio_path, chunk_start, actual_end, start_offset_s=start_offset_s)
            raw_entries = transcribe_chunk(model, audio_array, language, initial_prompt)
            entries = [(chunk_start + s, chunk_start + e, t) for (s, e, t) in raw_entries]
        except Exception as e:
            print(f"AVISO: falha ao transcrever trecho {chunk_start:.1f}s-{actual_end:.1f}s ({e}). Marcando e seguindo em frente.")
            entries = [(chunk_start, actual_end, "[trecho nao pode ser transcrito]")]

        srt_index = write_srt_entries(srt_file, entries, srt_index)
        chunk_start = actual_end
        save_progress(progress_path, chunk_start, srt_index, fingerprint, params)

        pct = 100 * chunk_start / duration
        elapsed = time.time() - started_at
        print(f"Progresso: {chunk_start:.1f}s / {duration:.1f}s ({pct:.1f}%) | tempo decorrido: {elapsed:.0f}s")

    srt_file.close()
    progress_path.unlink(missing_ok=True)
    print(f"\nTranscricao concluida: {output_path}")


if __name__ == "__main__":
    main()
