"""
Bootstrap do ambiente de transcricao.

Cria um venv isolado dentro da propria skill, detecta se ha GPU NVIDIA
utilizavel de verdade (nao so presente, mas funcional com faster-whisper)
e instala o motor de transcricao adequado. Idempotente: se ja rodou antes
nesta maquina, so confirma a configuracao existente em vez de reinstalar.

Uso:
    python setup_env.py [--force]

--force refaz a deteccao de GPU e reinstala as dependencias mesmo que o
ambiente ja esteja configurado (util se o usuario acabou de instalar
drivers NVIDIA novos, por exemplo).
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_DIR / ".venv"
ENGINE_CONFIG_PATH = VENV_DIR / "engine.json"


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd, **kwargs):
    print("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_venv():
    if VENV_DIR.exists():
        print(f"Ambiente virtual ja existe em {VENV_DIR}, reaproveitando.")
        return
    print(f"Criando ambiente virtual isolado em {VENV_DIR} ...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])


def pip_install(*packages):
    run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", *packages])


def detect_nvidia_smi() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def gpu_smoke_test() -> bool:
    """Tenta de verdade carregar um modelo tiny na GPU e transcrever 1s de
    silencio. So confiamos na GPU se isso funcionar — nvidia-smi presente
    nao garante que CUDA/cuDNN estao corretamente instalados para o
    faster-whisper conseguir usar a placa."""
    test_code = (
        "from faster_whisper import WhisperModel\n"
        "import numpy as np\n"
        "m = WhisperModel('tiny', device='cuda', compute_type='float16')\n"
        "list(m.transcribe(np.zeros(16000, dtype=np.float32))[0])\n"
        "print('SMOKE_TEST_OK')\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python()), "-c", test_code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return result.returncode == 0 and "SMOKE_TEST_OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False


def main():
    force = "--force" in sys.argv

    ensure_venv()

    if ENGINE_CONFIG_PATH.exists() and not force:
        cfg = json.loads(ENGINE_CONFIG_PATH.read_text(encoding="utf-8"))
        print(
            f"Ambiente ja configurado (device: {cfg['device']}/{cfg['compute_type']}, "
            f"modelo padrao: {cfg['default_model']}). Use --force para reconfigurar."
        )
        _print_next_command()
        return

    print("Instalando dependencias (av, numpy, faster-whisper)...")
    pip_install("av", "numpy", "faster-whisper")

    device = None
    compute_type = None
    default_model = None

    if detect_nvidia_smi():
        print("GPU NVIDIA detectada. Instalando runtime CUDA (cuBLAS/cuDNN via pip)...")
        try:
            pip_install("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
            print("Validando se a GPU realmente funciona com faster-whisper (teste real, nao so deteccao)...")
            if gpu_smoke_test():
                device, compute_type = "cuda", "float16"
                default_model = "large-v3"
                print("GPU validada com sucesso. Device: cuda/float16 (modelo padrao: large-v3).")
            else:
                print(
                    "AVISO: GPU foi detectada mas o teste de transcricao falhou "
                    "(driver CUDA/cuDNN ausente ou incompativel). Caindo para CPU com seguranca."
                )
        except subprocess.CalledProcessError as e:
            print(f"AVISO: falha ao instalar/validar runtime CUDA ({e}). Caindo para CPU com seguranca.")

    if device is None:
        # faster-whisper roda em CPU com quantizacao int8 (ctranslate2), o que
        # e sensivelmente mais rapido que o pacote "whisper" classico no mesmo
        # tamanho de modelo — por isso da pra usar "medium" por padrao aqui em
        # vez de "small", com melhor qualidade sem pagar muito mais tempo.
        device, compute_type = "cpu", "int8"
        default_model = "medium"
        print(f"Sem GPU utilizavel. Device: cpu/int8 (modelo padrao: {default_model}).")

    VENV_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_CONFIG_PATH.write_text(
        json.dumps(
            {"engine": "faster-whisper", "device": device, "compute_type": compute_type, "default_model": default_model},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nAmbiente pronto. Device: {device}/{compute_type} (modelo padrao: {default_model})")
    _print_next_command()


def _print_next_command():
    print(
        "\nPara transcrever, use:\n"
        f'  "{venv_python()}" "{SKILL_DIR / "scripts" / "transcribe.py"}" <caminho_do_audio>'
    )


if __name__ == "__main__":
    main()
