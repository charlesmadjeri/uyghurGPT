#!/usr/bin/env python3
"""Install or verify flash-attn in uyghur_env (cluster GPU job / one-time setup).

``push.py --install-deps`` runs this before training. It prints full pip
output (no silent failure) and always exits 0 so a failed FA2 build does
not cancel the Slurm job — check the log for ``[flash-attn] OK`` vs
``FAILED``.

Always invoke with the micromamba interpreter (``micromamba activate`` is
often a no-op inside ``srun --pty bash``):

  $HOME/micromamba/envs/uyghur_env/bin/python scripts/ensure_flash_attn.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Same interpreter as scripts/push.py REMOTE_PYTHON (expanded on the cluster).
UYGHUR_ENV_PYTHON = Path(
    os.environ.get(
        "UYGHURGPT_PYTHON",
        Path.home() / "micromamba/envs/uyghur_env/bin/python",
    )
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _reexec_if_wrong_python() -> None:
    """``micromamba activate`` often does not apply in a plain ``srun`` shell."""
    exe = Path(sys.executable).resolve()
    if "uyghur_env" in str(exe):
        return
    if not UYGHUR_ENV_PYTHON.is_file():
        _log(
            f"[flash-attn] ERROR: running {sys.executable} but "
            f"{UYGHUR_ENV_PYTHON} is missing.\n"
            "  Create uyghur_env first (docs/SERVER_CONFIG.md §1.2), then rerun:\n"
            f"    {UYGHUR_ENV_PYTHON} {Path(__file__).resolve()}"
        )
        raise SystemExit(0)
    _log(
        f"[flash-attn] Wrong interpreter {sys.executable} "
        f"(expected uyghur_env). Re-execing with {UYGHUR_ENV_PYTHON} …"
    )
    os.execv(str(UYGHUR_ENV_PYTHON), [str(UYGHUR_ENV_PYTHON), *sys.argv[1:]])


def _try_import() -> tuple[bool, str]:
    try:
        import flash_attn  # noqa: F401

        import flash_attn as fa

        return True, getattr(fa, "__version__", "unknown")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _find_nvcc() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    for base in (
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12",
        "/usr/local/cuda",
    ):
        if not base:
            continue
        candidate = Path(base) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    return None


def _prepend_cuda_toolkit_to_path() -> str | None:
    """Expose nvcc when the cluster has /usr/local/cuda-* but no modules."""
    nvcc = _find_nvcc()
    if not nvcc:
        return None
    cuda_bin = str(Path(nvcc).parent)
    cuda_home = str(Path(nvcc).parent.parent)
    path = os.environ.get("PATH", "")
    if cuda_bin not in path.split(":"):
        os.environ["PATH"] = f"{cuda_bin}:{path}"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    return nvcc


def _diagnostics() -> None:
    _log("[flash-attn] Environment diagnostics:")
    _log(f"  python: {sys.executable}")
    _log(f"  version: {sys.version.split()[0]}")
    nvcc = _prepend_cuda_toolkit_to_path() or _find_nvcc()
    _log(f"  nvcc: {nvcc or 'NOT ON PATH (set CUDA_HOME or module load cuda)'}")
    if os.environ.get("CUDA_HOME"):
        _log(f"  CUDA_HOME: {os.environ['CUDA_HOME']}")
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            first = (out.stdout or out.stderr).strip().splitlines()[:1]
            if first:
                _log(f"  nvcc version: {first[0]}")
        except OSError:
            pass
    try:
        import torch

        _log(f"  torch: {torch.__version__}")
        _log(f"  torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            _log(f"  GPU0: {torch.cuda.get_device_name(0)} capability={cap}")
        else:
            _log("  GPU0: cuda not available in this process (need a GPU allocation)")
    except ImportError:
        _log("  torch: not installed in this interpreter")


def _pip_install(*args: str) -> int:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    _log(f"[flash-attn] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def _prebuilt_wheel_url(version: str) -> str | None:
    """GitHub release wheel matching uyghur_env (torch 2.5 + cu12 + cp311)."""
    import torch

    torch_major_minor = ".".join(torch.__version__.split(".")[:2])
    # flash-attn release tags use torch2.5 not torch2.5.1
    torch_tag = torch_major_minor.replace(".", "")
    cxx = "TRUE" if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False) else "FALSE"
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    name = (
        f"flash_attn-{version}+cu12torch{torch_tag}cxx11abi{cxx}-"
        f"{py}-{py}-linux_x86_64.whl"
    )
    return (
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v{version}/{name}"
    )


def _try_prebuilt_wheel(version: str) -> bool:
    """Install from GitHub wheel (avoids NFS cross-device link on source builds)."""
    url = _prebuilt_wheel_url(version)
    if not url:
        return False
    _log(f"[flash-attn] Trying prebuilt wheel: {url}")
    rc = _pip_install("--no-cache-dir", url)
    if rc == 0:
        return True
    _log("[flash-attn] Prebuilt wheel install failed; falling back to source build.")
    return False


def _usage_hint() -> str:
    script = Path(__file__).resolve()
    return (
        f"  {UYGHUR_ENV_PYTHON} {script}\n"
        "Do not rely on bare `python` or `micromamba activate` inside "
        "`srun --pty` unless you ran `eval \"$(micromamba shell hook -s bash)\"` first."
    )


def main() -> int:
    _reexec_if_wrong_python()

    ok, detail = _try_import()
    if ok:
        _log(f"[flash-attn] OK — already importable (version {detail})")
        return 0

    _log(f"[flash-attn] Not importable yet: {detail}")
    _diagnostics()

    try:
        import torch  # noqa: F401
    except ImportError:
        _log(
            "[flash-attn] ERROR: torch is not installed in uyghur_env. "
            "Install the stack first (docs/SERVER_CONFIG.md §1.3):\n"
            f"  {UYGHUR_ENV_PYTHON} -m pip install -r requirements.txt jinja2 huggingface_hub\n"
            f"  {UYGHUR_ENV_PYTHON} -m pip install --index-url "
            "https://download.pytorch.org/whl/cu121 torch\n"
            f"  {UYGHUR_ENV_PYTHON} -m pip install "
            "'fsspec[http]>=2023.1.0,<=2026.2.0'\n"
            "Then rerun:\n"
            + _usage_hint()
        )
        return 0

    if _prepend_cuda_toolkit_to_path() is None:
        _log(
            "[flash-attn] ERROR: nvcc not found. On JU workers try:\n"
            "  export PATH=/usr/local/cuda-12.3/bin:$PATH\n"
            "  export CUDA_HOME=/usr/local/cuda-12.3\n"
            "Or: module load cuda/12.x if modules exist.\n"
            + _usage_hint()
        )
        return 0

    version = os.environ.get("UYGHURGPT_FLASH_ATTN_VERSION", "2.7.4.post1")
    if _try_prebuilt_wheel(version):
        ok, detail = _try_import()
        if ok:
            _log(f"[flash-attn] OK — prebuilt wheel installed (version {detail})")
            return 0

    _pip_install("ninja", "packaging", "wheel", "setuptools")
    env = os.environ.copy()
    _prepend_cuda_toolkit_to_path()
    env.setdefault("MAX_JOBS", "4")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9")
    # Keep pip temp/cache on one filesystem (NFS cross-device link otherwise).
    pip_home = Path.home() / ".cache" / "uyghur_pip"
    pip_home.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(pip_home)
    env["TMPDIR"] = str(pip_home)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        f"flash-attn=={version}",
    ]
    _log(f"[flash-attn] Building flash-attn=={version} from source (10–30 min)…")
    _log(f"[flash-attn] MAX_JOBS={env.get('MAX_JOBS')} TORCH_CUDA_ARCH_LIST={env.get('TORCH_CUDA_ARCH_LIST')}")
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc != 0:
        _log(f"[flash-attn] FAILED — pip exited {rc}")
        _log("[flash-attn] Rerun after fixing the diagnostics above:\n" + _usage_hint())
        return 0

    ok, detail = _try_import()
    if ok:
        _log(f"[flash-attn] OK — installed and importable (version {detail})")
        return 0

    _log(f"[flash-attn] FAILED — pip succeeded but import still fails: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
