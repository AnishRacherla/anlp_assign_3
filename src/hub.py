import os
from pathlib import Path
from typing import Optional

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    from huggingface_hub import HfApi, login as hf_login, hf_hub_download
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

def load_env(env_path: str = ".env"):
    p = Path(env_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[dict] = None,
):
    load_env()
    if not _HAS_WANDB:
        return None
    key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
    return wandb.init(project=project, name=name, config=config, resume="allow")

def push_to_hub(
    local_path: str,
    repo_id: str,
    filename: Optional[str] = None,
    repo_type: str = "model",
):
    load_env()
    if not _HAS_HF:
        raise ImportError("huggingface_hub is required to upload models.")
    token = os.environ.get("HUGGINGFACE_KEY") or os.environ.get("HF_TOKEN")
    if token:
        hf_login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    f_path = Path(local_path)
    remote_name = filename or f_path.name
    api.upload_file(
        path_or_fileobj=str(f_path),
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {f_path} to https://huggingface.co/{repo_id}/{remote_name}")

def pull_from_hub(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
) -> str:
    load_env()
    if not _HAS_HF:
        raise ImportError("huggingface_hub is required to download models.")
    token = os.environ.get("HUGGINGFACE_KEY") or os.environ.get("HF_TOKEN")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=local_dir,
    )
    return downloaded

