from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download


DEFAULT_GITHUB_REPO = "https://github.com/GunjanGrunge/Gemma-TIMMY-MLDL-Maths-v5"
DEFAULT_HF_REPO = "Gunjan/Gemma-TIMMY-MLDL-Maths-v5"


def download_github(repo_url: str, dest: Path, ref: str) -> Path:
    repo_url = repo_url.rstrip("/")
    archive_url = f"{repo_url}/archive/refs/heads/{ref}.zip"
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "repo.zip"
        with urlopen(archive_url) as response, archive_path.open("wb") as out_file:
            out_file.write(response.read())
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(tmp_dir)
        extracted = next(path for path in Path(tmp_dir).iterdir() if path.is_dir())
        runtime_path = dest / extracted.name
        if runtime_path.exists():
            shutil.rmtree(runtime_path)
        shutil.copytree(extracted, runtime_path)
        return runtime_path


def download_hf(repo_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    return Path(snapshot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Martha V6 runtime from GitHub or Hugging Face.")
    parser.add_argument("--source", choices=["github", "hf"], required=True)
    parser.add_argument("--dest", type=Path, default=Path("runtime"))
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    args = parser.parse_args()

    if args.source == "github":
        runtime_path = download_github(args.github_repo, args.dest, args.ref)
    else:
        runtime_path = download_hf(args.hf_repo, args.dest)

    print(runtime_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
