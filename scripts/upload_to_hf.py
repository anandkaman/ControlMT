"""Upload the release/ folder to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py --repo anandkaman/controlmt-v2.2 --private
    python scripts/upload_to_hf.py --repo anandkaman/controlmt-v2.2  # public

Prerequisites:
    - HF write token saved at /root/.config/controlmt/hf_write.token
    - release/ folder fully populated:
        * README.md, CHANGELOG.md, LICENSE, config.json
        * configuration_controlmt.py, modeling_controlmt.py, tokenization_controlmt.py, model.py
        * tokenizer.model (SentencePiece file)
        * model.safetensors (converted from best_swa.pt)
        * eval_results/*.json + samples + runlog

The script:
    1. Verifies all required files are present.
    2. Creates the repo (idempotent — uses exist_ok=True).
    3. Uploads everything in release/ via upload_folder().
    4. Prints the public/private URL.
"""

import argparse
import os
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "config.json",
    "configuration_controlmt.py",
    "modeling_controlmt.py",
    "tokenization_controlmt.py",
    "model.py",
    "tokenizer.model",
    "model.safetensors",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = PROJECT_ROOT / "release"
TOKEN_FILE = "/root/.config/controlmt/hf_write.token"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo name, e.g. anandkaman/controlmt-v2.2")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument("--token-file", default=TOKEN_FILE)
    ap.add_argument("--release-dir", default=str(RELEASE_DIR))
    ap.add_argument("--check-only", action="store_true",
                    help="Only check that release/ is complete; don't upload")
    args = ap.parse_args()

    release_dir = Path(args.release_dir)
    if not release_dir.exists():
        print(f"✗ release dir not found: {release_dir}")
        return 1

    # ── Verify required files ──
    print(f"Checking release dir: {release_dir}")
    missing = []
    for f in REQUIRED_FILES:
        p = release_dir / f
        if not p.exists():
            missing.append(f)
        else:
            sz = p.stat().st_size
            print(f"  ✓ {f}  ({sz/1024:.1f} KB)" if sz < 1e6 else f"  ✓ {f}  ({sz/1e9:.2f} GB)")
    if missing:
        print(f"\n✗ MISSING files: {missing}")
        print("Cannot upload. Make sure all required artifacts are in release/.")
        return 1
    print("\n✓ All required files present.")

    if args.check_only:
        print("(--check-only mode; skipping upload)")
        return 0

    # ── Load token ──
    if not os.path.exists(args.token_file):
        print(f"✗ HF token file not found: {args.token_file}")
        return 1
    with open(args.token_file) as f:
        hf_token = f.read().strip()
    print(f"\n✓ Loaded HF write token from {args.token_file}")

    # ── Upload ──
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)
    print(f"\nCreating repo {args.repo} (private={args.private}) ...")
    create_repo(
        repo_id=args.repo,
        private=args.private,
        repo_type="model",
        exist_ok=True,
        token=hf_token,
    )

    print(f"\nUploading {release_dir} → {args.repo} ...")
    api.upload_folder(
        folder_path=str(release_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message="ControlMT v2.2 initial release",
    )

    url_base = "https://huggingface.co"
    print(f"\n✓ Upload complete.")
    print(f"  URL: {url_base}/{args.repo}")
    if args.private:
        print(f"  (private — only you can see it)")
        print(f"  To make public: visit Settings → Make public on the HF page.")


if __name__ == "__main__":
    sys.exit(main() or 0)
