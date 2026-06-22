"""Run a single source sentence through all 4 styles in both directions.

For the README's 'Same sentence, four styles' showcase. Produces a small
markdown snippet that can be copy-pasted into release/README.md.

Defaults: a Bengaluru-context sentence with proper noun + verb that the four
styles should each shape differently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from model import ControlMT  # noqa
from eval_v22 import translate  # noqa

STYLES = ["strict", "natural", "formal", "casual"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", default="ಅವನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಬಂದು ನನ್ನನ್ನು ಭೇಟಿಯಾಗುತ್ತಾನೆ.",
                    help="Kannada source for KN→EN demo")
    ap.add_argument("--en", default="Mom, let's go for a movie tomorrow evening.",
                    help="English source for EN→KN demo")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "logs" / "showcase_4styles.md"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints_v22" / "best_swa.pt"))
    ap.add_argument("--tokenizer",
                    default=str(PROJECT_ROOT / "model" / "controlmt_v2_tokenizer.model"))
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--anti-lm-alpha", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model + tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    m = ControlMT(vocab_size=sp.get_piece_size()).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    print("Model loaded.")

    kn_outputs = {}
    en_outputs = {}
    t0 = time.time()
    for s in STYLES:
        kn_outputs[s] = translate(m, sp, device, args.kn, direction="kn2en",
                                  style=s, beam_size=args.beam, anti_lm_alpha=args.anti_lm_alpha)
        en_outputs[s] = translate(m, sp, device, args.en, direction="en2kn",
                                  style=s, beam_size=args.beam, anti_lm_alpha=args.anti_lm_alpha)
        print(f"  {s.upper():>8}  KN→EN: {kn_outputs[s]}")
        print(f"           EN→KN: {en_outputs[s]}")
    print(f"\nTotal: {time.time()-t0:.1f}s")

    lines = [
        "## Same sentence, four styles",
        "",
        f"**KN source:**  {args.kn}",
        "",
        "| Style | KN → EN |",
        "|-------|---------|",
    ]
    for s in STYLES:
        lines.append(f"| `{s.upper()}` | {kn_outputs[s]} |")
    lines += [
        "",
        f"**EN source:**  {args.en}",
        "",
        "| Style | EN → KN |",
        "|-------|---------|",
    ]
    for s in STYLES:
        lines.append(f"| `{s.upper()}` | {en_outputs[s]} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown → {args.out}")


if __name__ == "__main__":
    main()
