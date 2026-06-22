"""Style-control ablation: translate N samples through all 4 styles in both directions.

Purpose:
    - Demonstrate that the 4 style tokens (STRICT/NATURAL/FORMAL/CASUAL) actually
      change output meaningfully on real data.
    - Identify which style is most reliable for a given test-set type (e.g.
      CASUAL is expected to win on IN22-Conv; FORMAL on Wiki/news).
    - Produce a publishable sample table for the README's "style intro" section.

Usage:
    python scripts/compare_styles.py \
        --test final_dataset/eval/in22_conv.jsonl \
        --n 20 \
        --out logs/style_ablation_in22_conv.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sacrebleu
import torch
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from model import ControlMT  # noqa
from eval_v22 import translate  # noqa

STYLES = ["strict", "natural", "formal", "casual"]


def load_model(ckpt_path: str, tok_path: str, device: torch.device):
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    m = ControlMT(vocab_size=sp.get_piece_size()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m, sp


def run_all(test_path: Path, n: int, ckpt: str, tok: str, beam: int, alpha: float):
    """Returns dict[style] = list of {src_en, src_kn, hyp_kn2en, hyp_en2kn}."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = [json.loads(l) for l in open(test_path)][:n]
    print(f"Loading model + tokenizer...")
    model, sp = load_model(ckpt, tok, device)
    out = {s: [] for s in STYLES}
    t0 = time.time()
    total = len(pairs) * len(STYLES) * 2
    done = 0
    for style in STYLES:
        for p in pairs:
            en, kn = p["en"], p["kn"]
            h_kn2en = translate(model, sp, device, kn, direction="kn2en",
                                style=style, beam_size=beam, anti_lm_alpha=alpha)
            h_en2kn = translate(model, sp, device, en, direction="en2kn",
                                style=style, beam_size=beam, anti_lm_alpha=alpha)
            out[style].append({
                "src_en": en, "src_kn": kn,
                "hyp_kn2en": h_kn2en, "hyp_en2kn": h_en2kn,
            })
            done += 2
            if done % 20 == 0:
                rate = done / (time.time() - t0)
                eta = (total - done) / rate / 60
                print(f"  [{done}/{total}] {rate:.2f} hyps/s  ETA {eta:.1f} min")
    return out, pairs


def score(out_rows, refs_en, refs_kn):
    """Score each (style, direction) with BLEU + chrF."""
    scores = {}
    for s in STYLES:
        rows = out_rows[s]
        hk = [r["hyp_kn2en"] for r in rows]
        he = [r["hyp_en2kn"] for r in rows]
        bleu_k2e = sacrebleu.corpus_bleu(hk, [refs_en]).score
        bleu_e2k = sacrebleu.corpus_bleu(he, [refs_kn]).score
        chrf_k2e = sacrebleu.corpus_chrf(hk, [refs_en]).score
        chrf_e2k = sacrebleu.corpus_chrf(he, [refs_kn]).score
        scores[s] = dict(bleu_kn2en=bleu_k2e, bleu_en2kn=bleu_e2k,
                         chrf_kn2en=chrf_k2e, chrf_en2kn=chrf_e2k)
    return scores


def write_report(out_path: Path, test_name: str, n: int, scores: dict,
                 sample_rows: dict, pairs: list):
    lines = [
        f"# Style-control ablation — {test_name} (n={n})",
        "",
        "Translates the same N pairs through all 4 styles (STRICT / NATURAL / FORMAL / CASUAL)",
        "in both directions. Same model checkpoint, beam, anti-LM α — only the style token changes.",
        "",
        "## Aggregate scores by style",
        "",
        "| Style | BLEU kn→en | BLEU en→kn | chrF kn→en | chrF en→kn |",
        "|-------|-----------:|-----------:|-----------:|-----------:|",
    ]
    for s in STYLES:
        sc = scores[s]
        lines.append(
            f"| **{s.upper()}** | {sc['bleu_kn2en']:.2f} | {sc['bleu_en2kn']:.2f} | "
            f"{sc['chrf_kn2en']:.2f} | {sc['chrf_en2kn']:.2f} |"
        )
    lines += [
        "",
        "## Which style wins per metric",
        "",
    ]
    for metric in ["bleu_kn2en", "bleu_en2kn", "chrf_kn2en", "chrf_en2kn"]:
        best = max(STYLES, key=lambda s: scores[s][metric])
        lines.append(f"- **{metric}**: {best.upper()} ({scores[best][metric]:.2f})")
    lines += [
        "",
        "## Sample outputs (first 5 pairs, all 4 styles)",
        "",
    ]
    for i in range(min(5, n)):
        p = pairs[i]
        lines += [
            f"### Pair {i+1}",
            "",
            f"**EN source**: {p['en']}",
            "",
            f"**KN reference**: {p['kn']}",
            "",
            "| Style | KN→EN output | EN→KN output |",
            "|-------|--------------|--------------|",
        ]
        for s in STYLES:
            row = sample_rows[s][i]
            lines.append(f"| {s.upper()} | {row['hyp_kn2en']} | {row['hyp_en2kn']} |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints_v22" / "best_swa.pt"))
    ap.add_argument("--tokenizer",
                    default=str(PROJECT_ROOT / "model" / "controlmt_v2_tokenizer.model"))
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--anti-lm-alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="Markdown report output path")
    args = ap.parse_args()

    out_rows, pairs = run_all(Path(args.test), args.n, args.ckpt, args.tokenizer,
                               args.beam, args.anti_lm_alpha)
    refs_en = [p["en"] for p in pairs]
    refs_kn = [p["kn"] for p in pairs]
    scores = score(out_rows, refs_en, refs_kn)
    print("\nAGGREGATE SCORES")
    for s in STYLES:
        print(f"  {s.upper():>8} | "
              f"BLEU kn→en {scores[s]['bleu_kn2en']:.2f}  "
              f"BLEU en→kn {scores[s]['bleu_en2kn']:.2f}  "
              f"chrF kn→en {scores[s]['chrf_kn2en']:.2f}  "
              f"chrF en→kn {scores[s]['chrf_en2kn']:.2f}")

    test_name = Path(args.test).stem
    write_report(Path(args.out), test_name, args.n, scores, out_rows, pairs)

    # Also save raw JSON for later analysis
    raw_path = Path(args.out).with_suffix(".json")
    raw = {"scores": scores, "samples": {s: out_rows[s] for s in STYLES}}
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Raw JSON → {raw_path}")


if __name__ == "__main__":
    main()
