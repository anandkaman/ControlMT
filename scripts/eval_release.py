"""Release-gate eval — one test set at a time.

Per CONTROLMT.md §10.1, sequential GPU loading for 16 GB VRAM:
  1. Load ControlMT → translate all (src, ref) pairs in both directions →
     save hypotheses + free GPU.
  2. Load CometKiwi → score every (src, hyp) → mean → free.
  3. Load COMET-DA → score every (src, hyp, ref) → mean → free.
  4. sacrebleu BLEU + chrF (CPU).
  5. Show samples + write report.

Designed to be RUN ONCE PER TEST SET so user can monitor quality and pause
between sets. Each invocation processes ONE file.

Usage:
  python scripts/eval_release.py --test final_dataset/eval/flores_devtest.jsonl
  python scripts/eval_release.py --test final_dataset/eval/in22_gen.jsonl
  ... etc.

  --ckpt    default: checkpoints_v22/best_swa.pt (per release decision)
  --beam    default: 6 (per CONTROLMT.md §10.1)
  --anti-lm-alpha 0.5
  --sample-size N    how many translations to sample (use a smaller number to
                     iterate faster while monitoring quality)
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ─────────────────────────────────────────────────────────────────────────
# STAGE 1: ControlMT translation
# ─────────────────────────────────────────────────────────────────────────

def stage1_translate(test_path, ckpt_path, tokenizer_path, beam, anti_lm_alpha, limit, hyp_out,
                     style_kn_en="natural", style_en_kn="natural"):
    from model import ControlMT
    import sentencepiece as spm
    from eval_v22 import translate

    print(f"\n{'='*70}\nSTAGE 1: ControlMT translation\n{'='*70}")
    print(f"  ckpt: {ckpt_path}")
    print(f"  beam={beam}, anti_lm_alpha={anti_lm_alpha}")
    print(f"  style kn→en: {style_kn_en}   style en→kn: {style_en_kn}")

    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    model = ControlMT(vocab_size=sp.get_piece_size()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"  loaded model | val={state.get('val_loss')}")

    # Load test pairs
    pairs = [json.loads(l) for l in open(test_path)]
    if limit > 0:
        pairs = pairs[:limit]
    print(f"  test pairs: {len(pairs)}")

    print(f"  generating {len(pairs)*2} hypotheses (both directions)...")
    t0 = time.time()
    rows = []
    for i, p in enumerate(pairs):
        en = p["en"]
        kn = p["kn"]
        hyp_kn2en = translate(model, sp, device, kn, direction="kn2en", style=style_kn_en,
                              beam_size=beam, anti_lm_alpha=anti_lm_alpha)
        hyp_en2kn = translate(model, sp, device, en, direction="en2kn", style=style_en_kn,
                              beam_size=beam, anti_lm_alpha=anti_lm_alpha)
        rows.append({
            "src_en": en, "src_kn": kn,
            "hyp_kn2en": hyp_kn2en, "hyp_en2kn": hyp_en2kn,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate / 60
            print(f"    [{i+1:>5}/{len(pairs)}]  {rate:.2f} pairs/s  ETA {eta:.1f}min")

    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min  ({len(pairs)/elapsed:.2f} pairs/s)")

    # Write hyps
    with open(hyp_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved → {hyp_out}")

    # Free GPU
    del model, state, sp
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Sample display — quality eyeball
# ─────────────────────────────────────────────────────────────────────────

def show_samples(rows, n=15, label=""):
    import random
    random.seed(42)
    sample = random.sample(rows, min(n, len(rows)))
    print(f"\n{'='*70}\nQUALITY SAMPLE — {label} ({n} random)\n{'='*70}")
    for i, r in enumerate(sample, 1):
        print(f"\n[{i}]")
        print(f"  KN_src:  {r['src_kn'][:130]}")
        print(f"  KN→EN:   {r['hyp_kn2en'][:130]}")
        print(f"  EN_ref:  {r['src_en'][:130]}")
        print(f"  ---")
        print(f"  EN_src:  {r['src_en'][:130]}")
        print(f"  EN→KN:   {r['hyp_en2kn'][:130]}")
        print(f"  KN_ref:  {r['src_kn'][:130]}")


# ─────────────────────────────────────────────────────────────────────────
# STAGE 2: CometKiwi (reference-free)
# ─────────────────────────────────────────────────────────────────────────

def stage2_cometkiwi(rows):
    print(f"\n{'='*70}\nSTAGE 2: CometKiwi (reference-free)\n{'='*70}")
    from comet import download_model, load_from_checkpoint
    # CometKiwi expects {src, mt}
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    # Build batches: each row contributes 2 (one per direction)
    data_kn2en = [{"src": r["src_kn"], "mt": r["hyp_kn2en"]} for r in rows]
    data_en2kn = [{"src": r["src_en"], "mt": r["hyp_en2kn"]} for r in rows]
    print(f"  scoring kn→en ({len(data_kn2en)})...")
    out_k2e = model.predict(data_kn2en, batch_size=16, gpus=1)
    print(f"  scoring en→kn ({len(data_en2kn)})...")
    out_e2k = model.predict(data_en2kn, batch_size=16, gpus=1)
    mean_k2e = sum(out_k2e["scores"]) / len(out_k2e["scores"])
    mean_e2k = sum(out_e2k["scores"]) / len(out_e2k["scores"])
    print(f"  CometKiwi mean kn→en: {mean_k2e:.4f}")
    print(f"  CometKiwi mean en→kn: {mean_e2k:.4f}")
    # Attach per-row scores
    for r, s in zip(rows, out_k2e["scores"]):
        r["kiwi_kn2en"] = float(s)
    for r, s in zip(rows, out_e2k["scores"]):
        r["kiwi_en2kn"] = float(s)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return mean_k2e, mean_e2k


# ─────────────────────────────────────────────────────────────────────────
# STAGE 3: COMET-DA (reference-based)
# ─────────────────────────────────────────────────────────────────────────

def stage3_comet(rows):
    print(f"\n{'='*70}\nSTAGE 3: COMET-DA (reference-based)\n{'='*70}")
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    # COMET-DA expects {src, mt, ref}
    data_kn2en = [{"src": r["src_kn"], "mt": r["hyp_kn2en"], "ref": r["src_en"]} for r in rows]
    data_en2kn = [{"src": r["src_en"], "mt": r["hyp_en2kn"], "ref": r["src_kn"]} for r in rows]
    print(f"  scoring kn→en ({len(data_kn2en)})...")
    out_k2e = model.predict(data_kn2en, batch_size=16, gpus=1)
    print(f"  scoring en→kn ({len(data_en2kn)})...")
    out_e2k = model.predict(data_en2kn, batch_size=16, gpus=1)
    mean_k2e = sum(out_k2e["scores"]) / len(out_k2e["scores"])
    mean_e2k = sum(out_e2k["scores"]) / len(out_e2k["scores"])
    print(f"  COMET-DA mean kn→en: {mean_k2e:.4f}")
    print(f"  COMET-DA mean en→kn: {mean_e2k:.4f}")
    for r, s in zip(rows, out_k2e["scores"]):
        r["comet_kn2en"] = float(s)
    for r, s in zip(rows, out_e2k["scores"]):
        r["comet_en2kn"] = float(s)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return mean_k2e, mean_e2k


# ─────────────────────────────────────────────────────────────────────────
# STAGE 4: sacrebleu BLEU + chrF (CPU)
# ─────────────────────────────────────────────────────────────────────────

def stage4_sacrebleu(rows):
    print(f"\n{'='*70}\nSTAGE 4: sacrebleu BLEU + chrF\n{'='*70}")
    import sacrebleu
    refs_en = [r["src_en"] for r in rows]
    hyps_kn2en = [r["hyp_kn2en"] for r in rows]
    refs_kn = [r["src_kn"] for r in rows]
    hyps_en2kn = [r["hyp_en2kn"] for r in rows]
    bleu_k2e = sacrebleu.corpus_bleu(hyps_kn2en, [refs_en])
    bleu_e2k = sacrebleu.corpus_bleu(hyps_en2kn, [refs_kn])
    chrf_k2e = sacrebleu.corpus_chrf(hyps_kn2en, [refs_en])
    chrf_e2k = sacrebleu.corpus_chrf(hyps_en2kn, [refs_kn])
    print(f"  BLEU kn→en: {bleu_k2e.score:.2f}")
    print(f"  BLEU en→kn: {bleu_e2k.score:.2f}")
    print(f"  chrF kn→en: {chrf_k2e.score:.2f}")
    print(f"  chrF en→kn: {chrf_e2k.score:.2f}")
    return {
        "bleu_kn2en": bleu_k2e.score, "bleu_en2kn": bleu_e2k.score,
        "chrf_kn2en": chrf_k2e.score, "chrf_en2kn": chrf_e2k.score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Path to test JSONL with en/kn pairs")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints_v22" / "best_swa.pt"))
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "model" / "controlmt_v2_tokenizer.model"))
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--anti-lm-alpha", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="Limit test pairs (0=all)")
    ap.add_argument("--sample-size", type=int, default=15)
    ap.add_argument("--skip-comet", action="store_true", help="Skip stage 3 if you only want quick CometKiwi")
    ap.add_argument("--skip-stage1", action="store_true",
                    help="Skip translation (re-use cached hyps from prior run)")
    ap.add_argument("--style-kn-en", default="natural",
                    choices=["strict", "natural", "formal", "casual"],
                    help="Style preset for KN→EN translation (default: natural)")
    ap.add_argument("--style-en-kn", default="natural",
                    choices=["strict", "natural", "formal", "casual"],
                    help="Style preset for EN→KN translation (default: natural)")
    ap.add_argument("--output-suffix", default="",
                    help="Suffix appended to output filenames (use to distinguish "
                         "multiple runs of the same test set, e.g. '_casual')")
    args = ap.parse_args()

    test_name = Path(args.test).stem + args.output_suffix
    hyp_out = PROJECT_ROOT / "logs" / f"release_{test_name}_hyps.jsonl"
    hyp_out.parent.mkdir(exist_ok=True)
    report_out = PROJECT_ROOT / "logs" / f"release_{test_name}_report.md"

    if args.skip_stage1 and hyp_out.exists():
        rows = [json.loads(l) for l in open(hyp_out)]
        print(f"Reusing cached hypotheses from {hyp_out} ({len(rows)} rows)")
    else:
        rows = stage1_translate(args.test, args.ckpt, args.tokenizer,
                                args.beam, args.anti_lm_alpha, args.limit, hyp_out,
                                style_kn_en=args.style_kn_en,
                                style_en_kn=args.style_en_kn)

    # Show samples — user pause point
    show_samples(rows, n=args.sample_size, label=test_name)

    # Stage 2: CometKiwi
    kiwi_k2e, kiwi_e2k = stage2_cometkiwi(rows)

    # Stage 3: COMET-DA
    if not args.skip_comet:
        comet_k2e, comet_e2k = stage3_comet(rows)
    else:
        comet_k2e = comet_e2k = None

    # Stage 4: sacrebleu
    sb = stage4_sacrebleu(rows)

    # Write per-row scores
    with open(hyp_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final report
    lines = [
        f"# Release Eval Report — {test_name}",
        f"",
        f"- ckpt: `{args.ckpt}`",
        f"- beam: {args.beam} | anti_lm_alpha: {args.anti_lm_alpha}",
        f"- style kn→en: **{args.style_kn_en}** | style en→kn: **{args.style_en_kn}**",
        f"- test pairs: {len(rows)}",
        f"",
        f"## Aggregate scores",
        f"",
        f"| Metric | KN→EN | EN→KN |",
        f"|--------|-------|-------|",
        f"| CometKiwi (no ref) | **{kiwi_k2e:.4f}** | **{kiwi_e2k:.4f}** |",
    ]
    if comet_k2e is not None:
        lines.append(f"| COMET-DA (with ref) | **{comet_k2e:.4f}** | **{comet_e2k:.4f}** |")
    lines.extend([
        f"| BLEU | **{sb['bleu_kn2en']:.2f}** | **{sb['bleu_en2kn']:.2f}** |",
        f"| chrF | **{sb['chrf_kn2en']:.2f}** | **{sb['chrf_en2kn']:.2f}** |",
        f"",
        f"## Targets (CONTROLMT.md §10.1)",
        f"",
        f"- COMET-DA ship floor ≥ **0.82** / aspirational 0.85",
        f"- CometKiwi ship floor ≥ **0.75** / aspirational 0.80",
    ])
    report_out.write_text("\n".join(lines))
    print(f"\nReport → {report_out}")


if __name__ == "__main__":
    main()
