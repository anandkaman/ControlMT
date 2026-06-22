# Evaluation Results — Reproducibility & Methodology

This directory contains the **complete evidence** for the benchmark numbers
reported in the model card. Every number has a corresponding artifact here.

## Files (per test set)

| File pattern | Content |
|---|---|
| `<set>.json` | Aggregate scores, methodology, hardware, decoding config, ship-floor verdict |
| `<set>_samples.md` | 10 random sample translations (paired src/hyp/ref for both directions) |
| `<set>_runlog.txt` | Full run log: per-batch progress, stage transitions, model load info |

## Stage pipeline (`scripts/eval_release.py`)

Each test set is processed by a 4-stage sequential pipeline designed to fit 16 GB VRAM:

```
Stage 1: Load ControlMT → translate all (src, ref) pairs in both directions
         → save hypotheses to logs/release_<set>_hyps.jsonl
         → free GPU
         (~150 min on RTX 5060 Ti for 1,000-pair set with beam=6)

Stage 2: Load CometKiwi-DA → score every (src, hyp) → mean → write per-row score
         → free GPU  (~30 sec on this hardware)

Stage 3: Load COMET-DA → score every (src, hyp, ref) → mean → write per-row score
         → free GPU  (~30 sec)

Stage 4: sacrebleu BLEU + chrF (CPU)  (~5 sec)
         → write release_<set>_report.md
```

## Models used for scoring

| Tool | Model | Role |
|---|---|---|
| ControlMT v2.2 | `checkpoints_v22/best_swa.pt` (1.93 GB, SWA-averaged) | Generates hypotheses (Stage 1) |
| CometKiwi-DA | [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) (2.2 GB) | Reference-free QE (Stage 2). Won WMT22 QE shared task. Built on InfoXLM. |
| COMET-DA | [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) (2.2 GB) | Reference-based COMET (Stage 3). Same lineage as CometKiwi. |
| sacrebleu | `sacrebleu` (library) | Standardized BLEU + chrF (Stage 4) |

## Decoding configuration used for all reported scores

Matches the `default_decoding` preset in `config.json`:

| Setting | Value |
|---|---|
| Method | beam_search |
| `num_beams` | 6 |
| `length_penalty` | 1.2 |
| `no_repeat_ngram_size` | 3 |
| `anti_lm_alpha` | 0.5 (contrastive decoding) |
| `max_length` | 256 |
| Precision | bf16 |

This matches NLLB / IndicTrans2 published methodology (beam=5-6, length_penalty=1.0-1.2) —
apples-to-apples comparison.

## Reproducibility

Each `<set>.json` includes a `reproducibility.command` field with the exact
invocation. With same checkpoint, same scoring models, same GPU, the run is
deterministic.

```bash
python scripts/eval_release.py \
    --test final_dataset/eval/flores_devtest.jsonl \
    --ckpt checkpoints_v22/best_swa.pt \
    --beam 6 \
    --anti-lm-alpha 0.5
```

## Data contamination disclosure

Every test set has a `contamination_disclosure` field stating known overlap risk
with the training corpus. **All disclosed.** Not a disqualifier — same risk
applies to every model published on these benchmarks — but transparency matters.

## Per-row data

The complete per-row hypothesis + score data is in `logs/release_<set>_hyps.jsonl`
(not duplicated here to keep release/ compact). Each row:

```json
{
  "src_en": "...", "src_kn": "...",
  "hyp_kn2en": "...", "hyp_en2kn": "...",
  "kiwi_kn2en": 0.84, "kiwi_en2kn": 0.86,
  "comet_kn2en": 0.84, "comet_en2kn": 0.84
}
```

Researchers can re-score with their own metrics or load into Pandas for slice analysis.

## How to add a new test set

```bash
# 1. Place test JSONL in final_dataset/eval/ — each row: {"en": "...", "kn": "..."}
# 2. Run:
python scripts/eval_release.py --test final_dataset/eval/<your_test>.jsonl
# 3. Save the result to release/eval_results/:
cp logs/release_<your_test>_hyps.jsonl release/eval_results/
# 4. Generate the JSON proof file (see flores_devtest.json as template)
# 5. Re-run consolidator:
python scripts/build_release_summary.py
```
