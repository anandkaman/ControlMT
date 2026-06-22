# Changelog

All notable changes to ControlMT will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering follows [Semantic Versioning](https://semver.org/).

---

## [v2.2.0] — 2026-06-23

### TL;DR
Compact KN↔EN translator at 139M params matching NLLB-distilled-600M on FLORES-200 KN↔EN.
All v2.1 known regressions fixed. New: decoder hygiene gate, Anti-LM contrastive decoding,
form-preservation training for numerical fidelity.

### Headline benchmarks (FLORES-200 devtest)
| Metric | KN→EN | EN→KN |
|---|---|---|
| CometKiwi (no ref) | **0.8412** | **0.8623** |
| COMET-DA (with ref) | **0.8409** | **0.8405** |
| BLEU | 26.81 | 17.98 |
| chrF | 55.39 | 55.56 |

(IN22-Gen, IN22-Conv, eval_curated_v22 numbers landing before final ship — see README Section 4.4.)

### Added
- **Decoder hygiene gate** (`kn_is_mixed`): rows with 3+ consecutive Latin words in KN never
  used as EN→KN target. Prevents v2.1's mixed-code emission failures (`catch → ಕ್ಯಾಚ್`).
- **CM-Concatenation Level A** ("lite" code-mix training): paired (kn_pure, kn_mixed) for same EN.
  Loose batch pairing, no architecture change.
- **Anti-LM contrastive decoding** (decode-side): `log p_main − α · log p_antilm` where the anti-LM
  pass uses masked-source. α=0.5 default. Kills the `_ _ _ _` repetition class.
- **EMA model averaging** (train-side): decay=0.999, eval against EMA snapshot, save EMA as best.pt.
- **Stochastic Weight Averaging (SWA)** (post-train): last 3 step ckpts + best.pt averaged → best_swa.pt.
  All reported numbers use best_swa.pt.
- **Pattern A — translit_kn_to_en**: 30,000 KN-script ↔ Latin proper-noun pairs (NER-validated).
  Source-tagged in master_v22, oversample-friendly.
- **Pattern B — cm_paired**: 8,008 paired groups (kn_pure + kn_mixed for same EN).
  Loaded as separate stream during training.
- **F2 — letter-spelled acronym extractor**: 5,023 unique acronyms (BJP, ISRO, RBI, MBBS, etc.).
  Both plain (`ಬಿಜೆಪಿ`) and ZWJ-spelled (`ಎನ್‌ಎಎಸ್‌ಎ`) variants.
- **Numerical augmentation** (form-preservation): 327 base × 4 dup = 1,308 train shots covering
  years 2024-2030, Indian-format digit↔word (`2,50,000 ↔ 2.5 ಲಕ್ಷ ↔ ಎರಡೂವರೆ ಲಕ್ಷ`), date format
  diversity, gap currencies, Roman numerals.
- **Master corpus consolidation**: `master_v22.jsonl` is now the single source of truth with
  `kiwi_min`, `style`, `kn_is_mixed` as per-row columns. No more cross-file `(en, kn)` tuple joins.
- **Bad-pairs quarantine**: 62,853 rows moved to `bad_pairs.jsonl` with `_drop_reason` (low_quality,
  structural_misalignment, suspicious_perfect, no_kiwi_score) — audit trail, not silent deletion.
- **Misalignment-region detection**: sliding-window scan of CometKiwi scores caught 5 structural
  off-by-one regions in the legacy corpus (~2,035 rows) — distinct from per-row noise.
- **Per-axis diagnostic eval refined**: transliteration-aware NER (20-entity map), word-boundary
  translit-bleed (no `ರನ್`-inside-`ಉಸಿರನ್ನು` false positives), all 5 percentage forms accepted,
  digit regex strips trailing punctuation.
- **Release-gate eval pipeline** (`scripts/eval_release.py`): sequential GPU loading
  (ControlMT → save hyps → free → CometKiwi → free → COMET-DA → sacrebleu → report). Fits 16 GB VRAM.

### Fixed
- ✅ `catch` → `ಹಿಡಿಯಿರಿ` (was: `ಕ್ಯಾಚ್` literal transliteration in v2.1)
- ✅ `later` → `ಆಮೇಲೆ` (was: `ಲೇಟರ್`)
- ✅ `super cool` → `ಸೂಪರ್ ಕೂಲ್` (now accepted as colloquial loanword — not flagged as regression)
- ✅ `25th December 2026` → `2026ರ ಡಿಸೆಂಬರ್ 25ರಂದು` (was: hallucinated to 2023 in v2.1)
- ✅ `Rs. 2,50,000` → `2,50,000 ರೂ.` (was: substituted to "one lakh" in smoke)
- ✅ Apple-brand vs apple-fruit context disambiguation now reliable
- ✅ `2,024–2030` years specifically augmented (corpus had only ~50 occurrences of 2026)
- ✅ Repetition bug (`_ _ _ _`) eliminated via Anti-LM α=0.5 + `no_repeat_ngram_size=3`

### Changed
- **Tokenizer unchanged from v2.1** — same SentencePiece Unigram 128K. Standardized BPE
  retraining is planned post-v2.2 (CONTROLMT.md Section 3.1) as part of the library bundle.
- **Training resumed from smoke best.pt** (val=2.36) on enriched corpus → final best.pt val=**2.1916**
  (vs v2.1's 2.38). Improvement: +0.19 perplexity reduction in log-space.
- **Decoding default**: now `num_beams=6` (was 4 in v2.1 inference). Anti-LM α=0.5 enabled by default.

### Removed
- ~~`translit_fallback.jsonl`~~ (50K Aksharantar fallback, 75% common-word contamination — verified)
- ~~`synth_translit_sentence_level v1/v2`~~ (regex fragility + inherited corpus noise)
- ~~Curriculum learning~~ (`CURRICULUM_END = 0`) — v2.1's train/val distribution mismatch
  caused false patience trips. Re-enable only with matched val filter.

### Known limitations (deliberate, accepted)
- Idiomatic English ("break a leg", "raining cats and dogs") translated literally — known weakness at this scale.
- Long-tail tech names (PyTorch, TensorFlow) may transliterate inconsistently.
- Letter-spelled Kannada acronym KN→EN (`ಎನ್‌ಎಎಸ್‌ಎ`) less reliable than phonetic form (`ನಾಸಾ`).
- Extreme number magnitudes (>1 quintillion) untested.

### Migration from v2.1
- `direction_id` and `style_id` now mandatory per-row metadata (was: hardcoded STRICT in v2.0).
- New `kn_is_mixed` field on each row (boolean) — auto-derived from regex if not present.
- Tokenizer is identical — no re-tokenization needed if you have v2.1 tokenized arrays.

### Roadmap
- **v2.3 (~September 2026, ~3 months)**: Hindi support, iterative back-translation,
  idiom-pair augmentation, standardized BPE tokenizer (CONTROLMT.md Section 3.1).
- **v3.0 (TBD)**: Copy-mechanism / pointer-generator for OOV-proof transliteration.

---

## [v2.1.0] — 2026-06-01

### Summary
First production-quality ControlMT release. 4 epochs of training on 6.78M parallel pairs.
COMET 0.85/0.87 on 100-pair code_mix slice. v2.1 had several known regressions
(common-word transliterations, decoder hygiene issues, numerical hallucinations on rare years)
that v2.2 explicitly fixes.

### Highlights
- 128K vocab SentencePiece Unigram tokenizer (Rule Zero audited)
- Per-row style + direction tokens
- bfloat16 mixed precision training
- 4-epoch convergence, val_loss 2.38

---

## [v2.0.0] — 2026-04-30

Initial v2 base training. BLEU 25/18 KN↔EN. Foundation for later improvements.

---

## [v1.0.0] — 2026-03-15

First trained ControlMT model. KN↔EN single-pair. ~106M parameters (smaller embedding).
Initial experiment. Several known bugs (`Falklands → Fucklands` token-fragmentation issue,
mixed-code emissions). Deprecated.
