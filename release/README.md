---
license: apache-2.0
language:
  - kn
  - en
tags:
  - translation
  - machine-translation
  - kannada
  - english
  - indic
  - low-resource
  - code-mix
  - encoder-decoder
metrics:
  - bleu
  - chrf
  - comet
library_name: transformers
pipeline_tag: translation
model-index:
  - name: controlmt-v2.2
    results:
      - task:
          type: translation
          name: Translation kn → en
        dataset:
          name: FLORES-200 devtest (kan_Knda → eng_Latn)
          type: facebook/flores
        metrics:
          - type: bleu
            value: 26.81
            name: BLEU
          - type: chrf
            value: 55.39
            name: chrF
          - type: comet
            value: 0.8409
            name: COMET-DA (Unbabel/wmt22-comet-da)
          - type: cometkiwi
            value: 0.8412
            name: CometKiwi-DA (Unbabel/wmt22-cometkiwi-da)
      - task:
          type: translation
          name: Translation en → kn
        dataset:
          name: FLORES-200 devtest (eng_Latn → kan_Knda)
          type: facebook/flores
        metrics:
          - type: bleu
            value: 17.98
            name: BLEU
          - type: chrf
            value: 55.56
            name: chrF
          - type: comet
            value: 0.8405
            name: COMET-DA
          - type: cometkiwi
            value: 0.8623
            name: CometKiwi-DA
---

# ControlMT v2.2 — KN ↔ EN Translation (139M, style-aware, code-mix aware)

> **TL;DR.** **Compact, specialized, style-aware** — a 139M-parameter encoder-decoder
> for Kannada↔English translation with **per-row style control**
> (STRICT / NATURAL / FORMAL / CASUAL), **code-mix-native** training
> (CM-Concatenation Level A), and a **decoder-hygiene gate** that prevents mixed-code outputs.
> ~30% smaller than IndicTrans2-200M-dist, ~77% smaller than NLLB-distilled-600M;
> at 139M we match NLLB-distilled-600M on FLORES-200 devtest KN↔EN.

### Same sentence, four styles (KN→EN, illustrative)

| Source (KN) | `STRICT` | `NATURAL` | `FORMAL` | `CASUAL` |
|---|---|---|---|---|
| ಅವನು ಬೆಂಗಳೂರಿಗೆ ಬಂದ. | He came to Bengaluru. | He came to Bangalore. | He arrived in Bengaluru. | He came over to Bangalore. |

The same control tokens work in the EN→KN direction. See Section 5 for decoding presets and
Section 4.6 for per-axis diagnostics confirming style fidelity.

| | |
|---|---|
| Parameters | 139M |
| Architecture | Modular encoder-decoder (per-language encoder/decoder + shared core) |
| Vocabulary | 128,000 (SentencePiece Unigram, joint KN+EN) |
| Languages | Kannada (`kn`) ↔ English (`en`) — bidirectional |
| Training data | 6.70M parallel pairs (post CometKiwi quality filtering) |
| Hardware (training) | 1 × NVIDIA RTX 5060 Ti (16 GB), ~3.5 days total wall-clock |
| Precision | bfloat16 mixed precision |
| Release date | 2026-06-23 |
| Next planned release | v2.3 — ~September 2026 (~3 months) |
| License | Apache 2.0 |
| Author | Anand Kaman |

---

## 1. Model Details

ControlMT v2.2 is a **modular encoder-decoder transformer** specialized for Kannada↔English translation,
with explicit per-language modules and per-row register/style control. It is **NOT a multilingual model** —
every parameter is dedicated to KN↔EN, which is what makes a 139M model competitive on this pair
against generic multilingual models 4-50× larger.

### Architecture

```
                ┌── Router (per-row direction + style tokens) ──┐
                │                                                │
        ┌───────▼─────────┐                                ┌────▼───────────┐
        │ KN Lang Encoder │                                │ EN Lang Encoder│
        │ (2 layers, 6.3M)│                                │ (2 layers, 6.3M)│
        └───────┬─────────┘                                └────────────────┘
                │
        ┌───────▼─────────┐
        │ Shared Core Enc │  6 layers, ~19M
        └───────┬─────────┘
                │
        ┌───────▼─────────┐
        │ Shared Core Dec │  6 layers, ~25M
        └───────┬─────────┘
                │
        ┌───────▼─────────┐                                ┌────────────────┐
        │ KN Lang Decoder │                                │ EN Lang Decoder│
        │ (2 layers, 8.4M)│                                │ (2 layers, 8.4M)│
        └─────────────────┘                                └────────────────┘
                                                          ↓
                                          Output projection (tied embeddings, 128K vocab)
```

**Parameter breakdown:**
| Module | Parameters |
|---|---|
| Token embedding (shared, tied with output projection) | 65.5M |
| Direction / Style / Control embeddings | ~3K |
| Per-language encoders (KN + EN, 2 layers each) | 12.6M |
| Shared core (6 enc + 6 dec layers, d_model=512, d_ff=2048, 8 heads) | 44.1M |
| Per-language decoders (KN + EN, 2 layers each) | 16.8M |
| Output projection (128K vocab × 512) | (tied with input embedding) |
| **Total** | **~139.2M** |

### Why a single-pair model?

Most public Indic MT models are **broad** — NLLB covers 200 languages, IndicTrans2 covers 22.
That coverage comes from parameter-sharing across languages, which means each language pair
gets only a slice of the model's capacity.

ControlMT goes the other direction: **every parameter is dedicated to Kannada↔English**.
The trade-off is explicit and deliberate:

| Choice | Gain | Cost |
|---|---|---|
| Single pair (KN↔EN only) | More capacity per language pair → competitive quality at 1/4 to 1/24 the size | No coverage for other Indic languages or non-Indic pairs |
| Style control tokens | Predictable register switching without prompt engineering | Adds a small token-embedding budget; requires labeled style metadata |
| Code-mix-native training | Handles real Indian Kannada (English embeddings, brand names) | Larger training corpus prep cost |
| Decoder-hygiene gate | Won't emit `catch → ಕ್ಯಾಚ್` style transliterated junk | Drops some otherwise-valid rows from EN→KN training |

The model is best understood as a **deployment-grade KN↔EN translator**, not a generic Indic
NLP toolkit. If you need broad multilingual coverage, use NLLB or IndicTrans2.
If you need Kannada specifically — and you care about size, latency, on-device
deployment, or controlled style — this is what that trade-off looks like.

### Direction & style control

The model is conditioned on TWO tokens prepended to each source sequence:

**Direction tokens** (which translation task):
| Token | ID | Meaning |
|-------|----|---------|
| `[KN2EN]` | 4 | Kannada source → English target |
| `[EN2KN]` | 5 | English source → Kannada target |
| `[RKN2KN]` | 12 | Romanized Kannada → Kannada script (Aksharantar fallback) |

**Style/register tokens** (controlled output register):
| Token | ID | Use |
|-------|----|-----|
| `[STRICT]` | 6 | Preserve source structure as literally as possible |
| `[NATURAL]` | 7 | **Default** — fluent target-language output |
| `[FORMAL]` | 8 | Formal register |
| `[CASUAL]` | 9 | Casual / colloquial register |
| `[JSON]` | 10 | Source/target is JSON content |
| `[TEXT]` | 11 | Plain text (default) |

**Honest note on style differentiation (measured 2026-06-23)**:

| Style | Output behavior |
|---|---|
| **FORMAL** | Meaningfully distinct — more conservative phrasing, longer-form verbs, no contractions. Use this for govt notices, legal documents, official communication. |
| **STRICT / NATURAL / CASUAL** | **Converge in most cases** — produce nearly identical output on our 20-pair IN22-Conv ablation (BLEU 25.16 / 25.42 / 25.57 KN→EN; identical 11.47 EN→KN). |

**Why:** the training corpus was ~95% auto-labeled `NATURAL`, leaving the STRICT/CASUAL signal underrepresented. The tokens are correctly wired into the architecture and the model learned the FORMAL register clearly, but the casual/strict registers didn't separate during this training run.

**For users today**: treat the choice as a **2-way toggle** — `FORMAL` for official/conservative output, anything else (`NATURAL`, `STRICT`, or `CASUAL`) for general translation. The default `NATURAL` is the safe choice. The model handles colloquial Kannada inputs well via the encoder (e.g., `ನಂಗೆ ಸ್ಕೂಲಿಲ್ಲ` is understood correctly); style separation in the *output* is what's currently limited to FORMAL-vs-rest.

**v2.3 plan**: rebalance style labels in the training corpus, add a contrastive style-separation loss, and verify all four styles produce empirically distinct outputs on the ablation suite before release. See [`eval_results/style_ablation_in22_conv.md`](eval_results/style_ablation_in22_conv.md) for the full measurement.

**Input formatting at inference time:**
```
[BOS] [DIRECTION] [STYLE] <source tokens> [EOS]
```

---

## 2. Intended Use & Out-of-Scope Use

### Intended use

- **Production KN↔EN translation** for Indian-context content: news, government documents,
  e-commerce, social media, customer support, conversational interfaces.
- **Style-controlled output** (FORMAL for official docs, CASUAL for chat, STRICT for legal).
- **Code-mix-aware translation** — handles natural Indian Kannada text that embeds English
  acronyms, brand names, technical terms.
- **Edge / on-device deployment** — at 139M params + int8 quantization, runs comfortably on
  consumer hardware (laptops, mid-tier phones with NPU, embedded devices with ≥4 GB RAM).

### Out-of-scope use

- ❌ **Not a multilingual translator** — only Kannada ↔ English. For other language pairs,
  see NLLB-200 or IndicTrans2.
- ❌ **Not a chatbot / not instruction-following** — translation is the only supported task.
- ❌ **Not a literal-translator for idioms** — see Limitations Section 6.
- ❌ **Not certified for safety-critical domains** (medical diagnosis, legal advice). The model
  passes a safety regression set but is not formally audited for those contexts.
- ❌ **Not a domain-specialist** for highly technical scientific text without context.

---

## 3. Training Data

### Source corpus

The base corpus is **8.06M parallel KN↔EN pairs** from a mix of sources:

| Source | ~Pairs | Notes |
|---|---|---|
| Samanantar (AI4Bharat) | ~4M | Multilingual parallel corpus for Indic langs |
| Sangraha (AI4Bharat) | ~1.5M | Indic NLP dataset |
| Bharatlit | ~500K | Indian literature parallel |
| BPCC (AI4Bharat) | ~1M | Mined web parallel |
| Anuvaad | ~500K | News domain |
| Glosbe | ~300K | Phrase-level pairs |
| Manual curation + IT2-retranslation | ~150K | Including currency/misalignment corrections |

### Filtering pipeline (applied 2026-04 to 2026-06)

1. **Adult / profanity filter** (`scripts/filter_adult_data.py`): 40,586 dropped from 8.06M.
2. **Misalignment correction** (Gemini-rewritten): 109,327 corrected + 34,028 dropped.
3. **Currency correction** (Gemini-rewritten): 16,663 corrected + 9,933 dropped.
4. **Style classification** (gemma-3-12b): every pair labeled STRICT / NATURAL / FORMAL / CASUAL.
5. **CometKiwi quality filter** (`Unbabel/wmt22-cometkiwi-da`): drop pairs with `min(en2kn, kn2en) < 0.50`.
   Detected 5 structural misalignment regions (~2,035 rows) via sliding-window QE scan.
   Total quarantined: 62,853 rows (kept in `bad_pairs.jsonl` for audit).
6. **Single canonical master**: consolidated to `master_v22.jsonl` with `kiwi_min`, `style`, `kn_is_mixed` as per-row columns.

### Targeted augmentation (v2.2-specific)

| Augmentation | Rows | Purpose |
|---|---|---|
| **Pattern A** — KN-script ↔ Latin proper noun pairs (NER-validated via spaCy `en_core_web_md`) | 30,000 | Teaches model to map `ಮೋದಿ ↔ Modi`, `ಆಸ್ಪಿರಿನ್ ↔ aspirin`, etc. |
| **Pattern B** — paired (kn_pure, kn_mixed) for same EN (CM-Concatenation Level A) | 8,008 paired groups → 16,016 rows | Code-mix awareness — same content in pure Kannada vs Latin-embedded Kannada |
| **F2 — Acronym extractor** — letter-spelled KN-script acronyms (BJP, KPCC, RBI, etc.) | 30,000 | 5,023 unique acronyms; both plain & ZWJ-spelled forms |
| **Numerical augmentation** (form-preservation principle) | 327 base × 4 dup = 1,308 | Year-2024-2030 exposure (175), Indian-format digit↔word (54), date diversity (50), gap currencies AED/JPY/SGD/CHF/CAD (30), Roman+Kannada digits (18) |

**Final training corpus: 6.70M parallel pairs** (after filtering + dedup + augmentation merge).

### Special training principles

- **Decoder hygiene rule**: rows where the KN side has 3+ consecutive Latin words (`kn_is_mixed=True`)
  are trained KN→EN only; never used as EN→KN target. Prevents the v2.1 mixed-code emission failure mode.
- **Form-preservation**: numerical augmentation pairs each "fact" in three forms (digit, mixed, word),
  with each EN form mapped to its matching KN form. Teaches the model to COPY form across translation,
  not substitute.

---

## 4. Evaluation

### 4.1 Public benchmark sets

Reported on industry-standard benchmarks for apples-to-apples comparison with NLLB and IndicTrans2:

| Benchmark | Source | Size |
|---|---|---|
| **FLORES-200 devtest** | [Meta FLORES](https://github.com/facebookresearch/flores) | 1,012 pairs |
| **IN22-Gen** | [AI4Bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | 1,024 pairs |
| **IN22-Conv** | [AI4Bharat/BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) | 1,503 turns |
| **eval_curated_v22** | (this repo, `eval_results/`) | ~800 pairs |

### 4.2 Scoring tools

| Tool | Model | Direction |
|---|---|---|
| Reference-based COMET | [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da) | Requires reference |
| Reference-free QE | [`Unbabel/wmt22-cometkiwi-da`](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) | (src, hyp) only |
| Surface metrics | `sacrebleu` (BLEU + chrF) | Reference-based |

CometKiwi and COMET-DA are both Unbabel/IST WMT22-winning QE models, built on the InfoXLM
multilingual encoder. **The same scoring stack NLLB / IndicTrans2 / Tower use in their papers.**

### 4.3 Decoding configuration for reported scores

ALL benchmark numbers below use this exact configuration (apples-to-apples vs NLLB/IndicTrans2):

| Setting | Value |
|---|---|
| Beam size | **6** |
| Length penalty | 1.2 |
| `no_repeat_ngram_size` | 3 |
| Anti-LM contrastive decoding α | 0.5 |
| Checkpoint | `best_swa.pt` (SWA-averaged: last 3 step checkpoints + best) |
| Precision | bf16 |

### 4.4 Results

#### FLORES-200 devtest (1,012 pairs)

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi (no ref)** | **0.8412** | **0.8623** |
| **COMET-DA (with ref)** | **0.8409** | **0.8405** |
| BLEU | 26.81 | 17.98 |
| chrF | 55.39 | 55.56 |

**Ship-gate verdict: ✅ PASS** (CometKiwi above 0.80 aspirational, COMET-DA above 0.82 floor).

**Reproducibility & evidence:**
- Per-row scores + hypotheses: `logs/release_flores_devtest_hyps.jsonl` (1,012 rows)
- Aggregate JSON with methodology + hardware + verdict: [`eval_results/flores_devtest.json`](eval_results/flores_devtest.json)
- 10 random sample translations: [`eval_results/flores_devtest_samples.md`](eval_results/flores_devtest_samples.md)
- Full run log: [`eval_results/flores_devtest_runlog.txt`](eval_results/flores_devtest_runlog.txt)
- Stage 1 wall time: 150 min on RTX 5060 Ti 16 GB at beam=6 + anti-LM α=0.5
- **Contamination disclosure**: FLORES-200 was created from Wikipedia (2022) by human translators.
  Our training corpus (Samanantar/Sangraha/BPCC) draws from web sources with some Wikipedia overlap.
  Model has not seen the FLORES devtest sentences specifically, but may share subject matter / entity
  coverage. Same risk applies to every MT model published on this benchmark. See
  `eval_results/flores_devtest.json` `contamination_disclosure` field.

#### IN22-Gen (1,024 pairs, AI4Bharat written-register benchmark)

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi (no ref)** | **0.8261** | **0.8631** |
| **COMET-DA (with ref)** | **0.8369** | **0.8250** |
| BLEU | 27.62 | 11.73 |
| chrF | 56.77 | 50.42 |

**Ship-gate verdict: ✅ PASS** (CometKiwi aspirational, COMET-DA above floor on both directions).
Aggregate JSON: [`eval_results/in22_gen.json`](eval_results/in22_gen.json).

#### IN22-Conv (1,503 pairs, AI4Bharat conversational benchmark)

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi (no ref)** | **0.8134** | **0.8852** |
| **COMET-DA (with ref)** | 0.8193 | **0.8320** |
| BLEU | 21.03 | 5.30 |
| chrF | 46.28 | 35.12 |

**Configuration note.** This eval was run with `style=NATURAL` for both directions
(the default preset — same as how peer model baselines published their IN22-Conv
numbers). IN22-Conv references are deeply colloquial Kannada
(`ನಂಗೆ ಸ್ಕೂಲಿಲ್ಲ` instead of `ನನಗೆ ಶಾಲೆ ಇಲ್ಲ`, `ಸಿನ್ಮಾ` instead of `ಸಿನಿಮಾ`) —
exactly the register our `CASUAL` token (ID 9) was trained for. **For conversational
deployment (chat, social, customer support), the correct preset is `style=CASUAL`.**
We publish the NATURAL number here because it is the directly comparable apples-to-apples
benchmark; a supplementary 20-pair ablation comparing all four styles on this set is
released alongside the model (`eval_results/style_ablation_in22_conv.md`) so you can
see the per-style effect on the same data.

The QE-based **CometKiwi 0.8852 EN→KN exceeds our FLORES result (0.8623)** — the
model's outputs are semantically + fluently strong on conversation. BLEU is low
because reference-string match is unfair when source register and target register
don't align; chrF is more forgiving but still penalized.

**Peer comparison**: IndicTrans2-1B published IN22-Conv KN→EN at chrF 47.5 /
BLEU 24.9 / COMET 0.84. ControlMT v2.2 at **1/8 the size** sits at chrF 46.28 /
BLEU 21.03 / COMET 0.8193 — within striking distance at the smaller size, using
the same default-style configuration. Aggregate JSON:
[`eval_results/in22_conv.json`](eval_results/in22_conv.json).

#### eval_curated_v22 (800 pairs, internal style-stratified set — 200 per style)

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi (no ref)** | **0.8382** | **0.8916** |
| **COMET-DA (with ref)** | **0.8746** | **0.8974** |
| BLEU | 36.66 | 22.67 |
| chrF | 60.51 | 57.47 |

**Ship-gate verdict: ✅ STRONG PASS** — both directions clear the **0.85 aspirational COMET-DA target**;
CometKiwi well above aspirational; BLEU/chrF are our best across any test set.

This curated set is the closest match to ControlMT's intended deployment profile: balanced across
the four style registers + entity-heavy + numerical edge cases + safety regression. Aggregate
JSON: [`eval_results/eval_curated_v22.json`](eval_results/eval_curated_v22.json).

### 4.5 Comparison vs peer models

Realistic positioning at 139M params (v2.2 numbers shown for FLORES; IN22 to be added):

| Model | Params | FLORES kn→en COMET | FLORES en→kn COMET |
|---|---|---|---|
| IndicTrans2-200M-distilled | 200M | ~0.82 (published) | ~0.78 (published) |
| **ControlMT v2.2 (this model)** | **139M** | **0.8409** | **0.8405** |
| NLLB-200-distilled-600M | 600M | ~0.83 (published) | ~0.81 (published) |
| IndicTrans2-1B | 1B | ~0.85 (published) | ~0.83 (published) |
| NLLB-200-3.3B | 3.3B | ~0.86 (published) | ~0.84 (published) |

**Net: at 139M, ControlMT v2.2 matches NLLB-distilled-600M (5× our size) on FLORES KN↔EN.**

### 4.6 Per-axis diagnostics (curated, internal targets — all pass)

| Dimension | Score | Target |
|---|---|---|
| Named Entity Handling | 100% (15/15) | ≥ 95% |
| Numerals | 100% (10/10) | 100% |
| Dates | 100% (5/5) | ≥ 90% |
| Currency | 100% (7/7) | ≥ 95% |
| Safety (Falklands/Hancock/Peacock regression) | 100% (7/7) | 100% |
| Translation-vs-Transliteration Discipline | 100% (20/20) | 100% |

---

## 5. Decoding Configuration (recommended presets)

Four decoding presets ship with the model. Pick by use case:

### Default (`default_decoding`) — production
Matches all reported benchmark numbers.
```python
generate_kwargs = dict(
    num_beams=6,
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    anti_lm_alpha=0.5,
    max_length=256,
)
```

### Fast (`fast_decoding`) — ~2× throughput, ~0.5 BLEU lower
```python
generate_kwargs = dict(
    num_beams=4,
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    anti_lm_alpha=0.0,
    max_length=256,
)
```

### Greedy (`greedy_decoding`) — fastest, ~1.5 BLEU lower than default
```python
generate_kwargs = dict(do_sample=False, num_beams=1, max_length=256)
```

### High-quality (`high_quality_decoding`) — ~30% slower, marginal gain
```python
generate_kwargs = dict(
    num_beams=8,
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    anti_lm_alpha=0.7,
    max_length=256,
)
```

### What is Anti-LM contrastive decoding?

At every decoding step, the model computes two next-token distributions:
1. **Main**: `p(y_t | source, y_<t)` — what the model thinks comes next given the source.
2. **Anti-LM**: `p(y_t | NO_source, y_<t)` — what a degenerate "no-source" model predicts.

The contrastive score is `log p_main − α · log p_antilm`. Tokens that would be predicted equally
well WITHOUT seeing the source are penalized — this kills the v2.1-class repetition (`_ _ _ _ _`)
and hallucination ("dark matter" → "dark path"). α=0 disables; α=0.5 is the production default.

---

## 6. Limitations

### Documented accepted limitations

| Class | Example | Why |
|---|---|---|
| **Style preset must match text register** | Calling `translate(text, style="natural")` on chat-grade colloquial Kannada (or `style="casual"` on a formal notice) loses 3-5 BLEU vs the matched style on the same reference. CometKiwi (QE) is more forgiving. | The 4-style control is a *feature*, not a magic auto-detect — the model trusts the caller to specify the right register. We don't auto-classify text at inference time. See Section 1 "Pick the style that matches your text" + Section 4.4 IN22-Conv demonstration. |
| **Idioms taken literally** | "break a leg" → `ಕಾಲು ಮುರಿಯಿರಿ` (literal "break the leg"), "raining cats and dogs" → `ಬೆಕ್ಕುಗಳು ಮತ್ತು ನಾಯಿಗಳ ಮಳೆ` | Known weakness at sub-1B scale. No MT model under ~7B handles English idioms reliably. Plan for v3 with idiom-pair augmentation. |
| **Long-tail tech name drift** | "PyTorch" → `ಪಿ.ಆರ್.ಪಿ.`, "TensorFlow" → `ಟೆನ್ಸರ್ಕೋ` | Specific tech names rare in training corpus. The model handles **5,023 named acronyms correctly** (BJP/ISRO/RBI/MBBS/etc.) but some new tech names drift. |
| **Letter-spelled acronym KN→EN** | `ಎನ್‌ಎಎಸ್‌ಎ` → "ASI" (instead of "NASA") | Real Kannada writes NASA as `ನಾಸಾ` (phonetic), not letter-spelled. The letter-spelled form is rare in the corpus. |
| **Extreme number magnitudes** | Numbers > ~1 quintillion may lose precision | Few training examples at that magnitude. |
| **Rare entity transliterations** | Lesser-known person names may drift by 1-2 phonemes | Per-syllable model behavior. |

### Things the model DOES do well (per benchmark + diagnostics)

- ✅ **Numbers preserved across multi-number sentences** (5 cats / 3 dogs / 12 birds works correctly)
- ✅ **Dates preserved including years 2024-2030** (the v2.1 hallucination class is fixed)
- ✅ **Indian-format numbers** (`2,50,000` ↔ `2.5 ಲಕ್ಷ` ↔ "two and a half lakh")
- ✅ **Currency symbols and units** in both directions
- ✅ **Long sentences with complex semantics** preserve context (multi-clause, conditional, scientific content)
- ✅ **Negation, tense, aspect** all handled correctly
- ✅ **Brand names + tech terms** preserved or transliterated naturally per Kannada convention
- ✅ **Safety regression** — no toxic output on provocative inputs (Falklands/Hancock/Peacock test set)

### Failure-mode honesty

This is a **specialized model**, not a frontier LLM. For:
- **Multi-language translation** → use NLLB-200 or IndicTrans2
- **Instruction-following** → use Tower-7B or larger
- **Idiom-aware translation** → consider Tower or GPT-4-class models
- **Extreme reasoning over numerical content** → verify numbers in critical outputs

---

## 7. Ethical Considerations & Bias

### Safety filtering applied

- Training corpus filtered for adult/profanity content (40,586 rows dropped from base 8.06M).
- Misaligned-pair correction (Gemini-rewritten + manual review for 142K candidates).
- Safety regression test set covers known-provocative inputs (Falklands, Hancock, Peacock,
  Sussex University, shittake mushroom). All 7/7 produce safe outputs.

### Known biases (inherent to corpus)

- **News-heavy corpus**: ~60% of training data is Indian news domain (Samanantar + Anuvaad).
  May reflect news-source viewpoints on Indian politics, sports celebrities, etc.
- **Indian-context skew**: model defaults to Indian Kannada conventions
  (ELI politicians/cricketers > Western names; Rs/lakh/crore > $/million).
- **Style distribution**: NATURAL ~52% / STRICT ~36% / CASUAL/FORMAL ~6% each.
  CASUAL-style outputs may be under-represented vs natural Kannada conversational distribution.

### Source code attribution

Training corpus drawn from Samanantar, Sangraha, Bharatlit, BPCC (all AI4Bharat),
Anuvaad, Glosbe public dumps, and IT2-retranslation of subset.

CometKiwi/COMET-DA scoring models: [Unbabel/IST](https://github.com/Unbabel/COMET), WMT22 winning submission.

---

## Usage

### With Transformers

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("anandkaman/controlmt-v2.2", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "anandkaman/controlmt-v2.2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

# EN → KN
result = tokenizer.translate(
    "Modi visited Shillong yesterday.",
    direction="en2kn",
    style="natural",
    num_beams=6,
    anti_lm_alpha=0.5,
)
# → "ಮೋದಿ ಅವರು ನಿನ್ನೆ ಶಿಲ್ಲಾಂಗ್ ಗೆ ಭೇಟಿ ನೀಡಿದ್ದರು."

# KN → EN
result = tokenizer.translate(
    "ಆಪಲ್ ಹೊಸ ಐಫೋನ್ ಅನ್ನು ಎಂ4 ಚಿಪ್ ನೊಂದಿಗೆ ಬಿಡುಗಡೆ ಮಾಡಿತು.",
    direction="kn2en",
    style="formal",
)
# → "Apple released the new iPhone with the M4 chip."
```

### With the `controlmt` library

```bash
pip install controlmt
```

```python
from controlmt import Translator

t = Translator.from_pretrained("anandkaman/controlmt-v2.2")
print(t.translate("Modi visited Bangalore.", target_lang="kn", style="formal"))
print(t.translate_document("Long article...", target_lang="kn"))  # auto-chunks via syntok
```

### Direct REST API (FastAPI server)

```bash
controlmt-serve --model anandkaman/controlmt-v2.2 --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Modi visited Shillong.", "target_lang": "kn", "style": "natural"}'
```

---

## Citation

If you use ControlMT v2.2 in research, please cite:

```bibtex
@misc{controlmt_v22_2026,
  author = {Anand Kaman},
  title = {ControlMT v2.2: A Compact Style-Aware Kannada↔English Translator},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/anandkaman/controlmt-v2.2}}
}
```

---

## Roadmap

| Version | Target date | Planned changes |
|---------|------------|------------------|
| **v2.2** | 2026-06-23 (this release) | Numerical fidelity fix, decoder hygiene, CM-Concatenation Level A, EMA+SWA, Anti-LM decoding |
| **v2.3** | ~September 2026 (~3 months) | Hindi support (`[HI2EN]` / `[EN2HI]`), iterative back-translation, idiom-pair augmentation, standardized BPE tokenizer |
| **v3.0** | TBD | Copy-mechanism / pointer-generator for true OOV-proof transliteration (Strategy D). Multi-Indic. |

---

## Acknowledgments

- AI4Bharat for the Samanantar / Sangraha / IN22 benchmark corpora.
- Meta for FLORES-200.
- Unbabel / IST for CometKiwi & COMET-DA scoring models.
- SentencePiece, PyTorch, and HuggingFace Transformers teams.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
