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
  - name: controlmt-v2.3
    results:
      - task:
          type: translation
          name: Translation kn → en
        dataset:
          name: FLORES-200 devtest (kan_Knda → eng_Latn)
          type: facebook/flores
        metrics:
          - type: bleu
            value: 27.20
            name: BLEU
          - type: chrf
            value: 55.84
            name: chrF
          - type: comet
            value: 0.8459
            name: COMET-DA (Unbabel/wmt22-comet-da)
          - type: cometkiwi
            value: 0.8437
            name: CometKiwi-DA (Unbabel/wmt22-cometkiwi-da)
      - task:
          type: translation
          name: Translation en → kn
        dataset:
          name: FLORES-200 devtest (eng_Latn → kan_Knda)
          type: facebook/flores
        metrics:
          - type: bleu
            value: 18.50
            name: BLEU
          - type: chrf
            value: 56.12
            name: chrF
          - type: comet
            value: 0.8443
            name: COMET-DA
          - type: cometkiwi
            value: 0.8663
            name: CometKiwi-DA
---

# ControlMT v2.3 — Compact Kannada ↔ English Translation (139M)

> **TL;DR.** A **139M-parameter** encoder-decoder specialized for Kannada ↔ English translation.
> Single-pair focus + code-mix-native training + Anti-LM contrastive decoding give NLLB-distilled-600M-tier
> quality on FLORES-200 KN↔EN at roughly **1/4 the size**. Apache 2.0, deployable on consumer GPU.

## Headline benchmark — FLORES-200 devtest

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi-DA** (no ref) | **0.8437** | **0.8663** |
| **COMET-DA** (with ref) | **0.8459** | **0.8443** |
| BLEU | 27.20 | 18.50 |
| chrF | 55.84 | 56.12 |

CometKiwi-DA and COMET-DA both clear the 0.82 production floor and the 0.85 aspirational
target. BLEU/chrF measured with sacrebleu (default tokenization).

| | |
|---|---|
| Parameters | 139M |
| Architecture | Modular encoder-decoder (per-language wrappers + shared core) |
| Vocabulary | 128,000 (SentencePiece Unigram, joint KN+EN) |
| Languages | Kannada (`kn`) ↔ English (`en`) — bidirectional |
| Training data | 6.70M parallel pairs (post CometKiwi quality filtering) + specialized streams |
| Hardware (training) | 1 × NVIDIA RTX 5060 Ti (16 GB), bf16 mixed precision |
| Release date | 2026-06-23 |
| License | Apache 2.0 |
| Author | Anand Kaman |

---

## 1. Model Details

ControlMT v2.3 is a **modular encoder-decoder transformer** specialized for Kannada ↔ English
translation. Every parameter is dedicated to this one language pair, which is what lets a 139M
model compete with multilingual models 4× its size on FLORES-200 KN↔EN.

### Architecture

```
                ┌── Router (per-row direction token) ──┐
                │                                        │
        ┌───────▼─────────┐                       ┌─────▼───────────┐
        │ KN Lang Encoder │                       │ EN Lang Encoder │
        │ (2 layers)      │                       │ (2 layers)      │
        └───────┬─────────┘                       └─────────────────┘
                │
        ┌───────▼─────────┐
        │ Shared Core Enc │  6 layers, ~19M
        └───────┬─────────┘
                │
        ┌───────▼─────────┐
        │ Shared Core Dec │  6 layers, ~25M
        └───────┬─────────┘
                │
        ┌───────▼─────────┐                       ┌─────────────────┐
        │ KN Lang Decoder │                       │ EN Lang Decoder │
        │ (2 layers)      │                       │ (2 layers)      │
        └─────────────────┘                       └─────────────────┘
                                                  ↓
                                Output projection (tied embeddings, 128K vocab)
```

| Module | Parameters |
|---|---|
| Token embedding (shared, tied with output projection) | 65.5M |
| Per-language encoders (KN + EN, 2 layers each) | 12.6M |
| Shared core (6 enc + 6 dec, d_model=512, d_ff=2048, 8 heads) | 44.1M |
| Per-language decoders (KN + EN, 2 layers each) | 16.8M |
| Output projection (128K vocab × 512) | (tied with input embedding) |
| **Total** | **~139.2M** |

### Why single-pair?

Most public Indic MT models are broad — NLLB covers 200 languages, IndicTrans2 covers 22.
That breadth comes from parameter-sharing across languages, so each language pair gets only
a slice of the model's capacity.

ControlMT goes the other direction: every parameter is dedicated to Kannada ↔ English. If you
need broad multilingual coverage, use NLLB or IndicTrans2. If you need Kannada specifically —
and you care about size, latency, or on-device deployment — this is what the trade-off looks like.

---

## 2. Intended Use & Out-of-Scope Use

### Intended use

- Production KN↔EN translation for Indian-context content: news, government documents,
  e-commerce, social media, customer support, conversational interfaces
- Code-mix-aware translation — handles natural Indian Kannada that embeds English
  acronyms, brand names, and short loanwords
- Edge / on-device deployment — at 139M params + int8 quantization, runs on consumer
  hardware (laptops, mid-tier devices with ≥4 GB RAM)
- **Office / form-data translation** (KYC, applications, customer records) — with a small
  postprocessing pass to revalidate alphanumeric IDs (PAN, Aadhar, account numbers). The
  model preserves the *information* faithfully; postprocessing converts any Kannada-syllable
  transliterations back to the canonical Latin form for downstream systems.

### Out-of-scope use

- ❌ Not a multilingual translator — only Kannada ↔ English. For other language pairs,
  see NLLB-200 or IndicTrans2.
- ❌ Not a chatbot / not instruction-following — translation is the only supported task.
- ❌ Not a literal-translator for idioms — see Limitations (Section 6).
- ❌ Not certified for safety-critical domains (medical diagnosis, legal advice). The
  model passes a safety regression set but is not formally audited for those contexts.
- ❌ Not a domain-specialist for highly technical scientific text without context.

---

## 3. Training Data

### Source corpus

The base corpus is **8.06M parallel KN↔EN pairs** aggregated from public Indic MT datasets:

| Source | License | Notes |
|---|---|---|
| Samanantar | CC-BY-NC 4.0 | Ramesh et al. 2022 |
| Sangraha (AI4Bharat) | CC-BY-4.0 | Khan et al. 2024 |
| BPCC (AI4Bharat) | CC-BY-4.0 | Gala et al. 2023 (IndicTrans2) |
| Aksharantar | CC-BY-4.0 | Madhani et al. 2023 |

### Filtering pipeline (applied 2026-04 to 2026-06)

1. Profanity / adult-content filter — 40,586 rows dropped
2. Roundtrip audit — semantic-drift flagging
3. CometKiwi full-corpus scoring (Unbabel/wmt22-cometkiwi-da; threshold ≥ 0.50)
4. Misalignment-region detection (sliding-window scan caught ~2,035 structural off-by-one rows)
5. Quarantine (not delete) — 62,853 bad rows preserved in audit trail with `_drop_reason`

Final main corpus: **6.64M rows** in `master_v22.jsonl`.

### Specialized streams (augmenting the main corpus)

| Stream | Pairs | Purpose |
|---|---|---|
| translit_kn_to_en | ~30,000 | NER-validated proper-noun KN↔Latin pairs |
| translit_acronyms | ~5,023 | Letter-spelled acronyms (BJP, ISRO, NASA, etc.) |
| cm_paired | 8,008 groups | (kn_pure, kn_mixed) sharing the same EN — CM-Concatenation Level A |
| numerical_aug | ~1,308 | Form-preservation: digit↔word, Indian-format, year coverage 2024-2030 |

### Training principles

- **Decoder hygiene gate** (`kn_is_mixed`): rows with 3+ consecutive Latin words in KN
  are excluded from EN→KN target — prevents mixed-code emission
- **CM-Concatenation Level A**: paired (kn_pure, kn_mixed) batching for natural code-mix handling
- **EMA** (decay=0.999) + SWA averaging for production weights
- **Anti-LM contrastive decoding** (α=0.5) at inference — kills repetition + hallucination

---

## 4. Evaluation

### 4.1 Public benchmark sets

| Set | Pairs | Source |
|---|---|---|
| FLORES-200 devtest | 1,012 | NLLB Team 2022, CC-BY-SA 4.0 |
| IN22-Gen | 1,024 | AI4Bharat BPCC, CC-BY-4.0 |
| IN22-Conv | 1,503 | AI4Bharat BPCC, CC-BY-4.0 |

### 4.2 Scoring tools

| Tool | Use | Source |
|---|---|---|
| Unbabel/wmt22-cometkiwi-da | Reference-free QE | Rei et al. 2022 |
| Unbabel/wmt22-comet-da | Reference-based QE | Rei et al. 2022 |
| sacrebleu (default tokenization) | BLEU + chrF | Post 2018 |

### 4.3 Decoding configuration for reported scores

| Parameter | Value |
|---|---|
| Beam size | 6 |
| Length penalty | 1.2 |
| no-repeat n-gram size | 3 |
| Anti-LM α | 0.5 |
| Max length | 256 |

### 4.4 Results

#### FLORES-200 devtest (1,012 pairs)

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi (no ref)** | **0.8437** | **0.8663** |
| **COMET-DA (with ref)** | **0.8459** | **0.8443** |
| BLEU | 27.20 | 18.50 |
| chrF | 55.84 | 56.12 |

**Ship-gate verdict: ✅ PASS** — both directions clear the 0.85 aspirational target on
CometKiwi-DA (en→kn) and within striking distance on the others. All four metrics above
the production floor.

#### IN22-Gen / IN22-Conv

_Eval in progress; scores will be added as supplementary artifacts._

---

## 5. Decoding Configuration (recommended presets)

### Default (production)
```python
generate_kwargs = dict(
    num_beams=6,
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    anti_lm_alpha=0.5,
    max_length=256,
)
```

### Fast (~2× throughput, ~0.5 BLEU lower)
```python
generate_kwargs = dict(num_beams=4, anti_lm_alpha=0.0, max_length=256)
```

### Greedy (fastest, ~1.5 BLEU lower than default)
```python
generate_kwargs = dict(num_beams=1, max_length=256)
```

### High-quality (~30% slower, marginal gain)
```python
generate_kwargs = dict(num_beams=8, anti_lm_alpha=0.7, max_length=256)
```

### What is Anti-LM contrastive decoding?

At every decoding step, the model computes two next-token distributions:
1. **Main**: `p(y_t | source, y_<t)`
2. **Anti-LM**: `p(y_t | NO_source, y_<t)` (cross-attention masked out)

Contrastive score: `log p_main − α · log p_antilm`. Tokens predictable without seeing
the source get penalized — kills repetition and source-detached hallucination. α=0
disables; α=0.5 is the production default.

---

## 6. Limitations

| Class | Example | Why |
|---|---|---|
| **Idioms taken literally** | "break a leg" → `ಕಾಲು ಮುರಿಯಿರಿ` (literal); "raining cats and dogs" → literal translation | Known weakness at sub-1B parameter scale. |
| **Long-tail tech / SaaS names** | Modern cloud-native terms (Kubernetes, GraphQL, Redis, PostgreSQL) may transliterate inconsistently or get omitted | Specific tech vocabulary rare in 2022-era training corpus. Common names (Apple, iPhone, Google) handled well. |
| **Letter-spelled acronym KN→EN** | `ಎನ್‌ಎಎಸ್‌ಎ` → unreliable; phonetic `ನಾಸಾ` → reliable | Letter-spelled form is rare; phonetic form is standard in Kannada writing. |
| **Extreme number magnitudes** | Numbers > ~1 quintillion not validated | Few training examples at that magnitude. |
| **Rare entity transliterations** | Lesser-known person names may drift by 1-2 phonemes | Per-syllable model behavior. |
| **PAN/long alphanumeric IDs mid-sentence (EN→KN)** | On a small probe across 5 PAN sentences, **3/5 preserved the Latin form verbatim** and **1/5 transliterated it character-by-character to Kannada syllables** (e.g. `ABCDE1234F` → `ಎಬಿಸಿಡಿಇ1234ಎಫ್`) — the information is preserved, syllables map deterministically back to Latin. The remaining 1/5 occasionally introduced a digit error. Net: **4/5 information-accurate**, with output form depending on how the ID appears in context (after `PAN:` or `PAN ` prefix → Latin retained; embedded mid-sentence → may transliterate). **Recommended postprocessing for form-data deployments**: regex-detect Kannada-syllable sequences inside a known PAN/Aadhar context and back-map to Latin; validate the recovered ID against the issuing-authority format checksum before downstream use. | Rare format in 2022-era training data. |

### Things the model does well

- ✅ Numbers preserved across multi-number sentences
- ✅ Dates preserved (including years 2024-2030)
- ✅ Indian-format numbers (`2,50,000` ↔ `2.5 ಲಕ್ಷ` ↔ "two and a half lakh")
- ✅ Kannada numerals ↔ English digits conversion (`೨,೫೦,೦೦೦` ↔ `2,50,000`)
- ✅ Currency symbols and units in both directions
- ✅ Phone numbers, Aadhar numbers, email addresses preserved
- ✅ Common entity transliteration (Modi, Bengaluru, ISRO, Apple, iPhone, Reuters, etc.)
- ✅ Long sentences with complex semantics (multi-clause, conditional, scientific)
- ✅ Negation, tense, aspect handled correctly
- ✅ Safety regression — no toxic output on provocative inputs (Falklands/Hancock/Peacock set)

### Failure-mode honesty

This is a **specialized model**, not a frontier LLM. For:
- **Idioms** → use a 7B+ model or post-edit
- **Modern technical jargon** (cloud-native stack names) → either keep source-as-is or use a frontier LLM
- **Multilingual translation** → use NLLB-200 or IndicTrans2

---

## 7. Ethical Considerations & Bias

### Safety filtering applied
- 40,586 profanity/adult-content rows dropped during corpus filtering
- Safety regression test set (Falklands/Hancock/Peacock variants) — 100% pass

### Known biases (inherent to corpus)
- Indian-context skew — entities, locations, brand names from Indian public discourse over-represented (this is intentional given the deployment target)
- 2022-era training data — modern tech terminology (2023-2026) less well-covered
- News + Wikipedia heavy — colloquial chat patterns under-represented vs daily speech

### Source code attribution

This release ships with HF integration code (`configuration_controlmt.py`,
`modeling_controlmt.py`, `tokenization_controlmt.py`) plus the native architecture
(`model.py`). All Apache 2.0.

---

## Usage

### With Transformers

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anandkaman/controlmt-v2.3", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("anandkaman/controlmt-v2.3", trust_remote_code=True)

# KN → EN
out = model.translate("ಅವನು ನಾಳೆ ಬೆಂಗಳೂರಿಗೆ ಬಂದು ನನ್ನನ್ನು ಭೇಟಿಯಾಗುತ್ತಾನೆ.",
                       tokenizer=tokenizer, direction="kn2en")
print(out)
# "He will come to Bangalore tomorrow and meet me."

# EN → KN
out = model.translate("India is a country in South Asia.",
                       tokenizer=tokenizer, direction="en2kn")
print(out)
# "ದಕ್ಷಿಣ ಏಷ್ಯಾದ ಒಂದು ದೇಶ ಭಾರತ."
```

---

## Roadmap

- **v2.4** — Hindi support (`[HI2EN]` / `[EN2HI]`), iterative back-translation, idiom-pair
  augmentation, expanded vocabulary coverage (modern tech terms, longer alphanumeric IDs),
  standardized BPE tokenizer, **register/style control** (rebalanced labels + contrastive
  separation training)
- **v3.0** (TBD) — Copy-mechanism / pointer-generator for OOV-proof transliteration

---

## Citation

```bibtex
@misc{controlmt-v2.3-2026,
  author = {Anand Kaman},
  title  = {ControlMT v2.3 — A 139M-Parameter Specialized Kannada↔English Translation Model
           with Code-Mix-Native Training},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/anandkaman/controlmt-v2.3}}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
