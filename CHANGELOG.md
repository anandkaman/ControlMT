# Changelog

All notable changes to ControlMT will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering follows [Semantic Versioning](https://semver.org/).

---

## [v2.3.0] — 2026-06-23

### TL;DR
**Compact 139M-parameter KN↔EN translator** — focused single-pair training on the
v2.2 enriched corpus + specialized streams (transliteration pairs, code-mix paired
groups, letter-spelled acronyms, numerical augmentation). Anti-LM contrastive decoding,
EMA + SWA averaging.

### Headline benchmarks (FLORES-200 devtest)

| Metric | KN→EN | EN→KN |
|---|---|---|
| CometKiwi (no ref) | **0.8437** | **0.8663** |
| COMET-DA (with ref) | **0.8459** | **0.8443** |
| BLEU | 27.20 | 18.50 |
| chrF | 55.84 | 56.12 |

### Added
- **Refocused single-register training** — all 139M parameters dedicated to
  high-quality KN↔EN translation
- **Improved transliteration consistency** on common entities (Modi, Bengaluru, ISRO,
  Apple, iPhone, etc.)
- **Mixed-script numeral handling** — `೦-೯` Kannada numerals convert reliably to
  English digits in KN→EN direction
- **Cleaner inference API** — `model.translate(text, tokenizer, direction)`; no
  extra style/register surface

### Fixed
- ✅ Improved naturalness on register-appropriate phrasing (commute → ಪ್ರಯಾಣ vs
  ಸಂಚಾರ; finish → ಮುಗಿಸಿದರೆ vs ಪೂರ್ಣಗೊಳಿಸಿದರೆ)
- ✅ Better idiomatic constructions ("despite the rain" → ಮಳೆಯ ಹೊರತಾಗಿಯೂ)
- ✅ More natural sport-context vocabulary (cricket victories use ಭರ್ಜರಿ ಜಯ)

### Changed
- **Training**: warm-start fine-tune from v2.2 final weights with very low LR
  (1.5e-5 → 1e-5) — preserved all v2.2 strengths and added incremental gains
- **Decoding default**: `num_beams=6`, anti-LM α=0.5 (same as v2.2)
- **Tokenizer**: unchanged from v2.2 (SentencePiece Unigram 128K)

### Known limitations (deliberate, accepted)
- Idiomatic English ("break a leg", "raining cats and dogs") translated literally
- Modern SaaS / cloud-native tech names (Kubernetes, GraphQL, Redis, PostgreSQL)
  may transliterate inconsistently or get omitted — training corpus pre-dates
  much of this vocabulary
- 10-character alphanumeric PAN numbers embedded mid-sentence without
  demarcation can occasionally transliterate; with `PAN:` or `PAN ` prefix
  the preservation is reliable
- Letter-spelled Kannada acronym KN→EN (`ಎನ್‌ಎಎಸ್‌ಎ`) less reliable than
  phonetic form (`ನಾಸಾ`)
- Extreme number magnitudes (> ~1 quintillion) untested

### Roadmap
- **v2.4** — Hindi support (`[HI2EN]` / `[EN2HI]`), iterative back-translation,
  idiom-pair augmentation, expanded vocabulary (modern tech, long alphanumeric IDs),
  standardized BPE tokenizer, register/style control (rebalanced labels + contrastive
  separation training)
- **v3.0** (TBD) — Copy-mechanism / pointer-generator for OOV-proof transliteration

---

## [v2.2.0] — internal milestone (not released publicly)

Multi-register training run with style-prefix tokens (STRICT/NATURAL/FORMAL/CASUAL).
Internal eval showed register separation didn't generalize cleanly at the 139M scale,
so the next release (v2.3) consolidated capacity into single-register training.
Kept as internal reference; not uploaded to public HuggingFace.

---

## [v2.0.0] — 2026-04-30

Initial v2 base training. BLEU 25/18 KN↔EN. Foundation for later improvements.

---

## [v1.0.0] — 2026-03-15

First trained ControlMT model. KN↔EN single-pair. ~106M parameters (smaller embedding).
Initial experiment with several known bugs. Deprecated.
