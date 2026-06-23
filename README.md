# ControlMT

> **Compact, specialized** Kannada ↔ English translation model — 139M parameters,
> with code-mix-native training and Anti-LM contrastive decoding.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/version-v2.3-blue)](#)

---

## Headline benchmark — FLORES-200 devtest

| Metric | KN → EN | EN → KN |
|---|---|---|
| **CometKiwi-DA** (no ref) | **0.8437** | **0.8663** |
| **COMET-DA** (with ref) | **0.8459** | **0.8443** |
| BLEU | 27.20 | 18.50 |
| chrF | 55.84 | 55.56 |

Both directions clear the 0.82 production COMET-DA floor; EN→KN clears the 0.85
aspirational target on CometKiwi-DA.

---

## What's in this repo

| Path | Contents |
|---|---|
| [release_v23/](release_v23/) | v2.3 release artifacts — HF model card, integration code (`configuration_*.py` / `modeling_*.py` / `tokenization_*.py`), native architecture, eval results JSONs |
| [scripts/eval_release.py](scripts/eval_release.py) | 4-stage release-gate eval (translate → CometKiwi → COMET-DA → sacrebleu) |
| [scripts/render_showcase.py](scripts/render_showcase.py) | Renders the release banner from SVG template + JSON values |
| [scripts/upload_to_hf.py](scripts/upload_to_hf.py) | HF push helper |
| [assets/](assets/) | Banner SVG template + values JSON |
| [CHANGELOG.md](CHANGELOG.md) | Versioned release history |

---

## Quick start

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

> **Note**: v2.3 is currently private on HuggingFace. The model files will become
> public once the release is finalized. If you'd like early access, please get in
> touch directly.

---

## Model summary

| Property | Value |
|---|---|
| Parameters | 139M |
| Architecture | Modular encoder-decoder (per-language wrappers + shared core) |
| Languages | Kannada (`kn`) ↔ English (`en`) — bidirectional |
| Vocabulary | 128,000 (SentencePiece Unigram, joint KN+EN) |
| Training data | 6.70M parallel pairs (post CometKiwi quality filtering) + specialized streams |
| Hardware | 1 × NVIDIA RTX 5060 Ti (16 GB) |
| License | Apache 2.0 |

---

## Why a single-pair model?

Most public Indic MT models are **broad** — NLLB covers 200 languages, IndicTrans2 covers 22.
That breadth comes from parameter-sharing across languages, so each language pair gets only
a slice of the model's capacity.

ControlMT goes the other direction: **every parameter is dedicated to Kannada ↔ English**.

If you need broad multilingual coverage, use NLLB or IndicTrans2. If you need Kannada
specifically — and you care about size, latency, or on-device deployment — this is what
the trade-off looks like.

---

## Roadmap

- **v2.4** — Hindi support (`[HI2EN]` / `[EN2HI]`), iterative back-translation,
  idiom-pair augmentation, expanded vocabulary coverage (modern tech terms, long
  alphanumeric IDs, more transliteration coverage for unseen entities),
  standardized BPE tokenizer, **register/style control** (4 styles with rebalanced
  labels + contrastive separation training)
- **v3.0** (TBD) — Copy-mechanism / pointer-generator for OOV-proof transliteration

---

## Citation

```bibtex
@misc{controlmt-2026,
  author = {Anand Kaman},
  title  = {ControlMT v2.3 — A 139M-Parameter Specialized Kannada↔English Translation Model
           with Code-Mix-Native Training},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/anandkaman/controlmt-v2.3}}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
