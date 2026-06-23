# ControlMT

> **Compact, specialized** Kannada ↔ English translation model — 139M parameters,
> with code-mix-native training and Anti-LM contrastive decoding.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-v2.3%20in%20development-orange)](#)

> **🚧 v2.3 is in active development.** Earlier v2.2 weights have been pulled from
> public HuggingFace pending v2.3 release. v2.3 refocuses the model's 139M-parameter
> capacity on translation quality (single-register). Expected ship: late June 2026.
> If you're looking for the model files, please get in touch directly.
>
> **On style control**: the architecture reserves dedicated tokens for
> register/style (STRICT / NATURAL / FORMAL / CASUAL). v2.3 ships without
> advertising these — they will become a properly-trained, separately-released
> feature in an upcoming version once the label distribution is rebalanced and
> contrastive separation training is added.

---

## What's in this repo

| Path | Contents |
|---|---|
| [release/](release/) | Release artifacts — model card README, HF integration code (`configuration_*.py` / `modeling_*.py` / `tokenization_*.py`), native architecture, eval results JSONs |
| [scripts/eval_release.py](scripts/eval_release.py) | 4-stage release-gate eval (translate → CometKiwi → COMET-DA → sacrebleu); sequential GPU loading for 16 GB VRAM |
| [scripts/render_showcase.py](scripts/render_showcase.py) | Renders the release banner from SVG template + JSON values |
| [scripts/upload_to_hf.py](scripts/upload_to_hf.py) | HF push helper (used to publish releases) |
| [assets/](assets/) | Banner SVG template + values JSON |
| [CHANGELOG.md](CHANGELOG.md) | Versioned release history |

---

## Quick start (once v2.3 ships)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anandkaman/controlmt-v2.3", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("anandkaman/controlmt-v2.3", trust_remote_code=True)

out = model.translate("ಅವನು ಬೆಂಗಳೂರಿಗೆ ಬಂದು ನನ್ನನ್ನು ಭೇಟಿಯಾಗುತ್ತಾನೆ.",
                       tokenizer=tokenizer, direction="kn2en")
print(out)
```

---

## Model summary

| Property | Value |
|---|---|
| Parameters | 139M |
| Architecture | Modular encoder-decoder (per-language wrappers + shared core) |
| Languages | Kannada (`kn`) ↔ English (`en`) — bidirectional |
| Vocabulary | 128,000 (SentencePiece Unigram, joint KN+EN) |
| Training data | 6.70M parallel pairs (post CometKiwi quality filtering) + specialized streams (NER-validated transliteration pairs, code-mix-paired groups, letter-spelled acronyms, numerical augmentation) |
| Hardware | 1 × NVIDIA RTX 5060 Ti (16 GB) |
| License | Apache 2.0 |

---

## Why a single-pair model?

Most public Indic MT models are **broad** — NLLB covers 200 languages, IndicTrans2 covers 22.
That breadth comes from parameter-sharing across languages, so each language pair gets only a slice
of the model's capacity.

ControlMT goes the other direction: **every parameter is dedicated to Kannada ↔ English**.

If you need broad multilingual coverage, use NLLB or IndicTrans2. If you need Kannada
specifically — and you care about size, latency, or on-device deployment —
this is what that trade-off looks like.

---

## Roadmap

- **v2.3** — single-register specialized KN↔EN model (in active training)
- **v2.4** — properly-trained style control (rebalanced labels + contrastive separation loss); Hindi support (`[HI2EN]` / `[EN2HI]`); iterative back-translation; idiom-pair augmentation; standardized BPE tokenizer
- **v3.0** (TBD) — Copy-mechanism / pointer-generator for OOV-proof transliteration

---

## Citation

```bibtex
@misc{controlmt-2026,
  author = {Anand Kaman},
  title  = {ControlMT — A 139M-Parameter Specialized Kannada↔English Translation Model
           with Code-Mix-Native Training},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/anandkaman/controlmt-v2.3}}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
