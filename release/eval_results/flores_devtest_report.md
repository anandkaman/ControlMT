# Release Eval Report — flores_devtest

- ckpt: `checkpoints_v23/step_255000.pt`
- beam: 6 | anti_lm_alpha: 0.5
- style kn→en: **natural** | style en→kn: **natural**
- test pairs: 1012

## Aggregate scores

| Metric | KN→EN | EN→KN |
|--------|-------|-------|
| CometKiwi (no ref) | **0.8437** | **0.8663** |
| COMET-DA (with ref) | **0.8459** | **0.8443** |
| BLEU | **27.20** | **18.50** |
| chrF | **55.84** | **56.12** |

## Targets (CONTROLMT.md §10.1)

- COMET-DA ship floor ≥ **0.82** / aspirational 0.85
- CometKiwi ship floor ≥ **0.75** / aspirational 0.80