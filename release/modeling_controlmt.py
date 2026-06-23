"""ControlMT HuggingFace integration — minimal wrapper around the native model.

Lets users load via:
    AutoModelForSeq2SeqLM.from_pretrained("anandkaman/controlmt-v2.2", trust_remote_code=True)

The actual architecture lives in `model.py` (sibling file). This module thinly
wraps it as a HF PreTrainedModel + adds a `.translate()` convenience that builds
the correct [BOS, direction, style] prefix and runs beam search with optional
Anti-LM contrastive decoding (the v2.2 default).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .configuration_controlmt import ControlMTConfig
# Relative import — HF's trust_remote_code auto-downloads model.py alongside this file
from .model import ControlMT as _NativeControlMT, BOS_ID, EOS_ID, PAD_ID


class ControlMTForSeq2SeqLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for ControlMT v2.2.

    NOTE: The model uses **explicit routing** (direction token selects which
    language encoder/decoder runs). The `.forward()` method here is for
    teacher-forced training/eval; for generation use `.translate()` which
    handles direction selection + decoding correctly.
    """

    config_class = ControlMTConfig
    base_model_prefix = "controlmt"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: ControlMTConfig):
        super().__init__(config)
        self.config = config
        # Build the native ControlMT module
        self.controlmt = _NativeControlMT(vocab_size=config.vocab_size)
        # HF expects post-init
        self.post_init()

    def get_input_embeddings(self):
        return self.controlmt.token_embedding

    def set_input_embeddings(self, value):
        self.controlmt.token_embedding = value

    def get_output_embeddings(self):
        # Tied with input embedding
        return self.controlmt.token_embedding

    # ── Teacher-forced forward (for training / eval) ──────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        direction_id: int = None,
        labels: torch.LongTensor = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """Teacher-forced forward pass. Used for training/eval; not generation."""
        if direction_id is None:
            # Try to read from input_ids prefix: [BOS, DIRECTION, STYLE, ...]
            if input_ids is not None and input_ids.size(1) >= 2:
                direction_id = int(input_ids[0, 1].item())
            else:
                direction_id = self.config.direction_tokens["kn2en"]

        logits = self.controlmt(
            input_ids, attention_mask, decoder_input_ids,
            decoder_attention_mask, direction_id=direction_id,
        )

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID,
                                                 label_smoothing=0.1)
            loss = loss_fn(logits.view(-1, logits.size(-1)),
                          labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return Seq2SeqLMOutput(loss=loss, logits=logits)

    # ── Generation (beam search + Anti-LM contrastive) ────────────────────────

    @torch.no_grad()
    def translate(
        self,
        text: str,
        tokenizer,
        direction: str = "kn2en",
        num_beams: int = 6,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        anti_lm_alpha: float = 0.5,
        max_length: int = 256,
    ) -> str:
        """One-shot translation. The recommended entry point.

        Args:
            text: source string
            tokenizer: a ControlMTTokenizer (or compatible — needs .encode/.decode)
            direction: "kn2en" / "en2kn" / "rkn2kn"
            num_beams: beam search size (default 6, matches reported benchmark numbers)
            length_penalty: 1.2 (NLLB/IndicTrans2 default)
            no_repeat_ngram_size: 3 (prevents `_ _ _` class of repetitions)
            anti_lm_alpha: 0.5 (contrastive decoding strength; 0 disables)
            max_length: 256 (caps output length)
        """
        device = next(self.parameters()).device
        dir_id = self.config.direction_tokens[direction]
        # v2.3 ships single-register; control token is fixed to the default NATURAL.
        ctrl_id = self.config.default_control_token_id

        src_tokens = tokenizer.encode(text)
        src_ids = [BOS_ID, dir_id, ctrl_id] + src_tokens + [EOS_ID]
        src_t = torch.tensor([src_ids], device=device)
        src_mask = torch.ones_like(src_t)
        memory, mem_mask = self.controlmt.encode(src_t, src_mask, dir_id, ctrl_id)

        # Anti-LM memory: same shape but mask zeroed → cross-attention sees nothing
        anti_mem_mask = torch.zeros_like(mem_mask) if anti_lm_alpha > 0 else None

        def banned_ngrams(seq, n):
            if n <= 0 or len(seq) < n:
                return set()
            prefix = tuple(seq[-(n - 1):])
            return {tuple(seq[i:i + n])[-1] for i in range(len(seq) - n + 1)
                    if tuple(seq[i:i + n])[:-1] == prefix}

        beams = [([BOS_ID], 0.0)]
        finished = []
        for _ in range(max_length):
            if not beams:
                break
            cands = []
            for seq, score in beams:
                if seq[-1] == EOS_ID:
                    finished.append((seq, score))
                    continue
                t_t = torch.tensor([seq], device=device)
                tm = torch.ones_like(t_t)
                logits = self.controlmt.decode(t_t, tm, memory, mem_mask, dir_id)
                lp_main = torch.log_softmax(logits[0, -1], dim=-1).clone()
                if anti_lm_alpha > 0 and anti_mem_mask is not None:
                    logits_anti = self.controlmt.decode(t_t, tm, memory, anti_mem_mask, dir_id)
                    lp_anti = torch.log_softmax(logits_anti[0, -1], dim=-1)
                    lp = lp_main - anti_lm_alpha * lp_anti
                else:
                    lp = lp_main
                for tok in banned_ngrams(seq, no_repeat_ngram_size):
                    lp[tok] = -1e9
                topk = lp.topk(num_beams)
                for tok, lpv in zip(topk.indices.tolist(), topk.values.tolist()):
                    cands.append((seq + [tok], score + lpv))
            cands.sort(key=lambda x: x[1] / max(len(x[0]), 1) ** length_penalty,
                       reverse=True)
            beams = cands[:num_beams]
            if all(b[0][-1] == EOS_ID for b in beams):
                finished.extend(beams)
                break
        if not finished:
            finished = beams
        best = max(finished, key=lambda x: x[1] / max(len(x[0]), 1) ** length_penalty)
        seq = best[0]
        if seq and seq[0] == BOS_ID:
            seq = seq[1:]
        if seq and seq[-1] == EOS_ID:
            seq = seq[:-1]
        return tokenizer.decode(seq)
