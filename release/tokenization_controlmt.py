"""ControlMT tokenizer wrapper — SentencePiece + control/direction token handling.

Lets users load via:
    AutoTokenizer.from_pretrained("anandkaman/controlmt-v2.2", trust_remote_code=True)
"""

import os
from typing import List, Optional, Union

import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class ControlMTTokenizer(PreTrainedTokenizer):
    """Minimal SentencePiece wrapper with ControlMT's direction + style tokens.

    The model expects input formatted as:
        [BOS] [DIRECTION_ID] [STYLE_ID] <source tokens> [EOS]

    Use the high-level `.translate_text(...)` convenience that builds this prefix,
    or the lower-level `.encode(...)` if doing it manually.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        sp_model_kwargs: Optional[dict] = None,
        direction_tokens: Optional[dict] = None,
        control_tokens: Optional[dict] = None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.sp_model_kwargs = sp_model_kwargs or {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        self.direction_tokens = direction_tokens or {
            "kn2en": 4, "en2kn": 5,
            "rkn2kn": 12, "rkn2en": 13, "hi2en": 14, "en2hi": 15,
        }
        self.control_tokens = control_tokens or {
            "strict": 6, "natural": 7, "formal": 8,
            "casual": 9, "json": 10, "text": 11,
        }

        super().__init__(
            bos_token=bos_token, eos_token=eos_token,
            unk_token=unk_token, pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            direction_tokens=self.direction_tokens,
            control_tokens=self.control_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.sp_model.decode(tokens)

    def encode(self, text: str, **kwargs) -> List[int]:
        """Plain SentencePiece encoding (no prefix). Used inside .translate_text()."""
        return self.sp_model.encode(text, out_type=int)

    def decode(self, ids: List[int], **kwargs) -> str:
        # Strip any direction/style/special tokens that may have leaked
        special = set([0, 1, 2, 3])  # PAD, BOS, EOS, UNK
        special.update(self.direction_tokens.values())
        special.update(self.control_tokens.values())
        ids = [i for i in ids if i not in special]
        return self.sp_model.decode(ids)

    def translate_text(self, text: str, direction: str = "kn2en") -> List[int]:
        """Build the full HF-style input_ids prefix:  [BOS] [DIRECTION] [CONTROL] tokens [EOS]

        v2.3 ships single-register; the control token slot is fixed to the architectural
        default (NATURAL = id 7). Future versions may surface a register selector.
        """
        dir_id = self.direction_tokens[direction]
        ctrl_id = self.control_tokens.get("natural", 7)
        body = self.encode(text)
        return [1, dir_id, ctrl_id] + body + [2]  # 1=BOS, 2=EOS

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        import shutil
        out_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_file):
            shutil.copy(self.vocab_file, out_file)
        return (out_file,)
