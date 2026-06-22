"""ControlMT configuration class for HuggingFace integration.

Used by `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
"""

from transformers import PretrainedConfig


class ControlMTConfig(PretrainedConfig):
    """Configuration for ControlMT v2.2 — modular encoder-decoder Kannada↔English MT model."""

    model_type = "controlmt"
    is_composition = False
    keys_to_ignore_at_inference = []

    def __init__(
        self,
        vocab_size: int = 128000,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        encoder_layers_per_lang: int = 2,
        decoder_layers_per_lang: int = 2,
        shared_core_enc_layers: int = 6,
        shared_core_dec_layers: int = 6,
        max_position_embeddings: int = 320,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        decoder_start_token_id: int = 1,  # BOS for decoder start
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.encoder_layers_per_lang = encoder_layers_per_lang
        self.decoder_layers_per_lang = decoder_layers_per_lang
        self.shared_core_enc_layers = shared_core_enc_layers
        self.shared_core_dec_layers = shared_core_dec_layers
        self.max_position_embeddings = max_position_embeddings
        self.unk_token_id = unk_token_id

        # Direction tokens — task selector
        self.direction_tokens = kwargs.get("direction_tokens", {
            "kn2en": 4, "en2kn": 5,
            "rkn2kn": 12, "rkn2en": 13, "hi2en": 14, "en2hi": 15,
        })
        # Control tokens — register/style
        self.control_tokens = kwargs.get("control_tokens", {
            "strict": 6, "natural": 7, "formal": 8,
            "casual": 9, "json": 10, "text": 11,
        })
        self.default_control_token_id = kwargs.get("default_control_token_id", 7)  # NATURAL

        # Decoding presets — see config.json for full spec
        self.decoding_presets = kwargs.get("decoding_presets", {})
