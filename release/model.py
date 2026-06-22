"""
ControlMT Model — Modular Encoder-Decoder Transformer with Explicit Routing
Trained by: Anand Kaman

Architecture:
  - Shared Core (6 layers, ~40M params) — language-agnostic "brain"
  - Per-language Encoder (2 layers, ~10M each) — KN, EN
  - Per-language Decoder (2 layers, ~10M each) — KN, EN
  - Control Embeddings — style/format vectors injected into core
  - Explicit routing — code selects encoder/decoder, not learned
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model hyperparameters
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
CORE_LAYERS = 6
MAX_SEQ_LEN = 320  # 256 max content tokens + room for [BOS, direction, style, EOS] prefix + buffer

# Token IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Control token IDs (style/register, set per training example from style_labels.jsonl)
CONTROL_TOKENS = {
    "strict": 6, "natural": 7, "formal": 8,
    "casual": 9, "json": 10, "text": 11,
}
# Default for pairs without a style label (translit, synth, etc.)
DEFAULT_CONTROL_ID = CONTROL_TOKENS["natural"]

# Direction token IDs.
# v2: kn2en (4), en2kn (5)
# v2.1: + rkn2kn (12) for Aksharantar word-level transliteration data
# v3 reservations: rkn2en (13), hi2en (14), en2hi (15)
DIRECTION_TOKENS = {
    "kn2en": 4, "en2kn": 5,
    "rkn2kn": 12, "rkn2en": 13,
    "hi2en": 14, "en2hi": 15,
}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder layer: self-attention + FFN."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder layer: self-attention + cross-attention + FFN."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Cross-attention over encoder output
        cross_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_out))
        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


class LanguageEncoder(nn.Module):
    """Per-language encoder module (2 layers)."""

    def __init__(self, d_model=D_MODEL, n_layers=ENCODER_LAYERS, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class LanguageDecoder(nn.Module):
    """Per-language decoder module (2 layers)."""

    def __init__(self, d_model=D_MODEL, n_layers=DECODER_LAYERS, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return x


class SharedCore(nn.Module):
    """Shared core — the brain. 6 encoder layers + 6 decoder layers.

    The core processes encoder output through its encoder layers,
    then the decoder side uses cross-attention to attend to core encoder output.
    Control embeddings are prepended to the encoder sequence.
    """

    def __init__(self, d_model=D_MODEL, n_layers=CORE_LAYERS, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def encode(self, x, src_key_padding_mask=None):
        """Process encoder output through core encoder layers."""
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Process decoder through core decoder layers with cross-attention to core encoder output."""
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return x


class ControlMT(nn.Module):
    """
    ControlMT — Modular Encoder-Decoder Transformer

    Flow: Input -> Lang Encoder -> Shared Core Encoder -> Shared Core Decoder <- Lang Decoder -> Output
    Control embeddings prepended to encoder sequence for style/format control.
    """

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN, n_control_tokens=6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared token embedding (all languages share vocabulary)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Control embeddings (style/format — injected into encoder sequence)
        self.control_embedding = nn.Embedding(n_control_tokens, d_model)

        # Per-language encoders
        self.encoders = nn.ModuleDict({
            "kn": LanguageEncoder(d_model),
            "en": LanguageEncoder(d_model),
        })

        # Shared core (the brain)
        self.core = SharedCore(d_model)

        # Per-language decoders
        self.decoders = nn.ModuleDict({
            "kn": LanguageDecoder(d_model),
            "en": LanguageDecoder(d_model),
        })

        # Output projection (shared across languages)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie embedding weights with output projection
        self.output_proj.weight = self.token_embedding.weight

        # Init weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_lang(self, direction_id):
        """Get src/tgt language from direction ID.

        v2.1 supports:
          kn2en (4)   → kn enc, en dec
          en2kn (5)   → en enc, kn dec
          rkn2kn (12) → en enc, kn dec  (romanized Kannada uses EN encoder — it's Latin script)
        v3 reservations (rkn2en, hi2en, en2hi) not yet wired.
        """
        if direction_id == DIRECTION_TOKENS["kn2en"]:
            return "kn", "en"
        elif direction_id == DIRECTION_TOKENS["en2kn"]:
            return "en", "kn"
        elif direction_id == DIRECTION_TOKENS["rkn2kn"]:
            # Romanized Kannada is Latin-script — route through the EN encoder.
            # Target is Kannada script → KN decoder.
            return "en", "kn"
        else:
            raise ValueError(f"Unknown direction ID: {direction_id}")

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        """Causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def encode(self, src_ids, src_mask, direction_id, control_id=CONTROL_TOKENS["strict"]):
        """
        Encode source sequence.

        Args:
            src_ids: (batch, src_len) — source token IDs
            src_mask: (batch, src_len) — 1 for real tokens, 0 for padding
            direction_id: int — direction token ID (4=KN2EN, 5=EN2KN)
            control_id: int — control token ID (6=strict, etc.)

        Returns:
            memory: (batch, src_len+1, d_model) — encoded representation
            memory_key_padding_mask: (batch, src_len+1) — padding mask
        """
        src_lang, _ = self._get_lang(direction_id)

        # Embed tokens
        x = self.token_embedding(src_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Create padding mask (True = ignore)
        src_key_padding_mask = (src_mask == 0)

        # Pass through language-specific encoder
        x = self.encoders[src_lang](x, src_key_padding_mask=src_key_padding_mask)

        # Prepend control embedding
        batch_size = x.size(0)
        ctrl = self.control_embedding(torch.tensor([control_id - 6], device=x.device))  # offset by first control ID
        ctrl = ctrl.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([ctrl, x], dim=1)  # (batch, src_len+1, d_model)

        # Extend padding mask for control token (always attend to it)
        ctrl_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        memory_key_padding_mask = torch.cat([ctrl_mask, src_key_padding_mask], dim=1)

        # Pass through shared core encoder
        memory = self.core.encode(x, src_key_padding_mask=memory_key_padding_mask)

        return memory, memory_key_padding_mask

    def decode(self, tgt_ids, tgt_mask, memory, memory_key_padding_mask, direction_id):
        """
        Decode target sequence.

        Args:
            tgt_ids: (batch, tgt_len)
            tgt_mask: (batch, tgt_len) — 1 for real tokens, 0 for padding
            memory: (batch, src_len+1, d_model)
            memory_key_padding_mask: (batch, src_len+1)
            direction_id: int

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        _, tgt_lang = self._get_lang(direction_id)

        # Embed target tokens
        x = self.token_embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask for decoder
        tgt_len = tgt_ids.size(1)
        causal_mask = self.generate_square_subsequent_mask(tgt_len, tgt_ids.device)
        tgt_key_padding_mask = (tgt_mask == 0)

        # Pass through shared core decoder
        x = self.core.decode(x, memory, tgt_mask=causal_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)

        # Pass through language-specific decoder
        x = self.decoders[tgt_lang](x, memory, tgt_mask=causal_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask)

        # Project to vocabulary
        logits = self.output_proj(x)
        return logits

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask, direction_id, control_id=CONTROL_TOKENS["strict"]):
        """
        Full forward pass for training.

        Args:
            src_ids: (batch, src_len)
            src_mask: (batch, src_len)
            tgt_ids: (batch, tgt_len)
            tgt_mask: (batch, tgt_len)
            direction_id: int — single direction for the batch
            control_id: int

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        memory, memory_key_padding_mask = self.encode(src_ids, src_mask, direction_id, control_id)
        logits = self.decode(tgt_ids, tgt_mask, memory, memory_key_padding_mask, direction_id)
        return logits


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            breakdown[name] = params
    return total, breakdown


if __name__ == "__main__":
    # Test model with dummy data
    VOCAB_SIZE = 64000
    BATCH_SIZE = 4
    SRC_LEN = 20
    TGT_LEN = 15

    model = ControlMT(vocab_size=VOCAB_SIZE)

    total, breakdown = count_parameters(model)
    print(f"ControlMT — Total parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"\nParameter breakdown:")
    for name, params in breakdown.items():
        print(f"  {name}: {params:,} ({params/1e6:.1f}M)")

    # Dummy forward pass
    src_ids = torch.randint(4, VOCAB_SIZE, (BATCH_SIZE, SRC_LEN))
    tgt_ids = torch.randint(4, VOCAB_SIZE, (BATCH_SIZE, TGT_LEN))
    src_mask = torch.ones(BATCH_SIZE, SRC_LEN, dtype=torch.long)
    tgt_mask = torch.ones(BATCH_SIZE, TGT_LEN, dtype=torch.long)

    # Test KN->EN
    logits = model(src_ids, src_mask, tgt_ids, tgt_mask, direction_id=4)
    print(f"\nForward pass (KN->EN):")
    print(f"  Input: src={src_ids.shape}, tgt={tgt_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Expected: ({BATCH_SIZE}, {TGT_LEN}, {VOCAB_SIZE})")

    # Test EN->KN
    logits = model(src_ids, src_mask, tgt_ids, tgt_mask, direction_id=5)
    print(f"\nForward pass (EN->KN):")
    print(f"  Output logits: {logits.shape}")

    print("\nModel architecture test PASSED!")
