#!/usr/bin/env python3
import torch
import torch.nn as nn

class FlowBERTConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout

class FlowBERTForMaskedLM(nn.Module):
    def __init__(self, config: FlowBERTConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )

        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) with 1 for tokens to attend, 0 to mask
        """
        batch_size, seq_len = input_ids.size()
        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        # Prepare padding mask for TransformerEncoder: True values are positions to mask
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # (batch, seq_len)

        # TransformerEncoder expects (seq_len, batch, dim)
        x = self.encoder(x.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # back to (batch, seq_len, dim)

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits
