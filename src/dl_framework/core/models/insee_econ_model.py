from typing import List

import torch
from torch import nn

from src.dl_framework.core.core_modules.multihead_pooling import MultiHeadPooling
from src.dl_framework.core.core_modules.transformer_2d import Transformer2D
from src.dl_framework.core.models.decoder import TableDecoder
from src.dl_framework.core.models.encoder import TableEncoder


class InseeEconModel(nn.Module):
    def __init__(self, table_names: List[str], table_shapes: List[int],
                 embed_dim=32, n_heads=4, num_layers=2,
                 dropout=0.1, use_pos_encoding=True, pool_heads=4):
        """
        Core model for single country prediction. Currently, the pipeline only works with INSEE data.
        :param table_names: The name or identifier of each table.
        :param table_shapes: The list of shapes of each table (number of feature columns excluding augmented columns).
        :param embed_dim: The embedding dim of each table.
        :param n_heads: The number of heads in the 2D multi-head attention.
        :param num_layers: Number of layers of 2D transformer.
        :param dropout: Dropout rate, used in the transformer to prevent overfitting.
        :param use_pos_encoding: Whether to use positional encoding in the transformer.
        :param pool_heads: Number of heads in the pooling layer before the decoding heads.
        """
        super().__init__()

        ff_dim = embed_dim

        self.table_names = table_names
        self.N = len(table_names)

        self.table_embeds = nn.ModuleDict()
        self.table_decoders = nn.ModuleDict()

        # Create embeddings/decoders
        for tn, k_in in zip(table_names, table_shapes):
            self.table_embeds[tn] = TableEncoder(3 * k_in, embed_dim)
            self.table_decoders[tn] = TableDecoder(embed_dim * pool_heads,
                                                   k_in)  # Since we tripled the features for Nan representation

        # 2D Transformer core
        self.core_transformer = Transformer2D(
            embed_dim=embed_dim,
            num_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding
        )

        self.pooling_layer = MultiHeadPooling(embed_dim=embed_dim, pool_heads=pool_heads)

        # Learned parameter for masking
        self.mask_embedding = nn.Parameter(
            torch.randn(embed_dim) * 0.02  # Shape: (E), scale down following Transformer implementation
        )

        self.embed_dim = embed_dim

    def forward(self, batch_data):
        """
        Args:
            batch_data (dict): A dictionary containing:
                - full_data (dict): A dictionary mapping table names to tensors of shape (B, L_max, 3*k_i).
                - mask (BoolTensor): A tensor of shape (B, L_max, N) indicating masked positions.
                - padding_mask (BoolTensor): A tensor of shape (B, L_max, N) indicating padding positions.

        Returns:
            dict: A dictionary mapping table names to tensors of shape (B, L, k_i) containing predictions.
        """
        B, L = batch_data["full_data"][self.table_names[0]].shape[:2]

        # Embed each table
        embed_list = []

        for tn in self.table_names:
            x = batch_data["full_data"][tn]  # (B, L_max, 3*k_i)

            # embed
            x_emb = self.table_embeds[tn](x)  # -> (B, L, E)

            embed_list.append(x_emb)

        # Stack into (B, L, N, E)
        embed_stack = torch.stack(embed_list, dim=2)

        # Apply the mask with the learned masking vector. Where mask=1, we'll use the learned mask.
        mask = batch_data["mask"]  # (B, L_max, N)
        masked_embedding = torch.where(
            mask.unsqueeze(-1),  # (B, L, N, 1)
            self.mask_embedding,  # Will broadcast to (B, L, N, E)
            embed_stack  # (B, L, N, E)
        )

        # Pass through the transformer
        padding_mask = batch_data["padding_mask"]  # (B, L, N)
        out_2d = self.core_transformer(masked_embedding, padding_mask=padding_mask)  # (B, L, N, E)
        out_2d = out_2d.reshape(-1, out_2d.shape[2], out_2d.shape[3])  # (B * L, N, E)

        pooled_output = self.pooling_layer(out_2d)
        pooled_output = pooled_output.reshape(B, L, -1)

        # 3) decode table by table
        decoded = {}
        for i, tn in enumerate(self.table_names):
            out = self.table_decoders[tn](pooled_output)  # (B, L, k_i)
            decoded[tn] = out

        return decoded
