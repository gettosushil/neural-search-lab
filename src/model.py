import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        output_dim=16,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.projection = nn.Linear(
            embedding_dim,
            output_dim,
        )

    def forward(self, token_ids):
        # Shape: [batch_size, sequence_length, 32]
        word_vectors = self.embedding(token_ids)

        # Ignore padding tokens when averaging.
        mask = (token_ids != 0).unsqueeze(-1)

        summed = (word_vectors * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1)

        # Shape: [batch_size, 32]
        sentence_vector = summed / lengths

        # Shape: [batch_size, 16]
        projected = self.projection(sentence_vector)

        # Unit-length vectors make dot product
        # equivalent to cosine similarity.
        return F.normalize(projected, p=2, dim=-1)