import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CreditDecomposer(nn.Module):
    """
    Decomposes credit assignment across compositional task structure.
    Key innovation: Learns to identify which primitive action sequences contributed to successful task completion, enabling transfer to novel compositions.
    """

    def __init__(self, feature_dim: int = 128, num_primitives: int = 4, memory_size: int = 100):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_primitives = num_primitives
        self.memory_size = memory_size

        # Primitive embeddings
        self.primitive_embeddings = nn.Embedding(num_primitives, feature_dim)

        # Attention mechanism for credit routing
        self.query_net = nn.Linear(feature_dim, feature_dim)
        self.key_net = nn.Linear(feature_dim, feature_dim)
        self.value_net = nn.Linear(feature_dim, feature_dim)

        # Credit value estimator
        self.credit_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Credit in [0, 1]
        )

        # Compositional credit memory (stores successful patterns)
        self.register_buffer('credit_memory_keys', torch.zeros(memory_size, feature_dim))
        self.register_buffer('credit_memory_values', torch.zeros(memory_size, feature_dim))
        self.register_buffer('credit_memory_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, trajectory_features: torch.Tensor, primitive_sequence: List[int]) -> torch.Tensor:
        """
        Compute credit assignment for a trajectory.

        Args:
        trajectory_features: (batch, seq_len, feature_dim)
        primitive_sequence: List of primitive indices for this task

        Returns:
        credit_scores: (batch, seq_len) - credit
        for each timestep
        """
        batch_size, seq_len, _ = trajectory_features.shape

        # Get primitive embeddings for this task composition
        primitive_embeds = self.primitive_embeddings(
            torch.tensor(primitive_sequence, device=trajectory_features.device)
        )  # (num_primitives, feature_dim)

        # Compute attention between trajectory and primitives
        queries = self.query_net(trajectory_features)  # (batch, seq_len, feature_dim)
        keys = self.key_net(primitive_embeds.unsqueeze(0).expand(batch_size, -1, -1))
        values = self.value_net(primitive_embeds.unsqueeze(0).expand(batch_size, -1, -1))

        # Attention scores: which primitives are relevant at each timestep
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq_len, num_primitives)

        # Weighted primitive features
        weighted_primitives = torch.bmm(attention_weights, values)  # (batch, seq_len, feature_dim)

        # Compute credit scores by comparing trajectory features with primitive features
        combined = torch.cat([trajectory_features, weighted_primitives], dim=-1)
        credit_scores = self.credit_estimator(combined).squeeze(-1)  # (batch, seq_len)

        return credit_scores, attention_weights


def store_success_pattern(self, trajectory_features: torch.Tensor, primitive_sequence: List[int]):
    """Store successful trajectory pattern in compositional memory"""
    # Average trajectory features as pattern key
    pattern_key = trajectory_features.mean(dim=1)  # (batch, feature_dim)

    # Primitive composition as pattern value
    primitive_embeds = self.primitive_embeddings(
        torch.tensor(primitive_sequence, device=trajectory_features.device)
    )
    pattern_value = primitive_embeds.mean(dim=0)  # (feature_dim,)

    # Store in memory (circular buffer)
    for i in range(pattern_key.size(0)):
        ptr = int(self.credit_memory_ptr.item())
        self.credit_memory_keys[ptr] = pattern_key[i]
        self.credit_memory_values[ptr] = pattern_value
        self.credit_memory_ptr[0] = (ptr + 1) % self.memory_size


def retrieve_similar_patterns(self, query_features: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Retrieve similar successful patterns from memory"""
    # Compute similarity with stored patterns
    query = query_features.mean(dim=1) if query_features.dim() == 3 else query_features

    # Check if memory has any entries
    memory_ptr = int(self.credit_memory_ptr.item())
    if memory_ptr == 0:
        # No patterns stored yet, return zero tensor
        return torch.zeros_like(query)

    # Only use filled memory slots
    filled_keys = self.credit_memory_keys[:memory_ptr]
    filled_values = self.credit_memory_values[:memory_ptr]

    similarities = F.cosine_similarity(
        query.unsqueeze(1),  # (batch, 1, feature_dim)
        filled_keys.unsqueeze(0),  # (1, filled_size, feature_dim)
        dim=-1
    )  # (batch, filled_size)

    # Get top-k (limited by available patterns)
    k_actual = min(k, memory_ptr)
    top_k_indices = torch.topk(similarities, k_actual, dim=-1).indices

    # Retrieve values
    retrieved = filled_values[top_k_indices]  # (batch, k, feature_dim)

    return retrieved.mean(dim=1)  # (batch, feature_dim)


class HindsightCreditAssignment(nn.Module):
    """
    Performs hindsight credit assignment on completed trajectories.

    After task completion, retroactively identifies which states / actions were critical for success.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        # LSTM for temporal credit propagation
        self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True, bidirectional=True)

        # Credit scorer
        self.credit_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory_features: torch.Tensor, final_reward: torch.Tensor) -> torch.Tensor:
        """
        Compute hindsight credit for a completed trajectory.
            Args:
            trajectory_features: (batch, seq_len, feature_dim)
            final_reward: (batch,) - reward obtained at trajectory end
            Returns:
            hindsight_credits: (batch, seq_len) - credit
            for each timestep
        """
        # Bi-directional LSTM to capture temporal dependencies
        lstm_out, _ = self.lstm(trajectory_features)  # (batch, seq_len, feature_dim*2)

        # Compute credit scores
        credit_scores = self.credit_scorer(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Weight by final reward (only assign credit if task was successful)
        final_reward_expanded = final_reward.unsqueeze(-1)  # (batch, 1)
        weighted_credits = credit_scores * final_reward_expanded

        return weighted_credits