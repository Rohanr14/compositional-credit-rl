import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalNetwork(nn.Module):
    """Shared architecture for PPO and CCA"""

    def __init__(self, input_shape, vocab_size=10, embed_dim=32, hidden_dim=128):
        super().__init__()

        # 1. Image Encoder (Minigrid is 7x7x3 usually)
        c, h, w = input_shape
        self.image_encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.image_encoder(dummy).shape[1]

        # 2. Instruction Encoder (The Fix!)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instr_rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)

        # 3. Fusion
        self.fusion = nn.Linear(conv_out + embed_dim, hidden_dim)

    def forward(self, image, mission_tokens):
        # Process Image
        img_feat = self.image_encoder(image)

        # Process Instruction
        # mission_tokens: (Batch, Seq_Len)
        embeds = self.word_embedding(mission_tokens.long())
        _, rnn_out = self.instr_rnn(embeds)
        instr_feat = rnn_out.squeeze(0)

        # Fuse
        combined = torch.cat([img_feat, instr_feat], dim=1)
        x = F.relu(self.fusion(combined))
        return x


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.encoder = CompositionalNetwork(obs_shape)

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, image, mission):
        x = self.encoder(image, mission)
        return self.actor(x), self.critic(x)

    def get_action(self, image, mission, deterministic=False):
        logits, val = self.forward(image, mission)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        return action, val