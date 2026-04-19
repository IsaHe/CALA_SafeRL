from torch import nn
import torch


class ActorCritic(nn.Module):
    """
    Red Actor-Critic para PPO con acciones continuas.

    El input (735 dims) se divide en:
      - LIDAR (720 dims, 3 canales x 240 rayos en [0,1])
      - Vector features (15 dims: 8 lane + 2 vehicle + 5 route)

    El LIDAR pasa por un encoder dedicado que lo comprime a un embedding
    compacto antes de concatenarlo con las features vectoriales. Así las
    15 dims de lane/route no quedan diluidas entre 720 dims de LIDAR al
    entrar al trunk principal.
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX = -0.2

    LIDAR_TOTAL = 720
    VECTOR_DIM = 15

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        assert state_dim == self.LIDAR_TOTAL + self.VECTOR_DIM, (
            f"ActorCritic: expected state_dim={self.LIDAR_TOTAL + self.VECTOR_DIM}, "
            f"got {state_dim}"
        )

        lidar_embed = 64
        self.lidar_encoder = nn.Sequential(
            nn.Linear(self.LIDAR_TOTAL, 256),
            nn.Tanh(),
            nn.Linear(256, lidar_embed),
            nn.Tanh(),
        )

        trunk_in = lidar_embed + self.VECTOR_DIM

        self.critic = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # std ≈ 1.0 (log_std=0) para exploración más amplia en el espacio de 735 dims.
        # log_std=-0.7 → σ≈0.50 pre-tanh: distribución unimodal, no satura tanh.
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.7))

        nn.init.uniform_(self.actor_mean.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.actor_mean.bias)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[..., : self.LIDAR_TOTAL]
        vec = state[..., self.LIDAR_TOTAL :]
        return torch.cat([self.lidar_encoder(lidar), vec], dim=-1)

    def get_value(self, state):
        return self.critic(self._encode(state))

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Args:
            state  : Tensor (B, state_dim)
            action : Tensor (B, action_dim) en [-1,1] (post-tanh), opcional.

        Returns:
            action_squashed, log_prob, entropy, value
        """
        features_in = self._encode(state)
        features = self.actor(features_in)
        action_mean = self.actor_mean(features)

        log_std = torch.clamp(
            self.actor_log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
        )
        action_std = torch.exp(log_std)
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            raw_action = dist.rsample()
        else:
            raw_action = torch.atanh(torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6))

        action_squashed = torch.tanh(raw_action)

        log_prob_raw = dist.log_prob(raw_action)
        log_det_jacob = torch.log(1.0 - action_squashed.pow(2) + 1e-6)
        log_prob = (log_prob_raw - log_det_jacob).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        value = self.critic(features_in)

        return action_squashed, log_prob, entropy, value
