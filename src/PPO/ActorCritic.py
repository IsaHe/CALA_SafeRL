from torch.distributions import Normal
from torch import nn
import torch


class ActorCritic(nn.Module):
    """
    Red neuronal Actor-Critic para PPO con acciones continuas.
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.5

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # CRÍTICA (Value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # ACTOR (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

        # Inicialización de pesos
        nn.init.uniform_(self.actor_mean.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.actor_mean.bias)

    def get_value(self, state):
        """Obtiene el valor de un estado."""
        return self.critic(state)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,  # si se pasa, debe estar en espacio pre-tanh
    ):
        """
        Args:
            state   : Tensor (B, state_dim)
            action  : Tensor (B, action_dim) en espacio [-1,1] (post-tanh)
                      Si es None, se samplea una nueva acción.

        Returns:
            action_squashed : Tensor (B, action_dim) en (-1, 1)
            log_prob        : Tensor (B, 1)  — incluye corrección Jacobiano
            entropy         : Tensor (B, 1)
            value           : Tensor (B, 1)
        """
        # ── Distribución sobre raw action space ──────────────────────
        features = self.actor(state)
        action_mean = self.actor_mean(features)

        log_std = torch.clamp(
            self.actor_log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
        )
        action_std = torch.exp(log_std)
        dist = Normal(action_mean, action_std)

        # ── Samplear o invertir tanh ──────────────────────────────────
        if action is None:
            raw_action = dist.rsample()  # (B, action_dim), ∈ ℝ
        else:
            # Invertir tanh para recuperar el raw action correspondiente
            # a la acción squashed almacenada en memoria.
            # Clip para evitar atanh(±1) = ±inf
            raw_action = torch.atanh(torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6))

        # ── Squash ────────────────────────────────────────────────────
        action_squashed = torch.tanh(raw_action)  # ∈ (-1, 1)

        # ── Log-probabilidad con corrección Jacobiano ─────────────────
        # log p(a) = log p(raw) - Σ log(1 - tanh²(raw) + ε)
        log_prob_raw = dist.log_prob(raw_action)  # (B, action_dim)
        log_det_jacob = torch.log(
            1.0 - action_squashed.pow(2) + 1e-6
        )  # (B, action_dim)
        log_prob = (log_prob_raw - log_det_jacob).sum(dim=-1, keepdim=True)

        # ── Entropía (sobre la distribución raw; aproximación) ────────
        # La entropía exacta del squashed Normal no tiene forma cerrada,
        # pero la de la Normal base más una constante es suficiente para
        # regularización y monitorización.
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # ── Valor ─────────────────────────────────────────────────────
        value = self.critic(state)

        return action_squashed, log_prob, entropy, value
