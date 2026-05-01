from torch import nn
import torch
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Red Actor-Critic para PPO con acciones continuas (TanhNormal estable).

    El input (739 dims) se divide en:
      - LIDAR (720 dims, 3 canales x 240 rayos en [0,1])
        · combined / dynamic / static del LIDAR alto del techo
      - Vector features (19 dims: 8 lane + 4 lane_marking + 2 vehicle + 5 route)

    NOTA: la versión v3 incluía un cuarto canal LIDAR (sensor bajo del
    parachoques, 240 dims extra → state_dim=979). Se eliminó por ser
    redundante con el LIDAR alto. Los modelos entrenados con state_dim=979
    NO son compatibles con esta versión.

    log_prob estable: acepta `raw_action` pre-tanh directamente, evitando
    `atanh(clamp(a, -1+eps, 1-eps))` que diverge cuando |a|→1 (las acciones
    saturadas producían |raw|≈7 y `log_prob` muy negativo cuya sensibilidad
    a cambios de `mean` explotaba los ratios PPO).

    Jacobiano de tanh calculado con la forma numéricamente estable
    `log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2*x))`.
    """

    # σ∈[0.22, 0.50]: σ_min=0.22 evita el colapso determinista que hacía
    # explotar el log_prob de acciones fuera de la moda; σ_max=0.50 (antes
    # 0.82) impide el entropy-runaway observado en la run de sesión 3
    # (entropy 1.44→1.92) cuando la señal de reward colapsaba a ruido.
    LOG_STD_MIN = -1.5
    LOG_STD_MAX = -0.7

    # Bias inicial del throttle (índice 1) pre-tanh. tanh(0.8)≈0.66, así que
    # el agente arranca con ~66% throttle desde el primer paso sin depender
    # de sampling aleatorio — rompe el cold-start del reposo (sesión 4).
    ACTOR_BIAS_THROTTLE_INIT = 0.8

    LIDAR_TOTAL = 720
    VECTOR_DIM = 19

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
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.7))

        # Orthogonal init con gain=0.1 (sesión 5) — sustituye al
        # uniform(-3e-3, 3e-3) que dejaba al head bias-dominado (entradas
        # ~1.5e-3 vs bias throttle=0.8 → output state-independent). Con
        # gain=0.1 la contribución del peso al output es ~0.1, aún pequeña
        # frente al bias de arranque (0.8) pero suficiente para que la
        # política pueda diferenciar estados desde los primeros updates.
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.1)
        nn.init.zeros_(self.actor_mean.bias)
        with torch.no_grad():
            # action[0]=steering (sin sesgo), action[1]=throttle/brake (+0.8 → 66% gas).
            if action_dim >= 2:
                self.actor_mean.bias[1] = self.ACTOR_BIAS_THROTTLE_INIT

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        lidar = state[..., : self.LIDAR_TOTAL]
        vec = state[..., self.LIDAR_TOTAL :]
        return torch.cat([self.lidar_encoder(lidar), vec], dim=-1)

    def get_value(self, state):
        return self.critic(self._encode(state))

    @staticmethod
    def _log_det_tanh_jacobian(raw_action: torch.Tensor) -> torch.Tensor:
        """log|det J_tanh(x)| estable: 2·(log(2) - x - softplus(-2x))."""
        return 2.0 * (
            torch.log(torch.tensor(2.0, device=raw_action.device))
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )

    def get_action_and_value(
        self,
        state: torch.Tensor,
        raw_action: torch.Tensor = None,
    ):
        """
        Política TanhNormal con evaluación numéricamente estable.

        Args:
            state      : Tensor (B, state_dim)
            raw_action : Tensor (B, action_dim) pre-tanh, opcional.
                         Si es None, se samplea una nueva acción.
                         Si se pasa, se evalúa log_prob para esa raw_action
                         (ver PPOAgent.update()).

        Returns:
            action_squashed : tanh(raw_action) ∈ [-1, 1]
            log_prob        : (B, 1) log-probabilidad exacta de la acción
            entropy         : (B, 1) entropía de la Normal pre-tanh
            value           : (B, 1) V(s)
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

        if raw_action is None:
            raw_action = dist.rsample()

        action_squashed = torch.tanh(raw_action)

        log_prob_raw = dist.log_prob(raw_action)
        log_det_jacob = self._log_det_tanh_jacobian(raw_action)
        log_prob = (log_prob_raw - log_det_jacob).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        value = self.critic(features_in)

        return action_squashed, log_prob, entropy, value
