import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Red neuronal Actor-Critic mejorada.
    
    Cambios:
    - Soporta state augmentation (para safety_level)
    - Arquitectura más flexible
    - Mejor inicialización de pesos
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX =  0.5
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # ============================================================
        # CRÍTICA (Value function)
        # ============================================================
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # ============================================================
        # ACTOR (Policy)
        # ============================================================
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.5))
        
        # ============================================================
        # Inicialización de pesos
        # ============================================================
        # Inicializar la salida de media de forma pequeña (explora menos al inicio)
        nn.init.uniform_(self.actor_mean.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.actor_mean.bias)

    def get_value(self, state):
        """Obtiene el valor de un estado."""
        return self.critic(state)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None,   # si se pasa, debe estar en espacio pre-tanh
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
        features     = self.actor(state)
        action_mean  = self.actor_mean(features)
 
        log_std = torch.clamp(
            self.actor_log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
        )
        action_std = torch.exp(log_std)
        dist       = Normal(action_mean, action_std)
 
        # ── Samplear o invertir tanh ──────────────────────────────────
        if action is None:
            raw_action = dist.rsample()          # (B, action_dim), ∈ ℝ
        else:
            # Invertir tanh para recuperar el raw action correspondiente
            # a la acción squashed almacenada en memoria.
            # Clip para evitar atanh(±1) = ±inf
            raw_action = torch.atanh(
                torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            )
 
        # ── Squash ────────────────────────────────────────────────────
        action_squashed = torch.tanh(raw_action)           # ∈ (-1, 1)
 
        # ── Log-probabilidad con corrección Jacobiano ─────────────────
        # log p(a) = log p(raw) - Σ log(1 - tanh²(raw) + ε)
        log_prob_raw  = dist.log_prob(raw_action)          # (B, action_dim)
        log_det_jacob = torch.log(
            1.0 - action_squashed.pow(2) + 1e-6
        )                                                   # (B, action_dim)
        log_prob = (log_prob_raw - log_det_jacob).sum(dim=-1, keepdim=True)
 
        # ── Entropía (sobre la distribución raw; aproximación) ────────
        # La entropía exacta del squashed Normal no tiene forma cerrada,
        # pero la de la Normal base más una constante es suficiente para
        # regularización y monitorización.
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
 
        # ── Valor ─────────────────────────────────────────────────────
        value = self.critic(state)
 
        return action_squashed, log_prob, entropy, value


class PPOAgent:
    
    def __init__(
        self,
        state_dim:     int,
        action_dim:    int,
        lr:            float = 3e-4,
        scheduler_t_max: int = 1250,   # nº de updates esperados, NO episodios
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        eps_clip:      float = 0.2,
        k_epochs:      int   = 10,
        hidden_dim:    int   = 256,
        entropy_coef:  float = 0.01,
        value_loss_coef: float = 0.5,
        value_clip:    float = 0.5,    # None para desactivar
        max_grad_norm:   float = 1.0,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.eps_clip     = eps_clip
        self.k_epochs     = k_epochs
        self.entropy_coef = entropy_coef
        self.value_clip   = value_clip
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[PPOAgent] Using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = max(scheduler_t_max, 1),
            eta_min= 1e-5,
        )
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        """
        Samplea una acción.
 
        Returns:
            action     : np.ndarray (action_dim,) en [-1,1]
            log_prob   : float (escalar)
            value      : float (estimación del crítico)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
 
            if deterministic:
                features    = self.policy.actor(state_t)
                raw_mean    = self.policy.actor_mean(features)
                action      = torch.tanh(raw_mean)
                log_prob_val= 0.0
                value_val   = 0.0
            else:
                action, log_prob_t, _, value_t = self.policy.get_action_and_value(state_t)
                log_prob_val = log_prob_t.cpu().item()
                value_val    = value_t.cpu().item()
 
        return (
            action.cpu().numpy().flatten(),
            log_prob_val,
            value_val,
        )
    
    def evaluate_action(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Calcula el log_prob de una acción concreta bajo la política actual.
        Útil para recalcular log_probs de executed_actions en main_train.py.
 
        Args:
            state  : obs del paso
            action : acción ejecutada real (post-tanh, ∈ [-1,1])
 
        Returns:
            log_prob : float escalar
        """
        with torch.no_grad():
            state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            _, log_prob_t, _, _ = self.policy.get_action_and_value(state_t, action_t)
        return log_prob_t.cpu().item()

    def update(self, memory):
        """
        Actualiza la política usando datos acumulados.
        
        Implementa el algoritmo PPO estándar.
        
        Args:
            memory: Dict con states, actions, log_probs, rewards, dones
            
        Returns:
            Dict con métricas de entrenamiento
        """
        old_states    = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        old_actions   = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory["log_probs"])).to(self.device)
        rewards_t     = torch.FloatTensor(np.array(memory["rewards"])).to(self.device)
        dones_t       = torch.FloatTensor(np.array(memory["dones"])).to(self.device)

        truncated_t = torch.FloatTensor(
            np.array(memory.get("truncated", [False] * len(memory["dones"])))
        ).to(self.device)
        final_values_t = torch.FloatTensor(
            np.array(memory.get("final_values", [0.0] * len(memory["dones"])))
        ).to(self.device)

        # ── GAE ───────────────────────────────────────────────────────
        with torch.no_grad():
            state_values_old = self.policy.get_value(old_states).squeeze(1)
 
        advantages = torch.zeros_like(rewards_t)
        gae        = 0.0
        next_v     = 0.0
 
        for t in reversed(range(len(rewards_t))):
            mask_done = 1.0 - dones_t[t]

            if truncated_t[t] > 0.5:
                bootstrap_v = final_values_t[t].item()
            else:
                bootstrap_v = next_v
 
            delta = rewards_t[t] + self.gamma * bootstrap_v * mask_done - state_values_old[t]
            gae   = delta + self.gamma * self.gae_lambda * mask_done * gae
            advantages[t] = gae
            next_v = state_values_old[t].item()
 
        returns    = advantages + state_values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.unsqueeze(1)
        returns    = returns.unsqueeze(1)

        # ── Epochs ───────────────────────────────────────────────────
        total_policy_loss  = 0.0
        total_value_loss   = 0.0
        total_entropy      = 0.0
        total_approx_kl    = 0.0
        total_grad_norm    = 0.0

        for _ in range(self.k_epochs):
            # Evaluar política nueva sobre las acciones almacenadas
            _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                old_states, old_actions
            )
 
            # Ratio PPO
            ratios = torch.exp(new_log_probs - old_log_probs.unsqueeze(1))
 
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2)
 
            # Value loss con clipping opcional
            if self.value_clip is not None:
                v_clip = state_values_old.unsqueeze(1) + torch.clamp(
                    new_values - state_values_old.unsqueeze(1),
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss = 0.5 * torch.max(
                    (new_values - returns).pow(2),
                    (v_clip    - returns).pow(2),
                )
            else:
                value_loss = 0.5 * self.mse_loss(new_values, returns)
 
            entropy_loss = -self.entropy_coef * entropy
 
            loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
 
            self.optimizer.zero_grad()
            loss.mean().backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.max_grad_norm
            )

            self.optimizer.step()
 
            total_policy_loss  += policy_loss.mean().item()
            total_value_loss   += (value_loss.mean().item()
                                   if isinstance(value_loss, torch.Tensor)
                                   else value_loss.item())
            total_entropy      += entropy.mean().item()
            total_grad_norm    += float(grad_norm)
 
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs.unsqueeze(1)
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                total_approx_kl += approx_kl.item()
 
        k = self.k_epochs
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss":  total_value_loss / k,
            "entropy":     total_entropy / k,
            "approx_kl":   total_approx_kl / k,
            "grad_norm":   total_grad_norm / k,
        }

    def save(self, filename: str):
        torch.save(self.policy.state_dict(), filename)
        print(f"[PPOAgent] saved → {filename}")
 
    def load(self, filename: str):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"[PPOAgent] loaded ← {filename}")

    def step_scheduler(self):
        """Avanza un paso del scheduler de learning rate."""
        self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr