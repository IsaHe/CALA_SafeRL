import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.PPO.ActorCritic import ActorCritic
from src.PPO.RunningMeanStd import RunningMeanStd


# LIDAR (3 canales × 240 rayos) ya está en [0,1] por construcción en carla_env.
# Normalizar esas dimensiones con RunningMeanStd es contraproducente:
#   - En stage 0 del curriculum (0 NPCs) la scan dinámica es constante 1.0 →
#     std≈0 → la normalización genera outliers masivos cuando aparece un NPC.
#   - El LIDAR ya tiene escala consistente; normalizar perjudica la señal.
# Solo normalizamos las 15 features vectoriales (lane/vehicle/route).
LIDAR_END = ActorCritic.LIDAR_TOTAL  # 720
VECTOR_DIM = ActorCritic.VECTOR_DIM  # 15


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        scheduler_t_max: int = 1250,  # nº de updates esperados, NO episodios
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        hidden_dim: int = 256,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        value_clip: float = None,  # None para desactivar; con returns sin normalizar el clip 0.5 mata al crítico
        max_grad_norm: float = 1.0,
        kl_target: float = 0.05,
        normalize_obs: bool = True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.kl_target = kl_target
        self.normalize_obs = normalize_obs

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
            T_max=max(scheduler_t_max, 1),
            eta_min=1e-5,
        )
        self.mse_loss = nn.MSELoss()

        # Solo normalizamos las 15 dims vectoriales (lane/vehicle/route).
        # Ver nota al inicio del módulo.
        self.obs_normalizer = (
            RunningMeanStd(shape=(VECTOR_DIM,)) if normalize_obs else None
        )
        # Normalización de returns: divide por std running para estabilizar
        # el MSE del crítico sin sesgar el baseline (no restamos media).
        self.ret_rms = RunningMeanStd(shape=(1,))

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        """Aplica RMS solo al bloque vectorial final; deja el LIDAR intacto."""
        if self.obs_normalizer is None:
            return state
        out = state.copy() if state.ndim == 1 else state.copy()
        out[..., LIDAR_END:] = self.obs_normalizer.normalize(state[..., LIDAR_END:])
        return out

    def _update_obs_stats(self, state: np.ndarray):
        """Actualiza las estadísticas del RMS solo con el bloque vectorial."""
        if self.obs_normalizer is None:
            return
        self.obs_normalizer.update(state[..., LIDAR_END:])

    def select_action(self, state, deterministic=False):
        """
        Samplea una acción.

        Si normalize_obs=True, actualiza las estadísticas del normalizador con
        la observación cruda y pasa la observación normalizada a la red.

        Returns:
            action     : np.ndarray (action_dim,) en [-1,1]
            log_prob   : float (escalar)
            value      : float (estimación del crítico)
        """
        # Actualizar estadísticas solo sobre el bloque vectorial y normalizar.
        state_raw = np.asarray(state, dtype=np.float32)
        self._update_obs_stats(state_raw)
        state_input = self._normalize_obs(state_raw)

        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)

            if deterministic:
                features = self.policy.actor(state_t)
                raw_mean = self.policy.actor_mean(features)
                action = torch.tanh(raw_mean)
                log_prob_val = 0.0
                value_val = 0.0
            else:
                action, log_prob_t, _, value_t = self.policy.get_action_and_value(
                    state_t
                )
                log_prob_val = log_prob_t.cpu().item()
                value_val = value_t.cpu().item()

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
        state_input = self._normalize_obs(np.asarray(state, dtype=np.float32))

        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            _, log_prob_t, _, _ = self.policy.get_action_and_value(state_t, action_t)
        return log_prob_t.cpu().item()

    def compute_bootstrap_value(self, state: np.ndarray) -> float:
        """Calcula V(s) para bootstrap en episodios truncados (timeout)."""
        state_input = self._normalize_obs(np.asarray(state, dtype=np.float32))

        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
            value = self.policy.get_value(state_t)
        return value.cpu().item()

    def update(self, memory):
        """
        Actualiza la política usando datos acumulados.

        Implementa el algoritmo PPO estándar.

        Args:
            memory: Dict con states, actions, log_probs, rewards, dones

        Returns:
            Dict con métricas de entrenamiento
        """
        states_raw = np.array(memory["states"], dtype=np.float32)

        # Normalizar solo el bloque vectorial; LIDAR pasa tal cual.
        states_input = self._normalize_obs(states_raw)

        old_states = torch.FloatTensor(states_input).to(self.device)
        old_actions = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory["log_probs"])).to(self.device)
        rewards_t = torch.FloatTensor(np.array(memory["rewards"])).to(self.device)
        dones_t = torch.FloatTensor(np.array(memory["dones"])).to(self.device)

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
        gae = 0.0
        next_v = 0.0

        for t in reversed(range(len(rewards_t))):
            mask_done = 1.0 - dones_t[t]

            if truncated_t[t] > 0.5:
                bootstrap_v = final_values_t[t].item()
            else:
                bootstrap_v = next_v

            delta = (
                rewards_t[t]
                + self.gamma * bootstrap_v * mask_done
                - state_values_old[t]
            )
            gae = delta + self.gamma * self.gae_lambda * mask_done * gae
            advantages[t] = gae
            next_v = state_values_old[t].item()

        returns = advantages + state_values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.unsqueeze(1)

        # Normalizar returns por std corriente para estabilizar value loss.
        # Solo dividimos por std (no restamos media) para preservar el signo
        # del baseline y no sesgar las ventajas ya normalizadas.
        self.ret_rms.update(returns.cpu().numpy().reshape(-1, 1))
        ret_std = float(np.sqrt(self.ret_rms.var[0]) + 1e-8)
        returns = returns / ret_std
        returns = returns.unsqueeze(1)

        # ── Epochs ───────────────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0
        epochs_run = 0

        for _ in range(self.k_epochs):
            # Evaluar política nueva sobre las acciones almacenadas
            _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                old_states, old_actions
            )

            # Ratio PPO
            ratios = torch.exp(new_log_probs - old_log_probs.unsqueeze(1))

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                * advantages
            )
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
                    (v_clip - returns).pow(2),
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

            total_policy_loss += policy_loss.mean().item()
            total_value_loss += (
                value_loss.mean().item()
                if isinstance(value_loss, torch.Tensor)
                else value_loss.item()
            )
            total_entropy += entropy.mean().item()
            total_grad_norm += float(grad_norm)
            epochs_run += 1

            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs.unsqueeze(1)
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                total_approx_kl += approx_kl.item()

            if (
                self.kl_target is not None
                and total_approx_kl / epochs_run > self.kl_target
            ):
                break

        k = epochs_run
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss": total_value_loss / k,
            "entropy": total_entropy / k,
            "approx_kl": total_approx_kl / k,
            "grad_norm": total_grad_norm / k,
            "epochs_run": k,
        }

    def save(self, filename: str):
        checkpoint = {
            "policy": self.policy.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if self.obs_normalizer is not None
                else None
            ),
        }
        torch.save(checkpoint, filename)
        print(f"[PPOAgent] saved → {filename}")

    def load(self, filename: str):
        # PyTorch 2.6 can default to restricted loading in some setups.
        # These checkpoints include numpy arrays in obs_normalizer state.
        checkpoint = torch.load(
            filename,
            map_location=self.device,
            weights_only=False,
        )
        # Soporte para checkpoints antiguos que solo guardan state_dict del modelo
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            if (
                self.obs_normalizer is not None
                and checkpoint.get("obs_normalizer") is not None
            ):
                rms = checkpoint["obs_normalizer"]
                if np.asarray(rms["mean"]).shape == self.obs_normalizer.mean.shape:
                    self.obs_normalizer.load_state_dict(rms)
                else:
                    print(
                        f"[PPOAgent] obs_normalizer shape mismatch "
                        f"(ckpt={np.asarray(rms['mean']).shape}, "
                        f"current={self.obs_normalizer.mean.shape}); "
                        f"skipping load — stats will rebuild online."
                    )
        else:
            # Formato legacy: solo pesos de la política
            self.policy.load_state_dict(checkpoint)
        print(f"[PPOAgent] loaded ← {filename}")

    def step_scheduler(self):
        """Avanza un paso del scheduler de learning rate."""
        self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
