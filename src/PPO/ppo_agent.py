import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.PPO.ActorCritic import ActorCritic
from src.PPO.RunningMeanStd import RunningMeanStd


LIDAR_END = ActorCritic.LIDAR_TOTAL  # 720 (3 canales × 240 rayos del LIDAR alto)
VECTOR_DIM = ActorCritic.VECTOR_DIM  # 19


class PPOAgent:
    """
    PPO con *Masked Policy Loss* para entrenamiento bajo safety shield.

    Principios que lo distinguen del PPO estándar:
      1. El buffer guarda la acción **propuesta** por la política (raw_action
         pre-tanh + acción post-tanh) junto con su `log_prob` original.
      2. `shield_mask[t]=1.0` si el shield modificó la acción; 0.0 en caso
         contrario. La *policy loss*, la *entropy regularization* y la
         *approx_kl* se calculan sólo sobre pasos unshielded (teorema del
         gradiente de la política: a ∼ π(·|s)).
      3. El crítico aprende del reward real (todos los samples) porque V(s)
         debe modelar el retorno bajo la política de comportamiento real
         (que incluye intervenciones del shield).
      4. `approx_kl` se comprueba ANTES de `optimizer.step()`:
           - Si kl > 1.5·kl_target → se descarta el epoch (sin step).
           - Si kl > 1.0·kl_target → se aplica el step y se rompe el bucle.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        scheduler_t_max: int = 1250,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        hidden_dim: int = 256,
        entropy_coef: float = 0.01,
        entropy_coef_min: float = 0.005,
        entropy_coef_decay_updates: int = 500,
        value_loss_coef: float = 0.5,
        value_clip: float = None,
        max_grad_norm: float = 0.5,
        kl_target: float = 0.05,
        normalize_obs: bool = True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        # Schedule de entropy_coef (sesión 4): decae linealmente de
        # `entropy_coef_initial` a `entropy_coef_min` en los primeros
        # `entropy_coef_decay_updates` updates. Previene el entropy-runaway
        # observado cuando la señal de reward era pobre.
        self.entropy_coef_initial = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_decay_updates = max(1, int(entropy_coef_decay_updates))
        self._entropy_update_count = 0
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

        self.obs_normalizer = (
            RunningMeanStd(shape=(VECTOR_DIM,)) if normalize_obs else None
        )
        self.ret_rms = RunningMeanStd(shape=(1,))

    def _normalize_obs(self, state: np.ndarray) -> np.ndarray:
        if self.obs_normalizer is None:
            return state
        out = state.copy() if state.ndim == 1 else state.copy()
        out[..., LIDAR_END:] = self.obs_normalizer.normalize(state[..., LIDAR_END:])
        return out

    def _update_obs_stats(self, state: np.ndarray):
        if self.obs_normalizer is None:
            return
        self.obs_normalizer.update(state[..., LIDAR_END:])

    def select_action(self, state, deterministic=False):
        """
        Samplea una acción.

        Returns:
            action_squashed : np.ndarray (action_dim,) ∈ [-1,1]
            raw_action      : np.ndarray (action_dim,) pre-tanh
            log_prob        : float escalar
            value           : float escalar
        """
        state_raw = np.asarray(state, dtype=np.float32)
        self._update_obs_stats(state_raw)
        state_input = self._normalize_obs(state_raw)

        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)

            if deterministic:
                features_in = self.policy._encode(state_t)
                features = self.policy.actor(features_in)
                raw_mean = self.policy.actor_mean(features)
                action = torch.tanh(raw_mean)
                return (
                    action.cpu().numpy().flatten(),
                    raw_mean.cpu().numpy().flatten(),
                    0.0,
                    0.0,
                )

            features_in = self.policy._encode(state_t)
            features = self.policy.actor(features_in)
            action_mean = self.policy.actor_mean(features)
            log_std = torch.clamp(
                self.policy.actor_log_std,
                self.policy.LOG_STD_MIN,
                self.policy.LOG_STD_MAX,
            )
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action_t = dist.rsample()
            action_squashed_t = torch.tanh(raw_action_t)

            log_prob_raw = dist.log_prob(raw_action_t)
            log_det = self.policy._log_det_tanh_jacobian(raw_action_t)
            log_prob_t = (log_prob_raw - log_det).sum(dim=-1, keepdim=True)

            value_t = self.policy.critic(features_in)

        return (
            action_squashed_t.cpu().numpy().flatten(),
            raw_action_t.cpu().numpy().flatten(),
            log_prob_t.cpu().item(),
            value_t.cpu().item(),
        )

    def compute_bootstrap_value(self, state: np.ndarray) -> float:
        state_input = self._normalize_obs(np.asarray(state, dtype=np.float32))

        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
            value = self.policy.get_value(state_t)
        return value.cpu().item()

    def update(self, memory):
        """
        Actualización PPO con masked policy loss.

        Args:
            memory: Dict con keys:
              - states, raw_actions, log_probs, rewards, dones
              - truncated, final_values
              - shield_mask (1.0 si el shield modificó la acción en ese paso)

        Returns:
            Dict con métricas por update.
        """
        states_raw = np.array(memory["states"], dtype=np.float32)
        states_input = self._normalize_obs(states_raw)

        old_states = torch.FloatTensor(states_input).to(self.device)
        old_raw_actions = torch.FloatTensor(
            np.array(memory["raw_actions"], dtype=np.float32)
        ).to(self.device)
        old_log_probs = (
            torch.FloatTensor(np.array(memory["log_probs"], dtype=np.float32))
            .to(self.device)
            .unsqueeze(1)
        )
        rewards_t = torch.FloatTensor(np.array(memory["rewards"], dtype=np.float32)).to(
            self.device
        )
        dones_t = torch.FloatTensor(np.array(memory["dones"], dtype=np.float32)).to(
            self.device
        )
        truncated_t = torch.FloatTensor(
            np.array(
                memory.get("truncated", [False] * len(memory["dones"])),
                dtype=np.float32,
            )
        ).to(self.device)
        final_values_t = torch.FloatTensor(
            np.array(
                memory.get("final_values", [0.0] * len(memory["dones"])),
                dtype=np.float32,
            )
        ).to(self.device)
        shield_mask = (
            torch.FloatTensor(
                np.array(
                    memory.get("shield_mask", [0.0] * len(memory["dones"])),
                    dtype=np.float32,
                )
            )
            .to(self.device)
            .unsqueeze(1)
        )

        mask_unshielded = 1.0 - shield_mask
        unshielded_count = mask_unshielded.sum().clamp(min=1.0)

        # ── GAE sobre TODOS los pasos (rewards son reales) ────────────
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

        # Normalizar ventajas (sobre todos los samples — reduce varianza)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.unsqueeze(1)

        # Normalizar returns por std corriente (no sesga advantages)
        self.ret_rms.update(returns.cpu().numpy().reshape(-1, 1))
        ret_std = float(np.sqrt(self.ret_rms.var[0]) + 1e-8)
        returns = (returns / ret_std).unsqueeze(1)

        # ── Epochs con KL early-stop pre-step ────────────────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0
        epochs_run = 0
        epochs_rejected = 0

        for _ in range(self.k_epochs):
            _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                old_states, old_raw_actions
            )

            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs
                approx_kl_per = (torch.exp(log_ratio) - 1.0) - log_ratio
                approx_kl = (
                    (approx_kl_per * mask_unshielded).sum() / unshielded_count
                ).item()

            # Hard stop: descartar este epoch (no aplicar step)
            if self.kl_target is not None and approx_kl > 1.5 * self.kl_target:
                epochs_rejected += 1
                break

            ratios = torch.exp(log_ratio)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                * advantages
            )
            policy_loss_per = -torch.min(surr1, surr2)
            policy_loss = (policy_loss_per * mask_unshielded).sum() / unshielded_count

            if self.value_clip is not None:
                v_clip = state_values_old.unsqueeze(1) + torch.clamp(
                    new_values - state_values_old.unsqueeze(1),
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss = (
                    0.5
                    * torch.max(
                        (new_values - returns).pow(2),
                        (v_clip - returns).pow(2),
                    ).mean()
                )
            else:
                value_loss = 0.5 * self.mse_loss(new_values, returns)

            entropy_loss_per = -self.entropy_coef * entropy
            entropy_loss = (entropy_loss_per * mask_unshielded).sum() / unshielded_count

            loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += (
                (entropy * mask_unshielded).sum() / unshielded_count
            ).item()
            total_grad_norm += float(grad_norm)
            total_approx_kl += approx_kl
            epochs_run += 1

            # Soft stop: rompe tras el step si el KL cruzó el target
            if self.kl_target is not None and approx_kl > self.kl_target:
                break

        k = max(epochs_run, 1)
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss": total_value_loss / k,
            "entropy": total_entropy / k,
            "approx_kl": total_approx_kl / k,
            "grad_norm": total_grad_norm / k,
            "epochs_run": epochs_run,
            "epochs_rejected": epochs_rejected,
            "shielded_fraction": float(shield_mask.mean().item()),
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
        checkpoint = torch.load(
            filename,
            map_location=self.device,
            weights_only=False,
        )
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
            self.policy.load_state_dict(checkpoint)
        print(f"[PPOAgent] loaded ← {filename}")

    def step_scheduler(self):
        self.scheduler.step()

    def step_entropy_decay(self):
        """Decae `entropy_coef` linealmente hacia `entropy_coef_min`.

        Llamar UNA vez por update PPO (después de `update()`). Tras
        `entropy_coef_decay_updates` llamadas, el valor queda fijo en
        `entropy_coef_min`.
        """
        self._entropy_update_count += 1
        frac = min(
            self._entropy_update_count / self.entropy_coef_decay_updates, 1.0
        )
        self.entropy_coef = self.entropy_coef_initial - frac * (
            self.entropy_coef_initial - self.entropy_coef_min
        )

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
