import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.PPO.ActorCritic import ActorCritic
from src.PPO.RunningMeanStd import RunningMeanStd


LIDAR_END = ActorCritic.LIDAR_TOTAL  # 720
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
         gradiente de la política: a ~ π(·|s)).
      3. El crítico aprende del reward real (todos los samples) porque V(s)
         debe modelar el retorno bajo la política de comportamiento real
         (que incluye intervenciones del shield).
      4. `approx_kl` se comprueba ANTES de `optimizer.step()`:
           - Si kl > 1.5·kl_target → se descarta el epoch (sin step).
           - Si kl > 1.0·kl_target → se aplica el step y se rompe el bucle.
      5. **Reward scaling (estilo SB3 VecNormalize)**: los rewards crudos se
         dividen por la std móvil de un acumulador descontado
         `R̄_t = γ·R̄_{t-1} + r_t` (reseteado a 0 al cerrar episodio). No se
         resta media — V(s) debe seguir modelando E[R|s], no R − baseline.
         Mantiene critic loss en escala unitaria aunque los returns crezcan
         a medida que el agente progresa, y deja GAE consistente porque
         rewards y V(s) viven en el mismo espacio normalizado.
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
        minibatch_size: int = 64,
        hidden_dim: int = 256,
        entropy_coef: float = 0.01,
        entropy_coef_min: float = 0.005,
        entropy_coef_decay_updates: int = 500,
        value_loss_coef: float = 0.5,
        value_clip: float = None,
        max_grad_norm: float = 0.5,
        kl_target: float = 0.05,
        normalize_obs: bool = True,
        shield_imitation_coef: float = 0.0,
        shield_imitation_anneal_updates: int = 150,
        shield_imitation_steer_only: bool = True,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.minibatch_size = max(1, int(minibatch_size))
        self.entropy_coef = entropy_coef
        self.entropy_coef_initial = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_decay_updates = max(1, int(entropy_coef_decay_updates))
        self._entropy_update_count = 0
        # Shield-as-teacher BC (R2): imita la acción ejecutada por el shield en
        # pasos shielded, dando gradiente al actor donde el mask de PPO lo anula.
        # 0.0 = desactivado (comportamiento previo EXACTO). Anela a 0.
        self.shield_imitation_coef_initial = float(shield_imitation_coef)
        self.shield_imitation_coef_current = float(shield_imitation_coef)
        self.shield_imitation_anneal_updates = max(
            1, int(shield_imitation_anneal_updates)
        )
        self._imitation_update_count = 0
        # Imitar SOLO el steering del shield (no su throttle/brake): las acciones
        # del shield son correcciones de frenado e imitarlas colapsa a "parar".
        self.shield_imitation_steer_only = bool(shield_imitation_steer_only)
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
        self._returns_acc = np.zeros(1, dtype=np.float64)

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

    def _normalize_rewards(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Reward-scaling tipo SB3 VecNormalize.

        Procesa el rollout en orden cronológico construyendo el acumulador
        descontado `R̄_t = γ·R̄_{t-1} + r_t`, lo resetea al final de cada
        episodio, actualiza `ret_rms` con la distribución resultante, y
        divide los rewards por `sqrt(var(R̄))`. NO se resta la media: el
        crítico debe seguir prediciendo E[R|s].

        El acumulador persiste entre llamadas a `update()` mediante
        `self._returns_acc`, de modo que episodios partidos por la frontera
        de un rollout mantienen continuidad.
        """
        acc = float(self._returns_acc[0])
        accumulators = np.empty(len(rewards), dtype=np.float64)
        for t in range(len(rewards)):
            acc = self.gamma * acc + float(rewards[t])
            accumulators[t] = acc
            if dones[t] > 0.5:
                acc = 0.0
        self._returns_acc[0] = acc

        self.ret_rms.update(accumulators.reshape(-1, 1))
        std = float(np.sqrt(self.ret_rms.var[0]) + 1e-8)
        return (rewards.astype(np.float32) / std).astype(np.float32)

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
            log_std = self.policy.log_std()
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

    def evaluate_executed_action(self, state, executed_action):
        state_input = self._normalize_obs(np.asarray(state, dtype=np.float32))
        a = np.asarray(executed_action, dtype=np.float32).flatten()
        a_clipped = np.clip(a, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = np.arctanh(a_clipped).astype(np.float32)
        with torch.no_grad():
            state_t = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
            raw_t = torch.FloatTensor(raw).unsqueeze(0).to(self.device)
            _, log_prob, _, _ = self.policy.get_action_and_value(state_t, raw_t)
        return raw, float(log_prob.cpu().item())

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
        rewards_np = np.array(memory["rewards"], dtype=np.float32)
        dones_np = np.array(memory["dones"], dtype=np.float32)
        rewards_np = self._normalize_rewards(rewards_np, dones_np)
        rewards_t = torch.FloatTensor(rewards_np).to(self.device)
        dones_t = torch.FloatTensor(dones_np).to(self.device)
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
        shield_alpha = (
            torch.FloatTensor(
                np.array(
                    memory.get("shield_mask", [0.0] * len(memory["dones"])),
                    dtype=np.float32,
                )
            )
            .to(self.device)
            .unsqueeze(1)
            .clamp(0.0, 1.0)
        )

        mask_weight = (1.0 - shield_alpha).pow(2)

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

        returns = returns.unsqueeze(1)

        N = old_states.shape[0]
        batch_size = min(self.minibatch_size, N)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0
        total_bc_loss = 0.0
        n_updates = 0
        epochs_run = 0
        epochs_rejected = 0
        stop_training = False

        for _ in range(self.k_epochs):
            indices = torch.randperm(N, device=self.device)
            epoch_kl_sum = 0.0
            epoch_kl_count = 0
            epoch_had_step = False

            for start in range(0, N, batch_size):
                mb_idx = indices[start : start + batch_size]

                mb_states = old_states[mb_idx]
                mb_raw_actions = old_raw_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_state_values_old = state_values_old[mb_idx]
                mb_weight = mask_weight[mb_idx]
                mb_weight_sum = mb_weight.sum().clamp(min=1.0)

                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(mb_states, mb_raw_actions)
                )

                with torch.no_grad():
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl_per = (torch.exp(log_ratio) - 1.0) - log_ratio
                    mb_approx_kl = (
                        (approx_kl_per * mb_weight).sum() / mb_weight_sum
                    ).item()

                if self.kl_target is not None and mb_approx_kl > 1.5 * self.kl_target:
                    stop_training = True
                    if not epoch_had_step:
                        epochs_rejected += 1
                    break

                ratios = torch.exp(log_ratio)
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * mb_advantages
                )
                policy_loss_per = -torch.min(surr1, surr2)
                policy_loss = (policy_loss_per * mb_weight).sum() / mb_weight_sum

                if self.value_clip is not None:
                    v_clip = mb_state_values_old.unsqueeze(1) + torch.clamp(
                        new_values - mb_state_values_old.unsqueeze(1),
                        -self.value_clip,
                        self.value_clip,
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (new_values - mb_returns).pow(2),
                            (v_clip - mb_returns).pow(2),
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * self.mse_loss(new_values, mb_returns)

                entropy_loss_per = -self.entropy_coef * entropy
                entropy_loss = (entropy_loss_per * mb_weight).sum() / mb_weight_sum

                loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss

                # Shield-as-teacher BC (R2): sólo si está activado. En pasos
                # shielded (α>0) acerca tanh(mean) a la acción ejecutada por el
                # shield = tanh(raw_action almacenada). Pesado por α: cuanto más
                # intervino el shield, más fuerte la imitación. Llena el agujero
                # de gradiente que deja el mask (1-α)² del surrogate PPO.
                bc_loss_val = 0.0
                if self.shield_imitation_coef_initial > 0.0:
                    mb_shield_alpha = shield_alpha[mb_idx]
                    pred_mean = self.policy.squashed_mean(mb_states)
                    target_exec = torch.tanh(mb_raw_actions)
                    diff = pred_mean - target_exec
                    # Imita SOLO el steering (dim 0) por defecto: las acciones del
                    # shield son correcciones de FRENADO (LATERAL_RECOVERY_THROTTLE /
                    # emergency_brake), e imitar su throttle colapsa la política a
                    # "parar" — incluso restringido a intervenciones dinámicas
                    # (run learn_fix3: colapso al subir el tráfico). El throttle del
                    # shield es SIEMPRE un freno de emergencia, nunca buen ejemplo.
                    if self.shield_imitation_steer_only:
                        sq = diff[:, 0:1].pow(2)
                    else:
                        sq = diff.pow(2).sum(dim=-1, keepdim=True)
                    bc_per = mb_shield_alpha * sq
                    bc_loss = bc_per.sum() / mb_shield_alpha.sum().clamp(min=1.0)
                    loss = loss + self.shield_imitation_coef_current * bc_loss
                    bc_loss_val = bc_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += ((entropy * mb_weight).sum() / mb_weight_sum).item()
                total_grad_norm += float(grad_norm)
                total_approx_kl += mb_approx_kl
                total_bc_loss += bc_loss_val
                n_updates += 1
                epoch_had_step = True

                epoch_kl_sum += mb_approx_kl
                epoch_kl_count += 1

            if epoch_had_step:
                epochs_run += 1

            if stop_training:
                break

            if (
                self.kl_target is not None
                and epoch_kl_count > 0
                and (epoch_kl_sum / epoch_kl_count) > self.kl_target
            ):
                break

        with torch.no_grad():
            log_std_eff = self.policy.log_std().detach().flatten().cpu()
            log_std_max = self.policy.LOG_STD_MAX
            log_std_min = self.policy.LOG_STD_MIN
            near_top = log_std_eff >= log_std_max - 0.01
            near_bot = log_std_eff <= log_std_min + 0.01
            saturated = (near_top | near_bot).float().mean().item()
            log_std_steering_raw = float(log_std_eff[0].item())
            log_std_throttle_raw = (
                float(log_std_eff[1].item()) if log_std_eff.numel() > 1 else 0.0
            )

        k = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss": total_value_loss / k,
            "entropy": total_entropy / k,
            "approx_kl": total_approx_kl / k,
            "grad_norm": total_grad_norm / k,
            "epochs_run": epochs_run,
            "epochs_rejected": epochs_rejected,
            "n_updates": n_updates,
            "shielded_fraction": float((shield_alpha >= 0.05).float().mean().item()),
            "mean_shield_alpha": float(shield_alpha.mean().item()),
            "log_std_steering_raw": log_std_steering_raw,
            "log_std_throttle_raw": log_std_throttle_raw,
            "log_std_saturated_fraction": saturated,
            "entropy_coef_current": float(self.entropy_coef),
            "bc_loss": total_bc_loss / k,
            "shield_imitation_coef_current": float(self.shield_imitation_coef_current),
        }

    def save(self, filename: str):
        checkpoint = {
            "policy": self.policy.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict()
                if self.obs_normalizer is not None
                else None
            ),
            "ret_rms": self.ret_rms.state_dict(),
            "returns_acc": self._returns_acc.copy(),
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
            if checkpoint.get("ret_rms") is not None:
                self.ret_rms.load_state_dict(checkpoint["ret_rms"])
            if checkpoint.get("returns_acc") is not None:
                self._returns_acc[:] = np.asarray(
                    checkpoint["returns_acc"], dtype=np.float64
                ).reshape(self._returns_acc.shape)
        else:
            self.policy.load_state_dict(checkpoint)
        print(f"[PPOAgent] loaded <- {filename}")

    def step_scheduler(self):
        self.scheduler.step()

    def step_entropy_decay(self):
        """Decae `entropy_coef` linealmente hacia `entropy_coef_min`.

        Llamar UNA vez por update PPO (después de `update()`). Tras
        `entropy_coef_decay_updates` llamadas, el valor queda fijo en
        `entropy_coef_min`.
        """
        self._entropy_update_count += 1
        frac = min(self._entropy_update_count / self.entropy_coef_decay_updates, 1.0)
        self.entropy_coef = self.entropy_coef_initial - frac * (
            self.entropy_coef_initial - self.entropy_coef_min
        )

    def step_imitation_decay(self):
        """Anela `shield_imitation_coef` linealmente hacia 0 sobre
        `shield_imitation_anneal_updates` updates. Imita fuerte al principio
        (aprender las correcciones del shield), suelta al final (conducir solo).
        Llamar UNA vez por update PPO."""
        self._imitation_update_count += 1
        frac = min(
            self._imitation_update_count / self.shield_imitation_anneal_updates, 1.0
        )
        self.shield_imitation_coef_current = self.shield_imitation_coef_initial * (
            1.0 - frac
        )

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
