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
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # ============================================================
        # Inicialización de pesos
        # ============================================================
        # Inicializar la salida de media de forma pequeña (explora menos al inicio)
        nn.init.uniform_(self.actor_mean.weight, -3e-3, 3e-3)

    def get_value(self, state):
        """Obtiene el valor de un estado."""
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        """
        Obtiene acción, log-probabilidad, entropía y valor.
        
        Args:
            state: Estado actual
            action: Acción (si es None, se samplea)
            
        Returns:
            action: Acción tomada o sampeada
            action_log_prob: Log-probabilidad de la acción
            dist_entropy: Entropía de la distribución
            state_value: Valor del estado
        """
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)

        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        action_log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(axis=-1, keepdim=True)

        state_value = self.critic(state)

        return action, action_log_prob, dist_entropy, state_value


class PPOAgent:
    """
    Agente PPO mejorado con soporte para state augmentation.
    
    Cambios respecto a versión anterior:
    - Soporte para safety_augmentation
    - Mejor manejo de dispositivos
    - Más control sobre hiperparámetros
    - Método para resetear LR schedule
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        scheduler_t_max=1000,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=10,
        hidden_dim=256,
        use_safety_augmentation=False,
    ):
        """
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de la acción
            lr: Learning rate inicial
            scheduler_t_max: Número de pasos del scheduler para completar el ciclo de coseno
            gamma: Factor de descuento
            eps_clip: Clip ratio para PPO
            k_epochs: Epochs de actualización por batch
            hidden_dim: Dimensión de capas ocultas
            use_safety_augmentation: Si True, espera estado = (obs, safety_level)
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.use_safety_augmentation = use_safety_augmentation

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[PPOAgent] Using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=scheduler_t_max,
            eta_min=1e-5,  # LR mínimo al final del ciclo   
        )
        self.mse_loss = nn.MSELoss()
        
        # Para control de learning rate
        self.base_lr = lr

    def select_action(self, state, deterministic=False):
        """
        Selecciona una acción del ambiente.
        
        Args:
            state: Observación del ambiente
            deterministic: Si True, usa media en lugar de samplear
            
        Returns:
            action: Acción a ejecutar
            log_prob: Log-probabilidad de la acción
            value: Valor estimado del estado
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                # Usar la media directamente sin ruido
                actor_features = self.policy.actor(state_tensor)
                action = self.policy.actor_mean(actor_features)
                action_log_prob = torch.zeros(1, 1).to(self.device)
            else:
                action, action_log_prob, _, value = self.policy.get_action_and_value(
                    state_tensor
                )

        return (
            action.cpu().numpy().flatten(),
            action_log_prob.cpu().item() if not deterministic else 0.0,
            value.cpu().item() if not deterministic else 0.0,
        )

    def update(self, memory):
        """
        Actualiza la política usando datos acumulados.
        
        Implementa el algoritmo PPO estándar.
        
        Args:
            memory: Dict con states, actions, log_probs, rewards, dones
            
        Returns:
            Dict con métricas de entrenamiento
        """
        old_states = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        old_actions = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory["log_probs"])).to(self.device)
        rewards = memory["rewards"]
        is_terminals = memory["dones"]

        # ============================================================
        # Calcular returns (rewards-to-go)
        # ============================================================
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)
        
        # Normalizar rewards para estabilidad
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (
            rewards_to_go.std() + 1e-7
        )
        rewards_to_go = rewards_to_go.unsqueeze(1)

        # ============================================================
        # Múltiples epochs de actualización
        # ============================================================
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0

        for epoch in range(self.k_epochs):
            _, logprobs, dist_entropy, state_values = self.policy.get_action_and_value(
                old_states, old_actions
            )

            # Ratio de probabilidades nuevo/viejo
            ratios = torch.exp(logprobs - old_log_probs.unsqueeze(1))

            # Advantage: bootstrap estimator
            advantages = rewards_to_go - state_values.detach()

            # Pérdida de política (surrogate loss)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            policy_loss = -torch.min(surr1, surr2)

            # Pérdida de valor
            value_loss = 0.5 * self.mse_loss(state_values, rewards_to_go)

            # Pérdida de entropía (regularización, fomenta exploración)
            entropy_loss = -0.02 * dist_entropy

            # Pérdida total
            loss = policy_loss + value_loss + entropy_loss

            # Backprop y update
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Registrar métricas
            total_policy_loss += policy_loss.mean().item()
            total_value_loss += value_loss.item()
            total_entropy += dist_entropy.mean().item()

            # Aproximación de KL divergence para early stopping
            with torch.no_grad():
                log_ratio = logprobs - old_log_probs.unsqueeze(1)
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                total_approx_kl += approx_kl.item()

        return {
            "policy_loss": total_policy_loss / self.k_epochs,
            "value_loss": total_value_loss / self.k_epochs,
            "entropy": total_entropy / self.k_epochs,
            "approx_kl": total_approx_kl / self.k_epochs,
        }

    def save(self, filename):
        """Guarda los pesos del modelo."""
        torch.save(self.policy.state_dict(), filename)
        print(f"[PPOAgent] Model saved to {filename}")

    def load(self, filename):
        """Carga los pesos del modelo."""
        self.policy.load_state_dict(torch.load(filename))
        print(f"[PPOAgent] Model loaded from {filename}")

    def set_lr(self, new_lr):
        """Cambia el learning rate manualmente."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def step_scheduler(self):
        """Avanza un paso del scheduler de learning rate."""
        self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]