"""
reward_model.py - Reward model r_phi estilo Motif, entrenado por PREFERENCIAS.

Aprende un escalar de "seguridad de la escena" a partir de preferencias del LLM
sobre pares de observaciones (perdida Bradley-Terry). En la fase 3 se usa como el
POTENCIAL de un shaping invariante de politica: F_t = gamma*r_phi(s') - r_phi(s).

Entrada (FEATURE_INDICES, FUENTE UNICA DE VERDAD compartida con el shaping):
  - obs[240:480]  canal LIDAR dinamico (vehiculos/peatones) -> el riesgo frontal,
  - obs[720:728]  lane features (offset, heading, edges, curvatura) -> run-off-road,
  - obs[732:734]  estado del vehiculo (speed, steering).
Son exactamente las features de las que dependen los captions/etiquetas, asi que
r_phi puede ajustar la senal sin re-premiar nada que no este en las etiquetas.

Convencion de signo: r_phi MAS ALTO = MAS SEGURO (mayor potencial). En un par,
la escena etiquetada "safer" debe recibir mayor r_phi.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# obs[240:480] (dynamic LIDAR) + obs[720:728] (lane) + obs[732:734] (vehicle).
FEATURE_INDICES: list[int] = (
    list(range(240, 480)) + list(range(720, 728)) + list(range(732, 734))
)
REWARD_FEATURE_DIM = len(FEATURE_INDICES)  # 250
OBS_DIM = 739


class RewardModel(nn.Module):
    """MLP 250 -> hidden -> hidden -> 1. Toma la observacion COMPLETA (739) y
    selecciona internamente ``FEATURE_INDICES`` para que el shaping (fase 3) solo
    tenga que pasar ``obs`` sin replicar el slicing."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.register_buffer("idx", torch.tensor(FEATURE_INDICES, dtype=torch.long))
        self.net = nn.Sequential(
            nn.Linear(REWARD_FEATURE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.index_select(-1, self.idx)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Devuelve r_phi(obs) con forma [...] (sin la dim final de tamano 1)."""
        return self.net(self.features(obs)).squeeze(-1)

    @torch.no_grad()
    def predict_np(self, obs: np.ndarray) -> np.ndarray:
        """Conveniencia para el shaping: numpy [N,739] o [739] -> numpy r_phi."""
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.idx.device)
        single = t.ndim == 1
        if single:
            t = t.unsqueeze(0)
        r = self.forward(t).cpu().numpy()
        return float(r[0]) if single else r


def bradley_terry_loss(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    safer: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perdida Bradley-Terry ponderada por confianza.

    ``safer`` = 0 si A es mas seguro, 1 si B. Modelamos P(B mas seguro) =
    sigma(r_b - r_a) y aplicamos BCE contra ``safer``, de modo que el gradiente
    EMPUJA hacia arriba el r_phi de la escena etiquetada como segura.
    """
    logit = r_b - r_a
    target = safer.float()
    weight = None if confidence is None else confidence.float()
    return nn.functional.binary_cross_entropy_with_logits(logit, target, weight=weight)


def _split(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    return perm[n_val:], perm[:n_val]


def train_bradley_terry(
    model: RewardModel,
    dataset: dict,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    val_frac: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Entrena ``model`` sobre ``dataset`` (claves obs_a, obs_b, safer, confidence).

    Devuelve un historial con train/val loss y accuracy de ranking en validacion
    (fraccion de pares de val donde el r_phi mas alto coincide con la etiqueta).
    """
    model.to(device)
    obs_a = torch.as_tensor(dataset["obs_a"], dtype=torch.float32, device=device)
    obs_b = torch.as_tensor(dataset["obs_b"], dtype=torch.float32, device=device)
    safer = torch.as_tensor(dataset["safer"], dtype=torch.float32, device=device)
    conf = dataset.get("confidence")
    conf_t = (
        torch.as_tensor(conf, dtype=torch.float32, device=device)
        if conf is not None and len(conf)
        else torch.ones_like(safer)
    )

    n = obs_a.shape[0]
    tr_idx, va_idx = _split(n, val_frac, seed)
    tr = torch.as_tensor(tr_idx, device=device)
    va = torch.as_tensor(va_idx, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for _ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(tr.numel(), generator=gen)]
        ep_loss = 0.0
        for s in range(0, perm.numel(), batch_size):
            b = perm[s : s + batch_size]
            r_a = model(obs_a.index_select(0, b))
            r_b = model(obs_b.index_select(0, b))
            loss = bradley_terry_loss(
                r_a, r_b, safer.index_select(0, b), conf_t.index_select(0, b)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * b.numel()
        history["train_loss"].append(ep_loss / max(1, perm.numel()))

        if va.numel() > 0:
            model.eval()
            with torch.no_grad():
                ra, rb = (
                    model(obs_a.index_select(0, va)),
                    model(obs_b.index_select(0, va)),
                )
                vl = bradley_terry_loss(
                    ra, rb, safer.index_select(0, va), conf_t.index_select(0, va)
                )
                pred = (rb > ra).float()
                acc = float((pred == safer.index_select(0, va)).float().mean())
            history["val_loss"].append(float(vl))
            history["val_acc"].append(acc)
        if verbose and (_ep % max(1, epochs // 10) == 0 or _ep == epochs - 1):
            va_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
            print(
                f"  ep {_ep:4d}  train_loss={history['train_loss'][-1]:.4f}  "
                f"val_acc={va_acc:.3f}"
            )
    return history


def save_reward_model(model: RewardModel, path: str, meta: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_indices": FEATURE_INDICES,
            "feature_dim": REWARD_FEATURE_DIM,
            "meta": meta or {},
        },
        path,
    )


def load_reward_model(path: str, device: str = "cpu") -> RewardModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("feature_indices") != FEATURE_INDICES:
        raise ValueError(
            "checkpoint feature_indices do not match current FEATURE_INDICES; "
            "the reward model is incompatible with the current obs layout"
        )
    model = RewardModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
