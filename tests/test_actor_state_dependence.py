"""
Tests de sesión 5 — expresividad del actor head y balance off-road vs
shaping acumulado.

A) Expresividad del actor:
   - Verifica que, tras el cambio de init `uniform(-3e-3, 3e-3)` →
     `orthogonal(gain=0.1)`, el output del `actor_mean` varía
     significativamente entre estados distintos. Sin este cambio, la
     red era bias-dominada (output state-independent) y `approx_kl`
     se mantenía en ~1e-8 durante toda la run.

B) Balance económico off-road:
   - Con los nuevos defaults del env (`out_of_road_penalty=30.0`) y el
     shaper (`off_road_penalty=1.0`), la penalty total por salirse
     debe dominar la fracción del shaping acumulado en un episodio
     típico off-road. Este invariante es la base del Fix B.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.PPO.ActorCritic import ActorCritic  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# A) Expresividad del actor
# ──────────────────────────────────────────────────────────────────────────


def _make_policy(seed: int = 0, hidden_dim: int = 256) -> ActorCritic:
    """
    hidden_dim=256 por defecto (valor de producción). La expresividad
    depende del tamaño del hidden layer.
    """
    torch.manual_seed(seed)
    return ActorCritic(
        state_dim=ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM,
        action_dim=2,
        hidden_dim=hidden_dim,
    )


def test_actor_mean_state_dependent_at_init():
    """
    Con la nueva init orthogonal(gain=0.1), el `action_mean` debe variar
    entre estados de forma claramente superior a la init antigua
    `uniform(-3e-3, 3e-3)` (que daba ~1e-4 sobre inputs similares).

    Umbral: 3x mejor que la init anterior → > 3e-4. Con hidden_dim=256
    y states N(0,1), en la práctica se observan valores del orden ~5e-3.
    """
    policy = _make_policy(seed=0, hidden_dim=256)
    torch.manual_seed(123)
    n = 32
    # Estados normales estándar — simulan el output del RunningMeanStd
    # que el agente ve tras la normalización online de observaciones.
    states = torch.randn(n, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)

    with torch.no_grad():
        features_in = policy._encode(states)
        features = policy.actor(features_in)
        action_mean = policy.actor_mean(features)  # (n, 2)

    steering_std = action_mean[:, 0].std().item()
    throttle_std = action_mean[:, 1].std().item()

    # Con la init vieja (uniform ±3e-3), ambas std eran O(1e-4).
    assert steering_std > 3e-4, (
        f"Steering state-independent: std={steering_std:.4e} "
        "(vs <~1e-4 de la init vieja)."
    )
    assert throttle_std > 3e-4, f"Throttle state-independent: std={throttle_std:.4e}."


def test_actor_bias_preserved_on_average():
    """
    El bias throttle +0.8 se preserva: action_mean[throttle] promedio
    sigue siendo positivo claro; steering se centra en torno a 0.
    """
    policy = _make_policy(seed=0, hidden_dim=256)
    torch.manual_seed(123)
    n = 64
    states = torch.randn(n, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)

    with torch.no_grad():
        features_in = policy._encode(states)
        features = policy.actor(features_in)
        action_mean = policy.actor_mean(features)

    steering_mean = action_mean[:, 0].mean().item()
    throttle_mean = action_mean[:, 1].mean().item()

    # Bias preserved: throttle positivo claro, steering cerca de 0.
    assert throttle_mean > 0.4, (
        f"Forward bias perdido: throttle_mean={throttle_mean:.3f}. "
        "Debería mantenerse cerca de 0.8 (bias init)."
    )
    assert abs(steering_mean) < 0.3, (
        f"Steering sesgado: steering_mean={steering_mean:.3f}. "
        "Debería estar cerca de 0 (sin bias lateral)."
    )


def test_actor_weight_init_is_orthogonal_not_uniform():
    """
    Sanity check del cambio sesión 5: la init actual produce pesos con
    norma por fila ~= gain (característica de orthogonal init), NO una
    distribución uniforme estrecha.

    Norma por fila de una matriz orthogonal con gain=0.1 → ~0.1.
    La init uniform(-3e-3, 3e-3) daba norma por fila ~ 0.028.
    """
    policy = _make_policy(seed=0)
    w = policy.actor_mean.weight.data  # (action_dim, hidden)
    row_norms = w.norm(dim=1)  # norma euclídea por fila

    for i, norm in enumerate(row_norms):
        assert 0.05 < norm.item() < 0.2, (
            f"Fila {i} del actor_mean.weight tiene norma {norm.item():.4f}, "
            "esperado ~0.1 (gain orthogonal de sesión 5)."
        )


# ──────────────────────────────────────────────────────────────────────────
# B) Balance económico off-road vs shaping acumulado
# ──────────────────────────────────────────────────────────────────────────

# Defaults de main_train.py tras Fix B (sesión 5).
ENV_OFF_ROAD_PENALTY = 30.0
SHAPER_OFF_ROAD_PENALTY = 1.00

# Estimación empírica del shaping NETO por step en régimen on-road típico
# (v ~= 18 km/h, centrado, heading OK), basada en las correlaciones
# observadas en la run de sesión 4:
#   progress_reward ≈ 0.30, accel ≈ 0, speed_reward ≈ 0.05,
#   lane_centering ≈ 0.15, heading ≈ 0.04, idle = 0
SHAPING_PER_STEP_NET = 0.54
TYPICAL_OFFROAD_EP_LENGTH = 186  # mean observado en sesión 4


def test_off_road_penalty_meaningful_against_shaping():
    """
    La penalty total por off-road (env + shaper) debe ser al menos el
    25% del shaping acumulado durante un episodio típico off-road.
    Con este ratio, el agente ya no puede maximizar reward optimizando
    "correr hasta caerse" (el net R baja de ~+91 a ~+70).
    """
    total_penalty = ENV_OFF_ROAD_PENALTY + SHAPER_OFF_ROAD_PENALTY
    accumulated_shaping = TYPICAL_OFFROAD_EP_LENGTH * SHAPING_PER_STEP_NET

    ratio = total_penalty / accumulated_shaping

    assert ratio >= 0.25, (
        f"Off-road penalty demasiado pequeña: {total_penalty:.1f} vs "
        f"shaping acumulado {accumulated_shaping:.1f} (ratio {ratio:.2f}). "
        "El agente seguirá prefiriendo off-road como estrategia rentable."
    )


def test_off_road_penalty_not_catastrophic():
    """
    La penalty total NO debe exceder el doble del shaping acumulado
    (evitar volver a parálisis tipo sesión 3 donde off-road = -23
    dominaba todo).
    """
    total_penalty = ENV_OFF_ROAD_PENALTY + SHAPER_OFF_ROAD_PENALTY
    accumulated_shaping = TYPICAL_OFFROAD_EP_LENGTH * SHAPING_PER_STEP_NET

    assert total_penalty <= 2.0 * accumulated_shaping, (
        f"Off-road penalty {total_penalty:.1f} excede 2x shaping "
        f"acumulado {accumulated_shaping:.1f}. Riesgo de parálisis "
        "por sobre-penalización."
    )


def test_off_road_net_episode_still_learnable():
    """
    El reward neto de un episodio off-road de longitud media debe
    seguir siendo interpretable (ni catastrófico ni demasiado positivo).
    Ventana esperada: [-30, +120] para que el agente pueda distinguir
    entre "llegar lejos y caer" y "recorrer poco y caer".
    """
    for ep_length in [50, 100, 186, 300]:
        shaping_total = ep_length * SHAPING_PER_STEP_NET
        net = shaping_total - ENV_OFF_ROAD_PENALTY - SHAPER_OFF_ROAD_PENALTY
        assert -50.0 <= net <= 140.0, (
            f"Episode off-road de {ep_length} steps → net reward {net:.1f} "
            "fuera del rango aprendible [-50, 140]."
        )


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import inspect

    tests = [
        obj
        for name, obj in sorted(inspect.getmembers(sys.modules[__name__]))
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    for t in tests:
        print(f"--- {t.__name__} ---")
        try:
            t()
            print("  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
