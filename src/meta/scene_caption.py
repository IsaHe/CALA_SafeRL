"""
scene_caption.py - Observacion (739-dim) -> caption en lenguaje natural para la
anotacion de preferencias del reward model estilo Motif.

El LLM no ve el vector de observacion: ve una descripcion textual de la escena de
conduccion. Para que la senal de reward aprendida sea consistente entre la
ANOTACION (offline) y el ENTRENAMIENTO (el reward model lee obs[240:480]), el
caption se deriva EXACTAMENTE del mismo vector de observacion que ve el agente,
con las mismas convenciones del proyecto:

  - LIDAR dinamico = obs[240:480], 240 bins, normalizado dist/lidar_range
    (1.0 = libre, <1.0 = obstaculo a ``valor*50`` m). Convencion angular del
    ``SemanticLidarProcessor``: ``angle = atan2(-y, x)``, bin 0 = de frente, el
    angulo crece en sentido antihorario (hacia la IZQUIERDA, pues y=derecha en el
    sistema zurdo de CARLA). Cono frontal = ``FRONT_N`` bins a cada lado de 0.
  - Velocidad real recuperable de las features de ruta:
    ``speed_kmh = speed_ratio * speed_limit_norm * MAX_SPEED_LIMIT_KMH``.

Funcion pura y CARLA-free: testeable con vectores sinteticos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# ── Layout de la observacion (ver CarlaEnv / CLAUDE.md) ─────────────────────
OBS_DIM = 739
LIDAR_RANGE_M = 50.0
NUM_RAYS = 240
FRONT_N = 15  # = SemanticLidarProcessor.FRONT_N (cono frontal de +-22.5 deg)
MAX_SPEED_LIMIT_KMH = 130.0  # = CarlaEnv.MAX_SPEED_LIMIT_KMH

DYN_START, DYN_END = 240, 480  # canal LIDAR dinamico (vehiculos + peatones)

# Lane features obs[720:728]
IDX_LATERAL_OFFSET = 720
IDX_HEADING_ERR = 721
IDX_ON_EDGE = 722
IDX_LANE_WIDTH = 723
IDX_DIST_LEFT_EDGE = 724
IDX_DIST_RIGHT_EDGE = 725
IDX_LANE_CHANGE = 726
IDX_CURVATURE = 727
# Vehicle state obs[732:734]
IDX_SPEED_NORM = 732
IDX_STEERING = 733
# Route info obs[734:739]
IDX_PROGRESS = 736
IDX_SPEED_LIMIT_NORM = 737
IDX_SPEED_RATIO = 738

_FREE = 0.999  # valor de bin >= esto => sin obstaculo dentro de rango


@dataclass(frozen=True)
class SceneFeatures:
    """Resumen numerico de la escena (auditoria + posible uso directo)."""

    speed_kmh: float
    speed_limit_kmh: float
    front_dynamic_m: float  # nearest dynamic en el cono frontal (>=50 si libre)
    nearest_dynamic_m: float  # nearest dynamic en cualquier direccion
    nearest_bearing: str  # etiqueta de rumbo del nearest dynamic
    lateral_offset: float  # obs normalizada [-1,1]
    heading_error: float
    closer_edge: str  # "left" | "right" | "centered"
    on_edge: bool
    curvature: float

    def to_dict(self) -> dict:
        return asdict(self)


def _bearing_label(bin_idx: int, num_rays: int = NUM_RAYS) -> str:
    """Etiqueta de rumbo para un bin del scan (convencion atan2(-y, x))."""
    deg = (bin_idx % num_rays) * 360.0 / num_rays  # 0..360, 0=frente, CCW=izquierda
    if deg <= 22.5 or deg >= 337.5:
        return "directly ahead"
    if deg < 67.5:
        return "ahead and to the left"
    if deg < 112.5:
        return "to the left"
    if deg < 157.5:
        return "behind and to the left"
    if deg < 202.5:
        return "directly behind"
    if deg < 247.5:
        return "behind and to the right"
    if deg < 292.5:
        return "to the right"
    return "ahead and to the right"


def extract_scene_features(obs: np.ndarray) -> SceneFeatures:
    """Extrae las features de seguridad relevantes desde el vector de obs."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] < OBS_DIM:
        raise ValueError(f"obs must have >= {OBS_DIM} dims, got {obs.shape[0]}")

    dyn = obs[DYN_START:DYN_END]  # 240, normalizado (1=libre)
    # Cono frontal = bins [num-FRONT_N : num] U [0 : FRONT_N].
    front = np.concatenate([dyn[NUM_RAYS - FRONT_N :], dyn[:FRONT_N]])
    front_min = float(front.min())
    front_m = front_min * LIDAR_RANGE_M if front_min < _FREE else float("inf")

    nearest_bin = int(np.argmin(dyn))
    nearest_norm = float(dyn[nearest_bin])
    nearest_m = nearest_norm * LIDAR_RANGE_M if nearest_norm < _FREE else float("inf")
    nearest_bearing = _bearing_label(nearest_bin) if nearest_norm < _FREE else "none"

    speed_limit_kmh = float(obs[IDX_SPEED_LIMIT_NORM]) * MAX_SPEED_LIMIT_KMH
    speed_kmh = float(obs[IDX_SPEED_RATIO]) * speed_limit_kmh

    dist_left = float(obs[IDX_DIST_LEFT_EDGE])
    dist_right = float(obs[IDX_DIST_RIGHT_EDGE])
    if abs(dist_left - dist_right) < 0.05:
        closer_edge = "centered"
    else:
        closer_edge = "left" if dist_left < dist_right else "right"

    return SceneFeatures(
        speed_kmh=round(speed_kmh, 1),
        speed_limit_kmh=round(speed_limit_kmh, 1),
        front_dynamic_m=round(front_m, 1) if np.isfinite(front_m) else float("inf"),
        nearest_dynamic_m=round(nearest_m, 1)
        if np.isfinite(nearest_m)
        else float("inf"),
        nearest_bearing=nearest_bearing,
        lateral_offset=round(float(obs[IDX_LATERAL_OFFSET]), 3),
        heading_error=round(float(obs[IDX_HEADING_ERR]), 3),
        closer_edge=closer_edge,
        on_edge=bool(obs[IDX_ON_EDGE] > 0.5),
        curvature=round(float(obs[IDX_CURVATURE]), 3),
    )


def caption_from_obs(obs: np.ndarray) -> str:
    """Construye el caption en lenguaje natural que se le pasa al LLM."""
    f = extract_scene_features(obs)
    parts: list[str] = []

    # 1) Velocidad del ego.
    if f.speed_kmh < 1.0:
        parts.append("The car is stopped")
    else:
        parts.append(f"The car is driving at {f.speed_kmh:.0f} km/h")
    if f.speed_limit_kmh > 0:
        parts[-1] += f" (speed limit {f.speed_limit_kmh:.0f} km/h)"

    # 2) Obstaculos dinamicos (vehiculos/peatones) - lo mas relevante para colision.
    if np.isfinite(f.front_dynamic_m):
        parts.append(
            f"another road user is {f.front_dynamic_m:.0f} m directly ahead in the path"
        )
    elif np.isfinite(f.nearest_dynamic_m):
        parts.append(
            f"the path directly ahead is clear, but another road user is "
            f"{f.nearest_dynamic_m:.0f} m {f.nearest_bearing}"
        )
    else:
        parts.append("no other road users are detected within 50 m")

    # 3) Posicion en el carril.
    if abs(f.lateral_offset) < 0.15:
        lane = "well centred in the lane"
    elif abs(f.lateral_offset) < 0.5:
        lane = "slightly off-centre in the lane"
    else:
        lane = f"far off-centre, close to the {f.closer_edge} edge"
    if f.on_edge:
        lane = f"at the {f.closer_edge} edge of the road"
    parts.append(f"the car is {lane}")

    # 4) Alineacion y geometria.
    if abs(f.heading_error) > 0.3:
        parts.append("its heading is poorly aligned with the lane")
    if abs(f.curvature) > 0.3:
        parts.append("the road is curving")

    return ". ".join(_capitalize_first(p) for p in parts) + "."


def _capitalize_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s
