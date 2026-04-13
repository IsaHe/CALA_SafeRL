# CALA\_SafeRL — Safe Reinforcement Learning para Conducción Autónoma en CARLA

Investigación sobre la efectividad de un **escudo de seguridad reactivo-adaptativo** para la reducción de situaciones peligrosas durante el entrenamiento de agentes PPO en el simulador CARLA.

---

## Índice

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Entorno CARLA Gymnasium](#entorno-carla-gymnasium)
4. [Escudo Adaptativo de Horizonte](#escudo-adaptativo-de-horizonte)
5. [Agente PPO](#agente-ppo)
6. [Reward Shaping](#reward-shaping)
7. [Sensores semánticos](#sensores-semánticos)
8. [Métricas y monitorización](#métricas-y-monitorización)
9. [Estructura del proyecto](#estructura-del-proyecto)
10. [Instalación y requisitos](#instalación-y-requisitos)
11. [Uso](#uso)
12. [Experimentos](#experimentos)

---

## Descripción del proyecto

Este proyecto investiga el impacto de un **escudo de seguridad reactivo** aplicado durante el entrenamiento con Proximal Policy Optimization (PPO) en tareas de conducción autónoma urbana. El escudo actúa como un filtro sobre las acciones propuestas por el agente: cuando detecta que una acción llevaría al vehículo a una zona peligrosa, la sustituye por una acción correctiva sin interrumpir el flujo de aprendizaje.

### Hipótesis de investigación

> Un escudo adaptativo que ajusta su nivel de protección en función del riesgo percibido (principio de mínima interferencia) reduce las situaciones peligrosas durante el entrenamiento sin penalizar significativamente el aprendizaje del comportamiento de conducción.

### Comparativa de experimentos

| Configuración | Shield | Descripción |
|---|---|---|
| `baseline` | Ninguno | PPO estándar sin restricciones |
| `basic` | Básico (LIDAR + Waypoint API) | Umbrales fijos de seguridad |
| `adaptive` | **Adaptativo** (BicycleModel + Waypoint API) | Umbrales y horizontes dinámicos según riesgo |

---

## Arquitectura del sistema

La arquitectura sigue una cadena de wrappers Gymnasium:

```
CarlaAdaptiveHorizonShield   ← Escudo adaptativo (filtro de acciones)
    └── CarlaRewardShaper    ← Moldeado de recompensa
          └── CarlaEnv       ← Entorno base CARLA
```

Cada capa envuelve a la inferior siguiendo la interfaz estándar de `gymnasium.Wrapper`, de modo que el agente PPO interactúa con la capa exterior como si fuera un entorno único.

```
┌──────────────────────────────────────────────────────┐
│                    PPO Agent                         │
│  (ActorCritic + GAE + Clipped Surrogate Loss)        │
└───────────────────────┬──────────────────────────────┘
                        │ action ∈ [-1,1]²
                        ▼
┌──────────────────────────────────────────────────────┐
│          CarlaAdaptiveHorizonShield                  │
│  • Analiza riesgo (semántico + lateral)              │
│  • Predice trayectoria con BicycleModel              │
│  • Verifica con Waypoint API                         │
│  • Sustituye acción si es insegura                   │
└───────────────────────┬──────────────────────────────┘
                        │ acción corregida o original
                        ▼
┌──────────────────────────────────────────────────────┐
│              CarlaRewardShaper                       │
│  • Bonus: velocidad, centramiento, progreso          │
│  • Penalización: invasión, borde, heading, drift     │
│  • Suprime penalizaciones en maniobras válidas       │
└───────────────────────┬──────────────────────────────┘
                        │ reward moldeado + info dict
                        ▼
┌──────────────────────────────────────────────────────┐
│                   CarlaEnv                           │
│  • Gymnasium API (reset / step / close)              │
│  • Sensor semántico LIDAR 3ch × 240 rayos            │
│  • Waypoint API (carril, heading, speed limit)       │
│  • TrafficManager (NPCs)                             │
└──────────────────────────────────────────────────────┘
```

---

## Entorno CARLA Gymnasium

**Archivo:** `src/CARLA/Env/carla_env.py`

Implementa la interfaz `gymnasium.Env` sobre CARLA 0.9.x en modo síncrono (20 Hz).

### Espacio de observación (255 dimensiones)

| Índices | Dimensiones | Descripción |
|---|---|---|
| `[0:240]` | 240 | Scan LIDAR semántico combinado, normalizado `[0,1]` (1 = libre, ~0 = obstáculo) |
| `[240:248]` | 8 | Características del carril (offset lateral, heading error, borde, anchura, distancias, curvatura, cambio de carril) |
| `[248:250]` | 2 | Estado del vehículo (velocidad normalizada, steering) |
| `[250:255]` | 5 | Info de ruta (ángulo al waypoint 5 m, 20 m, progreso, speed limit, ratio velocidad/límite) |

### Espacio de acción (continuo, 2 dimensiones)

| Componente | Rango | Descripción |
|---|---|---|
| `action[0]` | `[-1.0, 1.0]` | Steering (giro) |
| `action[1]` | `[-1.0, 1.0]` | Throttle/Brake (>0 gas, <0 freno) |

### Parámetros de configuración relevantes

| Parámetro | Default | Descripción |
|---|---|---|
| `map_name` | `Town04` | Mapa CARLA (autopista, ciudad, etc.) |
| `num_npc_vehicles` | `20` | Vehículos NPC gestionados por TrafficManager |
| `fixed_delta_seconds` | `0.05` | Paso de simulación (20 Hz) |
| `target_speed_kmh` | `30.0` | Velocidad objetivo para reward |
| `success_distance` | `250.0` | Metros a recorrer para considerar éxito |
| `max_episode_steps` | `1000` | Pasos máximos por episodio |

### Condiciones de terminación de episodio

- **Colisión** con otro actor.
- **Salida de carretera** (`on_road = False`).
- **Vehículo parado** durante demasiados pasos consecutivos.
- **Éxito**: recorrer `success_distance` metros.
- **Timeout**: alcanzar `max_episode_steps`.

---

## Escudo Adaptativo de Horizonte

**Archivo:** `src/Adaptative_Shield/adaptive_horizon_shield.py`

El `CarlaAdaptiveHorizonShield` es el componente central de investigación. Implementa el principio de **mínima interferencia**: cuanto más seguro sea el entorno, menos interviene.

### Niveles de riesgo y horizontes adaptativos

| Nivel | Condición frontal | Condición lateral | Horizonte | Multiplicador umbral |
|---|---|---|---|---|
| `safe` | distancia > 0.50 | `lat_norm` ≤ 0.70 | 1 paso | 1.0× |
| `warning` | distancia > 0.20 | `lat_norm` > 0.70 | 5 pasos | 1.5× |
| `critical` | distancia ≤ 0.20 | `lat_norm` > 0.85 | 10 pasos | 2.0× |

El nivel final es el más restrictivo entre el riesgo frontal y el lateral.

### Pipeline de verificación de seguridad (`_check_trajectory_safety`)

```
1. Emergencia peatón
   └─ Si peatón < 4 m → UNSAFE (override inmediato, sin búsqueda)

2. Checks LIDAR semánticos
   ├─ min_front_combined < umbral_frontal → UNSAFE
   ├─ min_r_side_static  < umbral_lateral → UNSAFE
   └─ min_l_side_static  < umbral_lateral → UNSAFE

3. Predicción de trayectoria (BicycleModel, `horizon` pasos)
   └─ Para cada punto predicho → Waypoint API CARLA
      └─ Si |offset_lateral / semi_ancho| > lat_thr → UNSAFE

Si todos los checks pasan → SAFE
```

### Modelo Cinemático de Bicicleta

**Archivo:** `src/Adaptative_Shield/BicycleModel.py`

Predictor de trayectoria calibrado para el Tesla Model 3 de CARLA:

- **Distancia entre ejes:** 2.87 m
- **Máximo ángulo de giro:** 0.60 rad (~34°)
- **Paso de tiempo:** 0.05 s (sincronizado con 20 Hz)
- **Aceleración máxima:** 3.0 m/s²
- **Deceleración máxima:** 7.0 m/s²

Ecuaciones de movimiento:
```
Δθ = (v / L) · tan(δ) · dt
Δx = v · cos(θ) · dt
Δy = v · sin(θ) · dt
```

### Búsqueda de acción segura (`_find_safe_action`)

Cuando la acción propuesta es insegura, el escudo prioriza candidatos en función de la amenaza detectada:

| Tipo de amenaza | Estrategia |
|---|---|
| **Peatón cercano** (<4 m) | Freno de emergencia total `[0, -1]` |
| **Amenaza dinámica** (vehículo próximo) | Freno fuerte primero, luego corrección lateral |
| **Amenaza estática lateral** (muros, quitamiedos) | Corrección de carril prioritaria + freno moderado |
| **Solo lateral** (deriva sin obstáculo frontal) | Vuelta al centro con gas suave |

### Estadísticas del escudo

El shield registra por episodio:

- `safe_steps` / `warning_steps` / `critical_steps`
- `interventions_dynamic` / `interventions_static` / `interventions_pedestrian`
- `interventions_by_horizon` → `{1: N, 5: N, 10: N}`
- `total_shield_activations`

---

## Agente PPO

**Archivos:** `src/PPO/ppo_agent.py`, `src/PPO/ActorCritic.py`, `src/PPO/RunningMeanStd.py`

Implementación de PPO con las siguientes características:

### Red neuronal Actor-Critic

- **Arquitectura:** dos redes MLP con 2 capas ocultas de 256 neuronas y activaciones `tanh`.
- **Actor:** salida squashed con `tanh` para garantizar acciones en `(-1, 1)`.
- **Corrección Jacobiana:** log-probabilidad corregida para distribución Normal squashed.
- **Crítico:** estima V(s) para cálculo de ventajas con GAE.

### Hiperparámetros PPO

| Parámetro | Default | Descripción |
|---|---|---|
| `lr` | `1e-4` | Learning rate (scheduler coseno) |
| `gamma` | `0.99` | Factor de descuento |
| `gae_lambda` | `0.95` | Factor λ para GAE |
| `eps_clip` | `0.2` | Clipping del ratio PPO |
| `k_epochs` | `10` | Épocas de actualización por batch |
| `kl_target` | `0.05` | Early-stop de épocas si KL divergencia > target |
| `value_clip` | `0.5` | Clipping del value loss |
| `entropy_coef` | `0.01` | Coeficiente de regularización por entropía |

### Normalización de observaciones

`RunningMeanStd` implementa el algoritmo de Welford para normalización online estable. La normalización se actualiza en cada `select_action` y se aplica al batch en cada `update`. Los parámetros se persisten en los checkpoints.

### Manejo de episodios truncados

El agente distingue entre `done` (terminación real) y `truncated` (timeout), aplicando **bootstrap del valor final** en episodios truncados para evitar sesgo en el cálculo de las ventajas GAE.

---

## Reward Shaping

**Archivo:** `src/reward_shaper.py`

El `CarlaRewardShaper` modifica la recompensa base del entorno para guiar el aprendizaje hacia una conducción segura y eficiente. Todos los datos provienen del **Waypoint API de CARLA** para máxima fidelidad.

### Componentes del reward

| Componente | Signo | Descripción |
|---|---|---|
| `alive_bonus` | `+` | Bonus por paso mientras se está en carretera |
| `speed_reward` | `+` | Gaussiana centrada en la velocidad objetivo (ajustada por curvatura) |
| `lane_centering` | `+` | Bonus proporcional al centramiento en el carril |
| `heading_alignment` | `+` | Bonus por alineación con la dirección del carril |
| `progress_bonus` | `+` | Bonus por cada 25 m recorridos |
| `smoothness_penalty` | `−` | Penalización por cambios bruscos de steering |
| `invasion_penalty` | `−` | Penalización por cruce de línea sólida (no intencional) |
| `road_penalty` | `−` | Penalización cuadrática por proximidad al borde + fuerte por salirse |
| `wrong_heading_penalty` | `−` | Penalización por heading >90° respecto al waypoint |
| `shield_intervention_pen` | `−` | Penalización proporcional a la divergencia de acción corregida |
| `idle_penalty` | `−` | Penalización por vehículo parado (con grace period post-shield) |
| `drift_penalty` | `−` | Penalización por conducción sistemáticamente hacia un borde |

### Soporte de cambio de carril

Durante maniobras de cambio de carril permitidas (marcas discontinuas), el reward shaper **suprime** automáticamente: `invasion_penalty`, `road_penalty` (borde) y `drift_penalty`. Esto evita que el agente aprenda que cambiar de carril es siempre incorrecto.

---

## Sensores semánticos

**Archivos:** `src/CARLA/Sensors/`

### SemanticLidarSensor

Wrapper sobre `sensor.lidar.ray_cast_semantic` de CARLA con 3 canales, 240 rayos/revolución a 20 Hz y un FOV vertical de ±15°. Cada punto del scan incluye la etiqueta semántica del objeto golpeado.

### SemanticLidarProcessor

Procesa la nube de puntos semántica de forma vectorizada (sin bucles Python) y produce un `SemanticScanResult` con los siguientes scans separados:

| Scan | Contenido |
|---|---|
| `combined` | Todos los obstáculos + bordes de calzada |
| `dynamic` | Vehículos (no-ego) + peatones + objetos dinámicos |
| `static` | Muros, vallas, postes, quitamiedos, señales |
| `pedestrian` | Solo peatones (señal de alta prioridad) |
| `road_edge` | Acera y terreno no-calzada |

Además proporciona distancias absolutas en metros: `nearest_vehicle_m`, `nearest_pedestrian_m`, `nearest_static_m`, `nearest_road_edge_m`.

---

## Métricas y monitorización

### LiveMetricsLogger

**Archivo:** `src/Metrics/live_metrics.py`

Persiste las métricas de entrenamiento en una base de datos SQLite por cada run, accesible en tiempo real desde el dashboard.

### SafetyMetrics

**Archivo:** `src/Metrics/EvalMetrics/SafetyMetrics.py`

Batería de métricas de seguridad calculadas al finalizar la evaluación:

- **Risk distribution**: fracción de pasos en nivel safe/warning/critical.
- **Shield interventions**: tasa de intervención, desglose por tipo semántico (dinámico, estático, peatón).
- **Minimum distance distribution**: media, mínimo, fracción por debajo de umbrales críticos.
- **Lane safety**: offset lateral, invasiones, tasa fuera de carretera.
- **Lane edge proximity**: distancias normalizadas a ambos bordes, asimetría (deriva).
- **Speed metrics**: media, máximo, cumplimiento del límite dinámico.
- **Hidden unsafe states**: detección de pasos seguros precursores de situaciones peligrosas.
- **Horizon effectiveness**: tasa de intervención por horizonte de predicción (1/5/10 pasos).

### Dashboard de entrenamiento (Streamlit)

**Archivo:** `dataVisualizer.py`

Dashboard interactivo con auto-refresco que visualiza en tiempo real las métricas de todos los runs activos e históricos. Requiere `streamlit` y `ollama` (opcional para análisis AI).

```bash
streamlit run dataVisualizer.py
```

### Dashboard de evaluación (matplotlib)

El script `main_eval.py` incluye un `CarlaDashboard` con:
- Polar plot del scan LIDAR en tiempo real.
- Velocímetro con límite de velocidad dinámico.
- Offset lateral y heading error.
- Contador de intervenciones del shield.

---

## Estructura del proyecto

```
CALA_SafeRL/
│
├── main_train.py               # Entrypoint de entrenamiento
├── main_eval.py                # Entrypoint de evaluación + dashboard matplotlib
├── dataVisualizer.py           # Dashboard Streamlit en tiempo real
├── export_data.py              # Exportar datos de TensorBoard a CSV/pandas
├── run_experiments.sh          # Script para lanzar los 3 experimentos en secuencia
├── utils.py                    # Utilidades generales
│
├── src/
│   ├── CARLA/
│   │   ├── Env/
│   │   │   └── carla_env.py            # Entorno Gymnasium principal
│   │   └── Sensors/
│   │       ├── carla_sensors.py        # SensorManager (gestión del ciclo de vida)
│   │       ├── SemanticLidarSensor.py  # Wrapper del sensor semántico
│   │       ├── SemanticLidarProcessor.py # Procesador vectorizado de nube de puntos
│   │       ├── SemanticScanResult.py   # Dataclass con resultado completo del scan
│   │       ├── CollisionSensor.py      # Sensor de colisiones
│   │       └── LaneInvasionSensor.py   # Sensor de invasión de carril
│   │
│   ├── Adaptative_Shield/
│   │   ├── adaptive_horizon_shield.py  # Escudo adaptativo principal ← COMPONENTE CLAVE
│   │   └── BicycleModel.py             # Modelo cinemático para predicción de trayectoria
│   │
│   ├── PPO/
│   │   ├── ppo_agent.py        # Agente PPO (update, GAE, scheduler)
│   │   ├── ActorCritic.py      # Red neuronal Actor-Critic con squashed Normal
│   │   └── RunningMeanStd.py   # Normalización online de observaciones (Welford)
│   │
│   ├── Metrics/
│   │   ├── live_metrics.py              # Logger SQLite en tiempo real
│   │   └── EvalMetrics/
│   │       ├── SafetyMetrics.py         # Cálculo de métricas de seguridad
│   │       └── metrics.py              # Reporter formateado para consola/logs
│   │
│   ├── reward_shaper.py         # Wrapper de reward shaping
│   └── safety_shield.py         # Shield básico (referencia; no es el principal)
│
├── tests/
│   └── test_reward_balance.py   # Tests de balance de recompensas
│
└── data/
    └── models/                  # Checkpoints y modelos finales (.pth)
```

---

## Instalación y requisitos

### Requisitos del sistema

- **CARLA Simulator** 0.9.14 o superior con el cliente Python correspondiente.
- **Python** 3.9+
- **GPU** recomendada para entrenamiento (compatible con CUDA y Apple MPS).

### Dependencias Python

```bash
pip install torch gymnasium numpy matplotlib streamlit plotly pandas
```

> **Nota:** El cliente CARLA (`carla`) debe instalarse desde el paquete `.egg` o `.whl` incluido en la distribución del simulador, o mediante pip si está disponible para tu versión:
> ```bash
> pip install carla==0.9.15
> ```

### Configuración del servidor CARLA

**Linux (headless, recomendado para servidores):**
```bash
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

**Windows:**
```cmd
CarlaUE4.exe -quality-level=Low
```

**Con puerto personalizado:**
```bash
./CarlaUE4.sh -RenderOffScreen -carla-port=2000
```

---

## Uso

### Entrenamiento

```bash
# Con escudo adaptativo (configuración recomendada para investigación)
python main_train.py \
    --model_name mi_agente \
    --shield_type adaptive \
    --map Town04 \
    --num_npc 20 \
    --max_episodes 2500 \
    --target_speed_kmh 50

# Sin escudo (baseline)
python main_train.py \
    --model_name baseline \
    --shield_type none \
    --map Town04 \
    --num_npc 20

# Con escudo básico
python main_train.py \
    --model_name basic \
    --shield_type basic \
    --front_threshold 0.15 \
    --side_threshold 0.04
```

#### Curriculum de entrenamiento

Por defecto se activa un **curriculum basado en rendimiento**:

| Fase | NPCs | Condición de avance |
|---|---|---|
| 1 | 0 | `off_road_rate < 20%` AND `avg_reward > 0` |
| 2 | 20 | `crash_rate < 10%` |
| 3 | `--num_npc` | Tráfico completo |

Para desactivar: `--no-curriculum`

#### Argumentos principales de entrenamiento

| Argumento | Default | Descripción |
|---|---|---|
| `--model_name` | *(requerido)* | Nombre base del modelo |
| `--shield_type` | `adaptive` | `none` / `basic` / `adaptive` |
| `--lr` | `1e-4` | Learning rate inicial |
| `--max_episodes` | `2500` | Episodios de entrenamiento |
| `--update_timestep` | `2048` | Timesteps entre actualizaciones PPO |
| `--map` | `Town04` | Mapa CARLA |
| `--num_npc` | `5` | NPCs de tráfico |
| `--curriculum` | activado | Curriculum basado en rendimiento |
| `--ckpt_freq` | `200` | Frecuencia de guardado de checkpoints |

### Evaluación

```bash
# Evaluación con escudo adaptativo
python main_eval.py \
    --model_name mi_agente_adaptive_final.pth \
    --shield_type adaptive \
    --episodes 20

# Evaluación sin render (solo métricas)
python main_eval.py \
    --model_name baseline_none_final.pth \
    --shield_type none \
    --no_render \
    --episodes 30
```

### Dashboard de monitorización en tiempo real

```bash
streamlit run dataVisualizer.py
```

### Visualización de curvas de entrenamiento

```bash
tensorboard --logdir ./runs
```

### Tests de balance de recompensa

```bash
python -m pytest tests/test_reward_balance.py -v
# o directamente:
python tests/test_reward_balance.py
```

---

## Experimentos

El script `run_experiments.sh` lanza los tres experimentos de comparación en secuencia:

```bash
# Todos los experimentos
bash run_experiments.sh

# Solo un experimento específico
bash run_experiments.sh baseline
bash run_experiments.sh basic
bash run_experiments.sh adaptive
```

**Variables de entorno configurables:**

```bash
CARLA_HOST=localhost CARLA_PORT=2000 MAP=Town04 MAX_EPISODES=1000 \
bash run_experiments.sh
```

### Checkpoints y artefactos

| Artefacto | Ubicación |
|---|---|
| Modelo final | `data/models/<nombre>_<shield>_final.pth` |
| Mejor checkpoint | `data/models/<nombre>_<shield>_best.pth` |
| Checkpoints periódicos | `data/models/<run_name>_checkpoints/ep_<N>.pth` |
| Métricas SQLite | `runs/<run_name>/metrics.sqlite` |
| Logs TensorBoard | `runs/<run_name>/` |

---

## Referencia de la cadena de wrappers

```python
from src.CARLA.Env.carla_env import CarlaEnv
from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.reward_shaper import CarlaRewardShaper

# Construir la cadena de wrappers
env = CarlaEnv(host="localhost", port=2000, map_name="Town04", num_npc_vehicles=20)
env = CarlaAdaptiveHorizonShield(env, front_threshold_base=0.15, side_threshold_base=0.04)
env = CarlaRewardShaper(env, target_speed_kmh=50.0)

# Interacción estándar Gymnasium
obs, info = env.reset()
action = agent.select_action(obs)
obs, reward, done, truncated, info = env.step(action)
```

---

## Notas sobre reproducibilidad

- El entorno usa `synchronous=True` con `fixed_delta_seconds=0.05` para garantizar determinismo en la simulación.
- El TrafficManager se inicializa con `set_random_device_seed(seed)` para reproducir el comportamiento de los NPCs.
- El argumento `--seed` se propaga a `random`, `numpy` y al entorno CARLA.
- Los checkpoints guardan tanto los pesos de la política como el estado del normalizador de observaciones (`RunningMeanStd`), asegurando que la evaluación sea consistente con el entrenamiento.
