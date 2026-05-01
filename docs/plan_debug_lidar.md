# Plan de debugging del visualizador LIDAR semántico

> Documento de análisis y plan de acción sobre el sensor LIDAR semántico de CARLA en el proyecto Shielded-RL / CALA_SafeRL y, en particular, sobre la interfaz de visualización de [main_eval.py](main_eval.py).
> Las fuentes consultadas se listan en [docs/fuentes_lidar.md](docs/fuentes_lidar.md).
>
> ⚠️ **Notas posteriores a la redacción original:**
> 1. **Tabla de tags semánticos**: las tablas mencionadas en este documento (DYNAMIC_TAGS={4, 10, 20}, etc.) son las de CARLA 0.9.10–0.9.13. CARLA 0.9.14 (12/2022) las reasignó a CityScapes. Ver `src/CARLA/Sensors/SemanticLidarProcessor.py` para los IDs vigentes.
> 2. **LIDAR bajo eliminado**: la versión v3 del sistema usaba dos LIDAR (alto a z=1.0 m y bajo a z=0.5 m). En evaluación se demostró que el bajo era totalmente redundante con el alto y se eliminó. Como consecuencia, OBS_DIM bajó de **979 → 739** y los modelos previos no son compatibles. Las referencias a `lidar_low`, `low_lidar_*`, `LOW_LIDAR_DIM`, `sensor_manager.lidar_low` y similares en este documento describen una arquitectura ya retirada.

---

## 0. Síntesis del estado actual

### 0.1. Qué hay y dónde

El proyecto monta dos LIDAR semánticos por vehículo ego, no uno solo:

| LIDAR | Posición | Rango | Canales | FOV vertical | Para detectar |
|-------|----------|-------|---------|--------------|---------------|
| Alto | Techo, `z=1.0 m` | 50 m | 3 | `[-15°, +5°]` | Vehículos, muros, obstáculos en altura |
| Bajo | Parachoques, `x=2.0, z=0.5 m` | 30 m | 4 | `[-30°, +10°]` | Guardarraíles bajos, bordillos, escombros |

Los archivos relevantes:

- [src/CARLA/Sensors/SemanticLidarSensor.py](src/CARLA/Sensors/SemanticLidarSensor.py) — wrapper sobre `sensor.lidar.ray_cast_semantic`. Spawnea el sensor con `attach_to=vehicle`, registra un callback que vuelca cada frame en `queue.Queue`, y al llamar `get_result()` drena la cola y procesa **el frame más reciente**.
- [src/CARLA/Sensors/SemanticLidarProcessor.py](src/CARLA/Sensors/SemanticLidarProcessor.py) — procesa un `SemanticLidarMeasurement` y produce un `SemanticScanResult`. Pipeline: parse con `np.frombuffer` (dtype estructurado de 24 B/punto) → filtro ego por `object_idx` → filtro de altura asimétrico (rechaza suelo) → bin angular (`atan2(-y, x)`) → 5 sub-scans (combined / dynamic / static / pedestrian / road_edge) con `np.minimum.at`.
- [src/CARLA/Sensors/SemanticScanResult.py](src/CARLA/Sensors/SemanticScanResult.py) — dataclass con los 5 scans + mínimos por arco + estadísticas globales.
- [src/CARLA/Sensors/carla_sensors.py](src/CARLA/Sensors/carla_sensors.py) — `SensorManager`: instancia el LIDAR alto, el bajo, el de colisión y el de invasión.
- [src/CARLA/Env/carla_env.py](src/CARLA/Env/carla_env.py) — entorno gymnasium. La observación es un vector de 979 dims; los primeros 960 son `[combined_alto, dyn_alto, stat_alto, combined_bajo]` (4×240).
- [main_train.py](main_train.py) — bucle PPO + curriculum + shield. Usa el `info` para registrar métricas y nada más; **no visualiza**.
- [main_eval.py](main_eval.py) — contiene la clase `CarlaDashboard` que sí dibuja un polar plot del LIDAR durante la evaluación.

### 0.2. Qué hace exactamente el visualizador

El método clave es `CarlaDashboard.update()` en [main_eval.py:176](main_eval.py#L176). Pasos:

1. Lee `obs[0:960]` y los reparte en cuatro arrays de 240 valores normalizados en `[0, 1]`:
   - `lidar_combined = obs[0:240]` (LIDAR alto, todos los grupos)
   - `lidar_dynamic  = obs[240:480]` (vehículos + peatones)
   - `lidar_static   = obs[480:720]` (muros, postes, guardarraíles altos…)
   - `lidar_low      = obs[720:960]` (LIDAR bajo combinado, range 30 m)
2. Reescala el LIDAR bajo: `lidar_low_scaled = lidar_low * (30/50)` para superponerlo con coherencia métrica sobre el eje radial de 0–50 m.
3. Lo dibuja con `set_theta_zero_location("N")` y `set_theta_direction(1)`: 0 rad arriba (frente), aumenta antihorario → 90° a la izquierda, 180° atrás, 270° a la derecha.
4. Añade un círculo de umbral pintado a `front_threshold` (por defecto 0.15 = 7.5 m).
5. Pinta speed bar, lateral marker e info textual.

### 0.3. Cómo se entrena (relación con el sensor)

[main_train.py](main_train.py) construye la cadena `CarlaEnv → Shield → CarlaRewardShaper`. En cada `step()`:

1. PPO propone una acción, se ejecuta `env.step(action)`.
2. Internamente `_build_observation()` llama `sensor_manager.get_semantic_result()` para el LIDAR alto y `get_low_semantic_result()` para el bajo.
3. El `info` que devuelve el env contiene `semantic_data_fresh`, `semantic_stale_reads`, `min_front_dynamic`, `nearest_vehicle_m`, `nearest_pedestrian_m`, etc., y se loguea en SQLite vía `LiveMetricsLogger`.
4. El shield (basic o adaptive) lee directamente esas distancias del `info` para decidir si interviene la acción.

Conclusión: el LIDAR es el "sentido" principal del agente y del shield. Si los scans están sucios o desfasados, contamina simultáneamente la política, el reward y el shield.

---

## 1. Documentación oficial de CARLA — hallazgos

Fuentes principales: [Core sensors](https://carla.readthedocs.io/en/latest/core_sensors/), [Sensors reference](https://carla.readthedocs.io/en/latest/ref_sensors/), [Synchrony and time-step](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/).

### 1.1. Ciclo de vida de un sensor

- Se spawnea como cualquier actor: `world.spawn_actor(bp, transform, attach_to=parent)`. El `transform` es **relativo al actor padre** cuando hay `attach_to`, y absoluto cuando no.
- Tipo de anclaje: `Rigid` (defecto, sigue exactamente al padre), `SpringArm` (con amortiguación, útil para cámaras), `SpringArmGhost` (sin colisión).
- `sensor.listen(callback)` registra una lambda que se ejecuta **cada vez que el sensor produce un frame**. El callback se ejecuta en un hilo del lado del cliente, no del simulador.
- `sensor.stop()` deja de escuchar; `sensor.destroy()` libera el actor en el servidor.

### 1.2. Modelo de datos del LIDAR semántico

Blueprint: `sensor.lidar.ray_cast_semantic`. Atributos de blueprint:

| Atributo | Default | Comentario |
|----------|---------|------------|
| `channels` | 32 | Número de láseres (anillos) |
| `range` | 10.0 | Distancia máxima en metros |
| `points_per_second` | 56000 | Total de puntos/segundo entre todos los canales |
| `rotation_frequency` | 10.0 | Hz de rotación |
| `upper_fov` / `lower_fov` | +10° / −30° | Apertura vertical |
| `horizontal_fov` | 360° | Apertura horizontal |
| `sensor_tick` | 0.0 | Periodo entre capturas; 0 ⇒ por defecto del simulador |

Fórmula clave: `points_per_channel_each_step = points_per_second / (FPS · channels)`.

Output: `carla.SemanticLidarMeasurement`. El payload es un buffer binario plano. Cada punto son **24 bytes**:

```
struct {
    float32 x;           // adelante (UE coord LH)
    float32 y;           // DERECHA  (UE coord LH)
    float32 z;           // arriba   (UE coord LH)
    float32 cos_inc_angle;
    uint32  object_idx;
    uint32  object_tag;  // CityScapes 0..28
}
```

A diferencia del LIDAR no semántico (`sensor.lidar.ray_cast`), el semántico **no tiene ruido, dropoff ni atenuación atmosférica**: cada hit que el ray-cast detecte se reporta tal cual.

### 1.3. Sistema de referencia y coordenadas

CARLA usa el sistema de Unreal: **left-handed**, `x` = adelante, `y` = derecha, `z` = arriba. Los puntos del LIDAR vienen en **espacio local del sensor**, no en el del mundo. Para llevarlos a mundo se compone con `sensor.get_transform().get_matrix()` (ver [lidar_to_camera.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py)).

Para visualizar en un sistema right-handed (matplotlib, Open3D), el ejemplo oficial [open3d_lidar.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/open3d_lidar.py) **invierte y**: `points = np.array([x, -y, z]).T`.

El procesador del proyecto hace lo mismo de otra forma: `angles = np.arctan2(-y, x)`, ya que para una indexación angular sólo importa el signo del giro.

### 1.4. Tabla completa de tags semánticos (CityScapes en CARLA)

| Valor | Tag | Color RGB |
|------:|-----|-----------|
| 0 | Unlabeled | (0,0,0) |
| 1 | Building | (70,70,70) |
| 2 | Fence | (100,40,40) |
| 3 | Other | (55,90,80) |
| 4 | Pedestrian | (220,20,60) |
| 5 | Pole | (153,153,153) |
| 6 | RoadLine | (157,234,50) |
| 7 | Road | (128,64,128) |
| 8 | SideWalk | (244,35,232) |
| 9 | Vegetation | (107,142,35) |
| 10 | Vehicles | (0,0,142) |
| 11 | Wall | (102,102,156) |
| 12 | TrafficSign | (220,220,0) |
| 13 | Sky | (70,130,180) |
| 14 | Ground | (81,0,81) |
| 15 | Bridge | (150,100,100) |
| 16 | RailTrack | (230,150,140) |
| 17 | GuardRail | (180,165,180) |
| 18 | TrafficLight | (250,170,30) |
| 19 | Static | (110,190,160) |
| 20 | Dynamic | (170,120,50) |
| 21 | Water | (45,60,150) |
| 22 | Terrain | (145,170,100) |

Hallazgo importante: el procesador del proyecto **no contempla el tag 14 (Ground)** ni 6 (RoadLine) ni 7 (Road). Para 6 y 7 está bien (son carretera transitable, no se quieren como obstáculo). Para 14 (Ground) sí puede ser un agujero: en algunos mapas se marca terreno no carretera con tag 14 y el agente no lo "ve" como bordillo. Es candidato a añadir a `ROAD_EDGE_TAGS` después de inspeccionar `tag_counts` en [src/CARLA/Sensors/SemanticLidarProcessor.py:179](src/CARLA/Sensors/SemanticLidarProcessor.py#L179).

### 1.5. `object_idx`: ojo, no es exactamente `actor.id`

Discusión en [Issue #3191](https://github.com/carla-simulator/carla/issues/3191) y [Issue #5094](https://github.com/carla-simulator/carla/issues/5094):

- El `object_idx` que devuelve el LIDAR semántico es `FActorInfo->Description.UId`, no necesariamente `carla.Actor.id`.
- Para **vehículos y peatones** registrados en el `FActorRegistry` el índice suele coincidir con `actor.id` y el filtro `idx != ego_id` funciona.
- Para **objetos creados por blueprint** que no están en el registry (paredes, postes, vegetación de mapa), muchos colapsan a `object_idx = 0`. Esto **no afecta al filtro ego** (porque el ego es un actor registrado), pero sí a cualquier intento futuro de "rastrear" un obstáculo concreto por idx.

Verificar en runtime si el ego sale efectivamente filtrado: contar puntos con `idx == ego_id` antes de aplicar el filtro y loggearlo durante un sprint. Si vale 0 sistemáticamente, el ego es invisible al sensor (correcto). Si crece, hay puntos del cuerpo propio en el scan.

### 1.6. Modo síncrono y patrón canónico de cola

El ejemplo [synchronous_mode.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py) define el **patrón de referencia** para garantizar consistencia frame-a-frame:

```python
def _retrieve_data(self, sensor_queue, timeout):
    while True:
        data = sensor_queue.get(timeout=timeout)
        if data.frame == self.frame:   # frame del world tick
            return data
```

Es decir:
1. `world.tick()` devuelve un frame nuevo.
2. Para cada sensor, `queue.get(timeout=...)` **bloquea** hasta que llega un frame.
3. Se **descarta** todo frame cuyo `data.frame` no coincida con el del world tick.
4. Antes de avanzar se asserta que todos los sensores entregaron datos del mismo frame.

Esto es **diferente** de lo que hace el proyecto en [SemanticLidarSensor.get_result()](src/CARLA/Sensors/SemanticLidarSensor.py#L75): drena la cola sin bloquear con `get_nowait()` y se queda con el último elemento; si la cola está vacía, devuelve el resultado anterior y marca `_last_was_fresh = False`.

Implicaciones para el debugging: ver §4.2.

---

## 2. Repositorios de referencia

| Repo | Aporte que importa para nuestro visualizador |
|------|----------------------------------------------|
| [carla-simulator/carla — open3d_lidar.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/open3d_lidar.py) | Patrón canónico de parseo del LIDAR semántico con `np.dtype` estructurado y mapeo de los 23 tags a colores. Confirma la inversión de `y` para visualizar. |
| [carla-simulator/carla — lidar_to_camera.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py) | Ejemplo limpio de transformación sensor → mundo → cámara con `get_matrix()` y `get_inverse_matrix()`. |
| [carla-simulator/carla — synchronous_mode.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py) | Patrón de cola con timeout y assert de frame match para múltiples sensores. |
| [angelomorgado/CARLA-Sensor-Visualization](https://github.com/angelomorgado/CARLA-Sensor-Visualization) | Arquitectura modular sensor / vehicle / display en pygame, callback por sensor. Útil como referencia de estructura. |
| [fnozarian/CARLA-KITTI](https://github.com/fnozarian/CARLA-KITTI) | Aporta visualización BEV (bird's eye view) y proyección de point clouds — alternativa al polar plot. |
| [DaniCarias/CARLA_MULTITUDINOUS](https://github.com/DaniCarias/CARLA_MULTITUDINOUS) | Pipeline de adquisición y voxelización para datasets ground-truth. |
| [MukhlasAdib/CARLA-2DBBox](https://github.com/MukhlasAdib/CARLA-2DBBox) | Filtros típicos por `object_idx`/`object_tag` para extraer bounding boxes 2D, útil como ejemplo de cómo otros usan los índices. |

Survey académico: [arxiv 2509.08221 — A Comprehensive Review of RL for Autonomous Driving in CARLA](https://arxiv.org/abs/2509.08221) confirma que >80% del estado del arte usa modelos sin modelo (PPO, DQN, SAC) y que el LIDAR es la modalidad sensorial más extendida en seguro-RL.

---

## 3. Estándar de verificación del visualizador LIDAR

A partir de los hallazgos anteriores, defino un **checklist de verificación** que cualquier dashboard de LIDAR semántico debería pasar antes de considerarse correcto. Va de menor a mayor sofisticación.

### V1. Geometría y orientación

- [ ] **V1.1** Un objeto colocado a 10 m **delante** del coche aparece arriba (12 en punto) en el polar plot, a radio ≈ 0.2 (10/50).
- [ ] **V1.2** Un objeto colocado a 10 m **a la izquierda** aparece a las 9 en punto.
- [ ] **V1.3** Un objeto a 10 m **a la derecha** aparece a las 3 en punto.
- [ ] **V1.4** Un objeto **detrás** aparece a las 6 en punto.
- [ ] **V1.5** Si el coche gira a la izquierda, el objeto frontal estático del mundo "se mueve" a la derecha del polar plot (porque está en marco egocéntrico).

### V2. Frescura y sincronía

- [ ] **V2.1** En modo síncrono, en cada step `info["semantic_data_fresh"] == True`. Si no, hay desincronización entre `world.tick()` y el callback.
- [ ] **V2.2** El ratio `semantic_stale_reads / (stale + fresh)` debe ser **<1%** en una corrida de 200 episodios. Cualquier valor mayor indica deriva.
- [ ] **V2.3** El `semantic_last_frame` debe avanzar en exactamente +1 por step en modo síncrono.

### V3. Filtro ego

- [ ] **V3.1** Inyectar logging temporal en [SemanticLidarProcessor.process()](src/CARLA/Sensors/SemanticLidarProcessor.py#L106) que cuente `np.sum(idx == ego_id)` antes del filtro. Debe ser >0 en todos los frames (el ego siempre es visible para sus propios sensores).
- [ ] **V3.2** Después del filtro, `np.sum(idx == ego_id)` debe ser exactamente 0.
- [ ] **V3.3** Tras un `reset()`, comprobar que `update_ego_id` se ha llamado y `processor.ego_id == new_vehicle.id`. **El proyecto NO llama `update_ego_id` actualmente** — el filtro funciona porque `SensorManager` se reconstruye en cada reset, pero conviene auditar.

### V4. Filtro de altura

- [ ] **V4.1** En recta llana sin obstáculos, `min_front_dynamic` y `min_front_static` deben valer 1.0 (no hit). Si caen <1.0 sin objetos visibles, hay falsos positivos del suelo.
- [ ] **V4.2** Verificar con `tag_counts` en `info["semantic_tag_counts"]`: el conteo de tag 7 (Road) debería ser ≈0 tras el filtro de altura. Si no, el filtro deja pasar suelo.

### V5. Coherencia entre LIDAR alto y bajo

- [ ] **V5.1** Si hay un guardarraíl bajo a 5 m en frente, `lidar_low_combined[0]` debería tener un hit pero `lidar_combined[0]` (alto) podría no — porque el alto tiene `lower_fov=-15°` y a 5 m el rayo ya pasa por encima.
- [ ] **V5.2** Si hay un camión alto a 30 m en frente, **ambos** deberían detectarlo. El bajo lo verá entre 0–30 m, el alto entre 0–50 m.
- [ ] **V5.3** El reescalado `low * 30/50` en el plot **introduce un anillo aparente a radio 0.6** cuando el bajo no detecta nada. Esto es un artefacto: tratar como TODO en §5.

### V6. Categorías semánticas

- [ ] **V6.1** Acercar un peatón al coche en CARLA y verificar que `nearest_pedestrian_m` baja monotónicamente.
- [ ] **V6.2** Pasar al lado de una acera con `tag=8` y verificar que `nearest_road_edge_m` < lidar_range y que `min_r_side_road_edge` o `min_l_side_road_edge` se reduce.
- [ ] **V6.3** Inspeccionar `tag_counts` durante 10 episodios: identificar tags inesperados (p. ej. 14 Ground, 16 RailTrack, 21 Water) y decidir si añadirlos a algún grupo.

### V7. Umbrales y semántica del shield

- [ ] **V7.1** El círculo rojo del polar plot está dibujado a `front_threshold` pero **no** representa el umbral lateral (`side_threshold`) ni el del shield adaptativo. Para evitar confusión visual, debería pintarse como un arco frontal (±FRONT_N bins ≈ ±22.5°), no como un círculo completo.
- [ ] **V7.2** El umbral lateral marcado en el `lat marker` (líneas naranjas a ±0.82) corresponde al `lateral_threshold` antiguo del shield basic. El argument default actual de [main_train.py](main_train.py) es 0.65. Hay que sincronizar la línea visual con el valor real activo en el shield.

---

## 4. Plan de debugging por fases

> Ejecutar las fases en orden. Cada fase tiene comandos / parches concretos y una condición de salida.

### Fase A — Instanciación y anclaje del sensor

**Objetivo**: confirmar que los dos LIDAR están vivos, atados al ego correcto y configurados como esperamos.

**A1. Inspección de actores activos**
Añadir temporalmente al final de [SensorManager.__init__](src/CARLA/Sensors/carla_sensors.py#L78):

```python
import logging
logger.info(
    f"[LIDAR_DBG] alto: id={self.lidar.sensor.id} "
    f"parent={self.lidar.sensor.parent.id if self.lidar.sensor.parent else None} "
    f"transform={self.lidar.sensor.get_transform()}"
)
logger.info(
    f"[LIDAR_DBG] bajo: id={self.lidar_low.sensor.id} "
    f"parent={self.lidar_low.sensor.parent.id if self.lidar_low.sensor.parent else None} "
    f"transform={self.lidar_low.sensor.get_transform()}"
)
```

Salida esperada: `parent` igual al `ego_vehicle.id`. Transforms con `z=1.0` para alto y `x≈2.0, z≈0.5` para bajo (en coordenadas de mundo, sumando la posición del ego).

**A2. Verificar atributos blueprint reales**
Insertar en `SemanticLidarSensor.__init__` justo antes de `world.spawn_actor`:

```python
logger.info(f"[LIDAR_DBG] bp.attrs={ {a.id: bp.get_attribute(a.id).as_string() for a in bp} }")
```

Comprobar que `channels`, `range`, `rotation_frequency`, `upper_fov`, `lower_fov`, `points_per_second` coinciden con los configurados.

**A3. Cuenta de actores tras `reset()`**
En [CarlaEnv._cleanup](src/CARLA/Env/carla_env.py#L954) y `reset()` añadir:

```python
sensor_actors = [a for a in self.world.get_actors() if "sensor.lidar" in a.type_id]
logger.info(f"[LIDAR_DBG] sensores LIDAR vivos: {len(sensor_actors)}")
```

Tras 10 episodios, el número debe quedarse estable en 2 (alto + bajo). Si crece, hay leak de actores entre episodios — es la causa más típica de `RuntimeError: time-out` en CARLA.

**Condición de salida A**: los tres logs muestran lo esperado y no hay leak.

---

### Fase B — Callbacks y sincronía

**Objetivo**: confirmar que cada `world.tick()` produce un frame fresco en cada uno de los dos LIDAR y que `_build_observation()` consume el frame del tick actual, no uno previo.

**B1. Tasa de frames frescos**
Añadir al final de [main_train.py](main_train.py) o en [main_eval.py](main_eval.py) en cada step:

```python
if step % 50 == 0:
    logger.info(
        f"[LIDAR_DBG] fresh={info['semantic_data_fresh']} "
        f"stale_reads={info['semantic_stale_reads']} "
        f"fresh_reads={info['semantic_fresh_reads']} "
        f"last_frame={info['semantic_last_frame']}"
    )
```

Esperado: en modo síncrono `fresh=True` casi siempre (>99%). Si `stale_reads` crece linealmente, ver B2.

**B2. Frame match**
Modificar [SemanticLidarSensor.get_result()](src/CARLA/Sensors/SemanticLidarSensor.py#L75) para alinear con el patrón canónico de CARLA:

```python
def get_result(self, expected_frame: int = None, timeout: float = 1.0) -> SemanticScanResult:
    """
    Si expected_frame se pasa, bloquea hasta que llega un frame con ese número
    (descarta los antiguos). Esto fuerza la consistencia tick-a-tick.
    """
    if expected_frame is None:
        return self._legacy_get()  # comportamiento actual
    while True:
        try:
            data = self._queue.get(timeout=timeout)
        except queue.Empty:
            self._stale_reads += 1
            return self._last
        if int(getattr(data, "frame", -1)) == expected_frame:
            self._last = self.processor.process(data)
            self._last_was_fresh = True
            self._fresh_reads += 1
            self._last_frame = expected_frame
            return self._last
        # frame antiguo, lo descartamos y seguimos esperando
```

Y en [CarlaEnv.step()](src/CARLA/Env/carla_env.py#L301):

```python
frame = self.world.tick()
sem = self.sensor_manager.lidar.get_result(expected_frame=frame)
sem_low = self.sensor_manager.lidar_low.get_result(expected_frame=frame)
```

Esto **garantiza** que tanto alto como bajo entregan datos del mismo frame que la simulación.

**B3. Smoke test de no-blocking**
Bajar `rotation_frequency` a 5 Hz y verificar que el agente sigue avanzando (con `stale_reads` creciendo). Si no, hay un deadlock — el `queue.get(timeout=1.0)` con 1 s es generoso pero deja crashear si CARLA se cuelga.

**Condición de salida B**: ratio `fresh / (fresh+stale)` > 99% durante un episodio largo.

---

### Fase C — Parámetros de configuración

**Objetivo**: verificar que la resolución angular, vertical y radial del LIDAR coincide con la suposición del visualizador y del shield.

**C1. Resolución angular**
Para `num_rays=240` y `rotation_frequency=20.0`, el `points_per_second` debe ser `num_rays · channels · rotation_frequency`. El proyecto lo computa así en [SemanticLidarSensor:55-59](src/CARLA/Sensors/SemanticLidarSensor.py#L55) — correcto. Pero CARLA reparte los puntos por step según `pps / FPS / channels`. Con `FPS=20`, `pps=240·3·20=14400`, `puntos por canal por step = 14400 / 20 / 3 = 240`, **uno por bin** — lo justo. Si el FPS real fluctuara hacia abajo (CPU saturada) habría sub-muestreo; si fluctúa hacia arriba habría duplicados (bins con varios hits, resueltos por `np.minimum.at`).

Acción recomendada: añadir log del **número total de puntos recibidos por frame**:

```python
n_pts = len(np.frombuffer(latest.raw_data, dtype=_SEMANTIC_DTYPE))
logger.debug(f"[LIDAR_DBG] frame={latest.frame} pts={n_pts}")
```

Esperado para alto: ~720 (3 canales · 240 bins). Para bajo: ~960 (4 canales · 240 bins).

**C2. FOV vertical y filtro de altura**
La constante `Z_ABOVE_MAX = 1.5` significa: rechazamos hits a más de 1.5 m por encima del sensor. Para el LIDAR alto a `z=1.0`, eso es 2.5 m sobre el suelo — apenas la altura de un techo de furgoneta. Si en el futuro se quiere detectar puentes o señales colgantes, hay que subirla.

`GROUND_CLEARANCE = 0.15` (15 cm sobre la calzada): hits debajo de eso se descartan. Para una calzada perfectamente plana, tan ajustado puede dar problemas en cuestas o badenes. Añadir log en debug para detectar si en algún mapa este filtro se vuelve agresivo.

**C3. Coherencia eje radial visualizador**
En [main_eval.py:114-115](main_eval.py#L114), las etiquetas radiales son `["5 m", "15 m", "25 m", "35 m", "45 m"]` para ticks `[0.1, 0.3, 0.5, 0.7, 0.9]`. Esto es correcto **sólo** para el LIDAR alto (range 50 m). Para el bajo (range 30 m) los mismos ticks corresponden a 3, 9, 15, 21, 27 m. Como se reescala con `lidar_low * 30/50`, las etiquetas siguen siendo válidas en el eje del plot, pero hay que dejar claro en la leyenda que el bajo está reescalado y que su "no hit" cae en radio 0.6.

**Condición de salida C**: número de puntos por frame coherente con la fórmula, filtros de altura no falsean obstáculos en mapa plano, etiquetas radiales coherentes con cada LIDAR.

---

### Fase D — Transformaciones geométricas

**Objetivo**: garantizar que la proyección de un hit semántico a (bin, distancia) y de ahí al polar plot es correcta en sentido y orientación.

**D1. Test sintético en pizarra**
Inyectar 4 puntos sintéticos en `SemanticLidarProcessor.process` (modo debug):

| Punto | (x, y, z) en sensor | Esperado |
|-------|---------------------|----------|
| Frente | (10, 0, 0) | bin 0, dist 10 |
| Izquierda | (0, −10, 0) | bin 60, dist 10 |
| Atrás | (−10, 0, 0) | bin 120, dist 10 |
| Derecha | (0, 10, 0) | bin 180, dist 10 |

Cubrir como test unitario en `tests/`:

```python
def test_processor_angular_convention():
    p = SemanticLidarProcessor(num_rays=240, lidar_range=50.0, ego_id=-1)
    raw = np.array(
        [(10, 0, 0, 1, 99, 11),
         (0, -10, 0, 1, 99, 11),
         (-10, 0, 0, 1, 99, 11),
         (0, 10, 0, 1, 99, 11)],
        dtype=_SEMANTIC_DTYPE,
    )
    class M: raw_data = raw.tobytes(); frame = 1
    res = p.process(M())
    assert res.combined[0]   < 1.0   # frente
    assert res.combined[60]  < 1.0   # izquierda
    assert res.combined[120] < 1.0   # atrás
    assert res.combined[180] < 1.0   # derecha
```

**D2. Coherencia entre dashboard y procesador**
El dashboard usa `np.linspace(0, 2π, 240, endpoint=False)`. El procesador usa `bin_idx = (angle / (2π/240)).astype(int) % 240`. Ambas convenciones están alineadas si y sólo si `set_theta_zero_location("N")` y `set_theta_direction(1)`. **Está correcto** en [main_eval.py:71-72](main_eval.py#L71). Documentar como invariante en un comentario.

**D3. Reescalado del LIDAR bajo**
Hoy: `lidar_low_scaled = lidar_low * (30/50)`. Cuando no hay hit, `lidar_low = 1.0` ⇒ `scaled = 0.6`. Resultado: aparece un círculo continuo a radio 0.6, **incluso sin obstáculos**. Es un artefacto cosmético pero engañoso. Dos arreglos posibles:

  - **Opción A (sencilla)**: enmascarar los "no hit" antes de plotear:
    ```python
    mask = lidar_low < 1.0
    self.lidar_low_line.set_data(self.angles[mask], lidar_low_scaled[mask])
    ```
    El plot dibujará puntos sólo donde hay detección.

  - **Opción B (más precisa)**: usar un eje radial secundario o representar el bajo en su propia gráfica de 0–30 m.

Recomendado: **Opción A** por simplicidad.

**D4. Umbral del shield**
El círculo rojo a `front_threshold` solo aplica al frente (FRONT_N=15 bins, ±22.5°). Pintar como un wedge:

```python
# wedge frontal de ±22.5° a radio front_threshold
fov = (15 / 240) * 2 * np.pi  # half-angle del front
theta_w = np.linspace(-fov, fov, 100)
self.ax_lidar.fill_between(theta_w, 0, front_threshold, color="red", alpha=0.15)
```

Y, si shield es adaptativo, leer el umbral dinámico de `info["dynamic_front_threshold"]` (si el shield lo expone) y refrescarlo cada step.

**Condición de salida D**: el test sintético pasa, el plot ya no muestra el anillo cosmético del LIDAR bajo, y la zona roja del umbral es un wedge frontal coherente con la geometría.

---

### Fase E — Validación end-to-end

**Objetivo**: comprobar que con todo lo anterior aplicado, el visualizador refleja fielmente lo que ve el agente.

**E1. Escenario controlado**
Lanzar el simulador con `--num_npc 0` en un mapa vacío (Town01 funciona bien). Conducir el ego con el autopilot (set_autopilot True provisional). Observar:

- LIDAR alto: silencio total fuera de ±lateral_threshold.
- LIDAR bajo: detecta bordillos y aceras, tras un giro brusco se ve la acera correcta a las 9/3.

**E2. Escenario denso**
Lanzar con `--num_npc 40` (default training). Verificar:

- `nearest_vehicle_m` < 50 cuando hay un coche cerca.
- `nearest_pedestrian_m` < 50 sólo cerca de cruces con peatones; en autopista debe ser 999.0.

**E3. Cruce con `tag_counts`**
Imprimir `info["semantic_tag_counts"]` durante 1000 steps en Town04 y comparar con la tabla del §1.4. Si aparecen tags 14 (Ground), 16 (RailTrack) o 21 (Water) con cuentas no triviales, decidir si añadirlos a `STATIC_OBS_TAGS` o `ROAD_EDGE_TAGS`.

**Condición de salida E**: el dashboard, ejecutado durante un episodio completo con `python main_eval.py --no_dashboard False`, no muestra incoherencias visuales y los logs de `[LIDAR_DBG]` confirman frescura > 99% y filtro ego efectivo.

---

## 5. Plan de acción ejecutable (resumen accionable)

Si se quiere atacar el problema en una sola sesión y por orden de impacto:

1. **Aplicar B2** (`get_result(expected_frame)`) y propagarlo en `_build_observation`. Es el cambio que más impacta en correctitud, porque elimina la posibilidad de mostrar y entrenar con datos de frames anteriores.
2. **Aplicar D3 (Opción A)** para que el LIDAR bajo no pinte el anillo fantasma a 0.6.
3. **Aplicar D4** para que la zona roja del umbral sea un wedge frontal y, opcionalmente, sea dinámica.
4. **Añadir el test sintético D1** en `tests/test_lidar_processor.py`. Es regresión barata y duradera.
5. **Activar logs `[LIDAR_DBG]` (A1, A2, A3, B1)** durante el siguiente sprint y registrar en SQLite un par de métricas: `lidar_high_pts_per_frame`, `lidar_low_pts_per_frame`, `lidar_stale_ratio`. Permite detectar regresiones futuras sin tocar código.
6. Inspeccionar **`tag_counts`** durante una corrida completa para confirmar/rechazar la inclusión del tag 14 (Ground) en `ROAD_EDGE_TAGS`.
7. **Sincronizar la línea visual del lateral threshold** con el valor real activo en el shield (lectura dinámica desde args o `info`).

Pasos opcionales pero recomendables:

8. Añadir `update_ego_id` al flujo de `reset()` para no depender de la reconstrucción de `SensorManager` (defensa en profundidad).
9. Mostrar en el dashboard los sub-scans `pedestrian_scan` y `road_edge_scan` que ya están calculados y no se exhiben.
10. Renderizar también la frescura del frame (`semantic_data_fresh`) como un punto verde/rojo en el plot.

---

## 6. Riesgos y notas

- **Cambiar el patrón de cola** (paso 1) puede introducir un timeout duro si CARLA se cuelga. Mantener un timeout generoso (1.0 s) y un log de `queue.Empty` para detectarlo, pero no deshabilitar la sincronía: la corrección > la robustez ante CARLA enfermo.
- **El filtro por `object_idx`** funciona hoy porque el ego es un actor registrado. Si un día se cambia el blueprint del ego o se usa un actor no registrado, el filtro fallará silenciosamente y el ego aparecerá como obstáculo a ~0 m. Añadir el contador `n_ego_pts_filtered` en `SemanticScanResult` ayudaría a auditarlo.
- **Tag 14 (Ground)** y **tag 16 (RailTrack)** no están agrupados. Antes de añadirlos, medir su prevalencia en el mapa de entrenamiento — añadirlos a `STATIC_OBS_TAGS` puede meter ruido a 360°.
- **El LIDAR bajo a `z=0.5 m`** está dentro del bounding box del Tesla Model 3 (longitud ≈4.7 m). Si en algún momento se intenta detectar el suelo bajo el chasis del propio coche, el filtro asimétrico lo rechazará — eso es deseado.
