# Fuentes — Investigación sobre LIDAR semántico de CARLA

> Lista completa de las fuentes consultadas para producir [docs/plan_debug_lidar.md](plan_debug_lidar.md).
> Última consulta: abril de 2026.

---

## 1. Documentación oficial de CARLA

| Fuente | Qué aporta |
|--------|------------|
| [Core sensors — CARLA Read the Docs](https://carla.readthedocs.io/en/latest/core_sensors/) | Ciclo de vida de un sensor: spawn con `attach_to`, callbacks vía `sensor.listen()`, tipos de attachment (Rigid / SpringArm / SpringArmGhost), atributos comunes (frame, timestamp, transform). |
| [Sensors reference — CARLA Read the Docs](https://carla.readthedocs.io/en/latest/ref_sensors/) | Atributos del blueprint `sensor.lidar.ray_cast_semantic`: `channels`, `range`, `points_per_second`, `rotation_frequency`, `upper_fov`, `lower_fov`, `horizontal_fov`, `sensor_tick`. Estructura de `SemanticLidarMeasurement`: 4 floats + 2 uint32 por punto. |
| [Sensors reference — CARLA UE5](https://carla-ue5.readthedocs.io/en/latest/ref_sensors/) | Tabla **completa** de tags semánticos CityScapes (0..22) con colores y descripción. Confirma fórmula `points_per_channel_each_step = points_per_second / (FPS · channels)`. |
| [Synchrony and time-step — CARLA](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/) | Patrón canónico de `queue.Queue() + sensor.listen(queue.put) + queue.get(timeout)` en modo síncrono; advertencia de delay de varios frames en sensores GPU; `world.tick()` bloquea hasta que el cliente confirma. |
| [Tutorial — Retrieve simulation data](https://github.com/carla-simulator/carla/blob/master/Docs/tuto_G_retrieve_data.md) | Patrones básicos de spawn de sensores y callbacks. |

## 2. Ejemplos oficiales de CARLA (PythonAPI/examples)

| Fuente | Qué aporta |
|--------|------------|
| [open3d_lidar.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/open3d_lidar.py) | Patrón canónico de parseo del LIDAR semántico con `np.dtype([('x', f32), ('y', f32), ('z', f32), ('CosAngle', f32), ('ObjIdx', u32), ('ObjTag', u32)])`. Inversión de `y` para visualizar (LH → RH). Mapa de colores LABEL_COLORS. |
| [lidar_to_camera.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py) | Composición de transformaciones: `lidar.get_transform().get_matrix()` y `camera.get_transform().get_inverse_matrix()`. Pipeline sensor → mundo → cámara. |
| [synchronous_mode.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py) | Patrón canónico `CarlaSyncMode`: cola por sensor, `queue.get(timeout)` bloqueante, `assert all(x.frame == self.frame for x in data)`. Esta es la referencia explícita usada para diseñar la fase B del plan. |

## 3. Issues y discusiones de CARLA en GitHub

| Fuente | Qué aporta |
|--------|------------|
| [Issue #3191 — Semantic Lidar Object ObjIdx Definition and Uniqueness](https://github.com/carla-simulator/carla/issues/3191) | Confirma que `object_idx` devuelve `FActorInfo->Description.UId`, **no** necesariamente `actor.id`. Reportes de duplicados de id. |
| [Issue #5094 — Identifying objects created by blueprints via semantic lidar](https://github.com/carla-simulator/carla/issues/5094) | Objetos creados por blueprint que no están en `FActorRegistry` colapsan a `object_idx = 0`. Limitación reconocida y no resuelta. |
| [Issue #4067 — Lidar and semantic lidar results are different at one simulation](https://github.com/carla-simulator/carla/issues/4067) | Confirma que el LIDAR semántico y el no semántico no comparten el mismo número de puntos por escaneo. |
| [Issue #7449 — sensor.lidar.ray_cast_semantic and sensor.lidar.ray_cast collect different lengths](https://github.com/carla-simulator/carla/issues/7449) | Documenta la diferencia de tamaño entre payloads de los dos LIDAR (ray_cast filtra puntos sin hit; semántico no). |
| [Issue #7188 — How to visualize multiple Lidar data at the same time](https://github.com/carla-simulator/carla/issues/7188) | Discusión sobre múltiples LIDAR concurrentes y problemas de sincronización entre callbacks. |
| [Issue #5026 — Understanding how sensors and lidar work in CARLA](https://github.com/carla-simulator/carla/issues/5026) | FAQ general sobre comportamiento del LIDAR. |
| [Issue #1751 — add semantic label to lidar point cloud](https://github.com/carla-simulator/carla/issues/1751) | Histórico: cómo se introdujo la etiqueta semántica en el LIDAR. |
| [Issue #4001 — sensors took too long to send their data](https://github.com/carla-simulator/carla/issues/4001) | Errores típicos de timeout en modo síncrono y mitigaciones. |
| [Issue #2809 — Synchronous mode tick hangs](https://github.com/carla-simulator/carla/issues/2809) | Causas comunes de bloqueo en `world.tick()`. |
| [Issue #3653 — Camera RGB sensor_tick not in synch with fixed_delta_seconds](https://github.com/carla-simulator/carla/issues/3653) | Por qué `sensor_tick` puede desincronizarse con `fixed_delta_seconds`. |
| [Issue #7108 — Using sensors with low data rate in synchronous mode](https://github.com/carla-simulator/carla/issues/7108) | Buenas prácticas cuando el sensor no produce a la velocidad del tick. |
| [Issue #7736 — Question About Carla's Synchronous Mode](https://github.com/carla-simulator/carla/issues/7736) | Aclaraciones recientes sobre el orden de eventos en modo síncrono. |
| [Issue #1018 — Lidar sensor possibly wrongly modeled in carla](https://github.com/carla-simulator/carla/issues/1018) | Discusión histórica del modelo del LIDAR. |

## 4. Doxygen / API C++

| Fuente | Qué aporta |
|--------|------------|
| [SemanticLidarDetection class reference](https://carla.org/Doxygen/html/dd/d3c/classcarla_1_1sensor_1_1data_1_1SemanticLidarDetection.html) | Tipos exactos de los campos: `point: geom::Location`, `cos_inc_angle: float`, `object_idx: uint32_t`, `object_tag: uint32_t`. |

## 5. Repositorios de código abierto relacionados

| Fuente | Qué aporta |
|--------|------------|
| [angelomorgado/CARLA-Sensor-Visualization](https://github.com/angelomorgado/CARLA-Sensor-Visualization) | Arquitectura modular sensor / vehicle / display en pygame; callback por sensor; soporte para RGB, LIDAR, Radar, GNSS, IMU, colisión, invasión. |
| [fnozarian/CARLA-KITTI](https://github.com/fnozarian/CARLA-KITTI) | Generación de datos sintéticos al estilo KITTI; visualización BEV de point clouds. |
| [DaniCarias/CARLA_MULTITUDINOUS](https://github.com/DaniCarias/CARLA_MULTITUDINOUS) | Pipeline de adquisición y voxel-grid ground truth. |
| [MukhlasAdib/CARLA-2DBBox](https://github.com/MukhlasAdib/CARLA-2DBBox) | Anotador automático de bounding boxes 2D usando filtros por `object_idx` / `object_tag`. |
| [joedlopes/carla-simulator-multimodal-sensing](https://github.com/joedlopes/carla-simulator-multimodal-sensing) | Detección de vehículos con fusión RGB + LIDAR en CARLA. |
| [nsteve2407/CARLA_Localization_and_Mapping](https://github.com/nsteve2407/CARLA_Localization_and_Mapping) | Variante de `open3d_lidar.py` adaptada para localización y mapeo. |
| [ActuallySam/Carla-Reinforcement-Learning](https://github.com/ActuallySam/Carla-Reinforcement-Learning) | Variante de `open3d_lidar.py` integrada con flujo de RL. |
| [idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning](https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning) | PPO en CARLA from scratch — referencia para benchmarking del enfoque. |
| [yanlai00/RL-Carla](https://github.com/yanlai00/RL-Carla) | RL + recolección de datos en CARLA. |

## 6. Literatura científica

| Fuente | Qué aporta |
|--------|------------|
| [arXiv 2509.08221 — A Comprehensive Review of RL for Autonomous Driving in CARLA](https://arxiv.org/abs/2509.08221) ([HTML](https://arxiv.org/html/2509.08221v1)) | Revisión sistemática de ~100 papers que usan CARLA + RL. Confirma que >80% emplean métodos sin modelo (PPO, DQN, SAC). El LIDAR es la modalidad sensorial más extendida en seguro-RL. |
| [MDPI Applied Sciences 15(16):8972 — Deep RL & IL for Autonomous Driving in CARLA](https://www.mdpi.com/2076-3417/15/16/8972) | Revisión complementaria sobre integración de RL e IL en CARLA. |
| [OpenDriveLab — End-to-end Autonomous Driving / papers.md](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving/blob/main/papers.md) | Lista curada de papers end-to-end (incluye trabajos basados en CARLA + LIDAR). |
| [Hitesh Kumar — Neurips ML4AD 2019 poster](https://hitesh11.github.io/pdfs/Neurips%202019%20ML4AD%20Poster.pdf) | Modelo arquitectural temprano sobre RL en CARLA — antecedente histórico. |

## 7. Releases y notas oficiales de CARLA

| Fuente | Qué aporta |
|--------|------------|
| [CARLA 0.9.10 release notes](https://carla.org/2020/09/25/release-0.9.10/) | Release que introdujo el LIDAR semántico (`sensor.lidar.ray_cast_semantic`). |
| [CARLA 0.9.4 release notes](https://carla.org/2019/03/01/release-0.9.4/) | Histórico relevante para `synchronous_mode`. |

## 8. Foro de la comunidad CARLA

| Fuente | Qué aporta |
|--------|------------|
| [forum.carla.org — Plotting lidar data on pygame display (need clarification)](https://forum.carla.org/t/plotting-lidar-data-on-pygame-display-need-clarification/182) | Discusión sobre visualización LIDAR en pygame y convenciones angulares (en el momento de la consulta el sitio devolvió un error de conexión, pero el hilo es referenciado por la documentación oficial). |

## 9. Referencias auxiliares (matplotlib polar)

| Fuente | Qué aporta |
|--------|------------|
| [Matplotlib — Polar plot demo](https://matplotlib.org/stable/gallery/pie_and_polar_charts/polar_demo.html) | Documentación canónica de `set_theta_zero_location` y `set_theta_direction`. |
| [pythontic — Polar plots in Python](https://pythontic.com/visualization/charts/polar%20plot) | Recordatorio sobre coordenadas polares en matplotlib. |
| [GeeksforGeeks — Plotting polar curves in Python](https://www.geeksforgeeks.org/python/plotting-polar-curves-in-python/) | Misma idea, ejemplos prácticos. |
| [3D Math Primer — Polar Coordinate Systems](https://gamemath.com/book/polarspace.html) | Fundamento matemático de `atan2` y conversión cartesiano ↔ polar. |
