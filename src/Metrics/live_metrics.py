import math
import sqlite3
import time
from pathlib import Path


EPISODE_AXIS = "episode"
UPDATE_AXIS = "update"
_VALID_AXES = {EPISODE_AXIS, UPDATE_AXIS}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LiveMetricsLogger:
    """Persist RL metrics in SQLite for live dashboard consumption."""

    def __init__(
        self,
        db_path,
        run_name,
        model_name,
        shield_type,
        map_name,
        max_episodes,
        max_steps,
        update_timestep,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        self._create_schema()
        self._upsert_run_status(
            run_name=run_name,
            model_name=model_name,
            shield_type=shield_type,
            map_name=map_name,
            max_episodes=max_episodes,
            max_steps=max_steps,
            update_timestep=update_timestep,
            status="running",
            last_episode=0,
            last_update_step=0,
        )
        self.run_name = run_name

    def _create_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_status (
                run_name TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                shield_type TEXT NOT NULL,
                map_name TEXT NOT NULL,
                status TEXT NOT NULL,
                max_episodes INTEGER NOT NULL,
                max_steps INTEGER NOT NULL,
                update_timestep INTEGER NOT NULL,
                last_episode INTEGER NOT NULL DEFAULT 0,
                last_update_step INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metric_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                axis TEXT NOT NULL,
                step INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL,
                event_time REAL NOT NULL,
                FOREIGN KEY(run_name) REFERENCES run_status(run_name)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metric_events_run_axis_step
            ON metric_events (run_name, axis, step)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metric_events_run_axis_name
            ON metric_events (run_name, axis, metric_name)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                episode INTEGER NOT NULL,
                applied_permissiveness REAL,
                previous_permissiveness REAL,
                changed INTEGER,
                llm_ok INTEGER,
                llm_target REAL,
                llm_action TEXT,
                llm_reasoning TEXT,
                governor_verdict TEXT,
                governor_reason TEXT,
                crash_rate REAL,
                offroad_rate REAL,
                shielded_fraction REAL,
                kpi_json TEXT,
                raw_response TEXT,
                event_time REAL NOT NULL,
                FOREIGN KEY(run_name) REFERENCES run_status(run_name)
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def _safe_value(value):
        if value is None:
            return None
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(casted) or math.isinf(casted):
            return None
        return casted

    def _upsert_run_status(
        self,
        run_name,
        model_name,
        shield_type,
        map_name,
        max_episodes,
        max_steps,
        update_timestep,
        status,
        last_episode,
        last_update_step,
    ):
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO run_status (
                run_name,
                model_name,
                shield_type,
                map_name,
                status,
                max_episodes,
                max_steps,
                update_timestep,
                last_episode,
                last_update_step,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_name) DO UPDATE SET
                status=excluded.status,
                last_episode=excluded.last_episode,
                last_update_step=excluded.last_update_step,
                updated_at=excluded.updated_at
            """,
            (
                run_name,
                model_name,
                shield_type,
                map_name,
                status,
                int(max_episodes),
                int(max_steps),
                int(update_timestep),
                int(last_episode),
                int(last_update_step),
                now,
                now,
            ),
        )

    def log_metrics(self, axis, step, metrics):
        if axis not in _VALID_AXES:
            raise ValueError(f"Invalid axis '{axis}'. Expected one of {_VALID_AXES}.")
        if not metrics:
            return

        event_time = time.time()
        rows = [
            (
                self.run_name,
                axis,
                int(step),
                metric_name,
                self._safe_value(metric_value),
                event_time,
            )
            for metric_name, metric_value in metrics.items()
        ]

        self.conn.executemany(
            """
            INSERT INTO metric_events (run_name, axis, step, metric_name, value, event_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        if axis == EPISODE_AXIS:
            self.conn.execute(
                """
                UPDATE run_status
                SET last_episode=?, updated_at=?
                WHERE run_name=?
                """,
                (int(step), event_time, self.run_name),
            )
        else:
            self.conn.execute(
                """
                UPDATE run_status
                SET last_update_step=?, updated_at=?
                WHERE run_name=?
                """,
                (int(step), event_time, self.run_name),
            )

        self.conn.commit()

    def log_meta_decision(self, record: dict):
        """Persiste una decision del meta-controlador LLM (tabla meta_decisions).

        ``record`` es el dict de ``MetaDecision.audit_record()``
        (src/meta/meta_controller.py).
        """
        self.conn.execute(
            """
            INSERT INTO meta_decisions (
                run_name, episode, applied_permissiveness, previous_permissiveness,
                changed, llm_ok, llm_target, llm_action, llm_reasoning,
                governor_verdict, governor_reason, crash_rate, offroad_rate,
                shielded_fraction, kpi_json, raw_response, event_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_name,
                int(record.get("episode", 0)),
                self._safe_value(record.get("applied_permissiveness")),
                self._safe_value(record.get("previous_permissiveness")),
                int(bool(record.get("changed", 0))),
                int(bool(record.get("llm_ok", 0))),
                self._safe_value(record.get("llm_target")),
                str(record.get("llm_action", "")),
                str(record.get("llm_reasoning", "")),
                str(record.get("governor_verdict", "")),
                str(record.get("governor_reason", "")),
                self._safe_value(record.get("crash_rate")),
                self._safe_value(record.get("offroad_rate")),
                self._safe_value(record.get("shielded_fraction")),
                str(record.get("kpi_json", "")),
                str(record.get("raw_response", "")),
                float(record.get("ts", time.time())),
            ),
        )
        self.conn.commit()

    def set_status(self, status):
        self.conn.execute(
            """
            UPDATE run_status
            SET status=?, updated_at=?
            WHERE run_name=?
            """,
            (status, time.time(), self.run_name),
        )
        self.conn.commit()

    def close(self, status=None):
        if status is not None:
            self.set_status(status)
        self.conn.close()


def list_live_metric_dbs(runs_root=None):
    runs_dir = Path(runs_root) if runs_root is not None else PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return []

    dbs = []
    for db_path in sorted(runs_dir.glob("*/metrics.sqlite")):
        run_name = db_path.parent.name
        dbs.append((run_name, str(db_path)))
    return dbs


def get_run_metadata(db_path, run_name):
    """Returns a metadata dict for a run from the run_status table.

    Keys: status, last_episode, max_episodes, model_name, shield_type,
          map_name, update_timestep, updated_at.
    Returns {} if the DB cannot be read or the run is not found.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        row = conn.execute(
            """
            SELECT status, last_episode, max_episodes, model_name,
                   shield_type, map_name, update_timestep, updated_at
            FROM run_status
            WHERE run_name = ?
            """,
            (run_name,),
        ).fetchone()
        conn.close()
        if row:
            return {
                "status": row[0],
                "last_episode": int(row[1]),
                "max_episodes": int(row[2]),
                "model_name": row[3],
                "shield_type": row[4],
                "map_name": row[5],
                "update_timestep": int(row[6]),
                "updated_at": float(row[7]),
            }
    except Exception:
        pass
    return {}


def load_axis_frame(db_path, run_name, axis):
    import pandas as pd

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        events_df = pd.read_sql_query(
            """
            SELECT step AS Step, metric_name, value
            FROM metric_events
            WHERE run_name = ? AND axis = ?
            ORDER BY step ASC, id ASC
            """,
            conn,
            params=[run_name, axis],
        )
    finally:
        conn.close()

    if events_df.empty:
        return pd.DataFrame(columns=["Step"])

    pivot = events_df.pivot_table(
        index="Step",
        columns="metric_name",
        values="value",
        aggfunc="last",
    ).reset_index()

    pivot.columns.name = None
    return pivot.sort_values("Step").reset_index(drop=True)


def load_datasets_from_sqlite(db_path, run_name):
    episode_df = load_axis_frame(db_path, run_name, EPISODE_AXIS)
    update_df = load_axis_frame(db_path, run_name, UPDATE_AXIS)

    datasets = {}
    if not episode_df.empty:
        datasets[EPISODE_AXIS] = episode_df
    if not update_df.empty:
        datasets[UPDATE_AXIS] = update_df

    # Full dataset: use episode as the primary timeline (complete and aligned)
    # Update metrics are available separately by their own step axis
    if not episode_df.empty:
        datasets["full"] = episode_df.copy()
    elif not update_df.empty:
        datasets["full"] = update_df.copy()

    return datasets


def load_meta_decisions(db_path, run_name):
    """Devuelve un DataFrame con las decisiones del meta-controlador LLM.

    Vacio si la tabla no existe (runs sin --meta_tuner) o no hay filas.
    """
    import pandas as pd

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        df = pd.read_sql_query(
            """
            SELECT episode, applied_permissiveness, previous_permissiveness,
                   changed, llm_ok, llm_target, llm_action, llm_reasoning,
                   governor_verdict, governor_reason, crash_rate, offroad_rate,
                   shielded_fraction, event_time
            FROM meta_decisions
            WHERE run_name = ?
            ORDER BY episode ASC, id ASC
            """,
            conn,
            params=[run_name],
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df
