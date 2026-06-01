class CurriculumManager:
    """
    Curriculum progresivo de 5 etapas con sliding-window metrics y rollback.

    Las etapas dividen uniformemente [0, max_npc] en 4 incrementos iguales:
        max_npc=20  →  [0, 5, 10, 15, 20]
        max_npc=40  →  [0, 10, 20, 30, 40]

    Avance: las métricas deben superar los umbrales una vez transcurridos
    `min_eps_per_stage` episodios en la etapa actual (la ventana deslizante
    tarda ese mismo tiempo en reflejar el rendimiento real de la etapa).

    Rollback: si el agente colapsa durante `rollback_patience` episodios
    consecutivos (acumulados con decaimiento), retrocede una etapa.
    Esto evita el deadlock del Problema 2 en el que el agente se queda
    atascado por encima del umbral de crash_rate sin poder nunca avanzar.
    """

    def __init__(
        self,
        max_npc: int,
        enabled: bool = True,
        min_eps_per_stage: int = 100,
        rollback_patience: int = 50,
    ):
        self.enabled = enabled
        self.max_npc = max_npc
        self.min_eps = min_eps_per_stage
        self.rollback_patience = rollback_patience

        self.stages = self._build_stages(max_npc)

        self._stage_idx: int = 0 if enabled else len(self.stages) - 1
        self._eps_at_stage: int = 0
        self._rollback_counter: int = 0

    @staticmethod
    def _build_stages(max_npc: int) -> list:
        """Divide [0, max_npc] en 4 saltos iguales (5 etapas total)."""
        if max_npc <= 0:
            return [0]
        n = max_npc
        raw = [0, n // 4, n // 2, (3 * n) // 4, n]
        seen = set()
        stages = []
        for s in raw:
            if s not in seen:
                seen.add(s)
                stages.append(s)
        return stages

    @property
    def current_npc_count(self) -> int:
        return self.stages[self._stage_idx]

    @property
    def current_stage_idx(self) -> int:
        return self._stage_idx

    def step(self, offroad_rate: float, crash_rate: float, avg_reward: float):
        """
        Evalúa el rendimiento actual y actualiza la etapa si procede.

        Debe llamarse UNA VEZ al FINAL de cada episodio, después de actualizar
        las ventanas deslizantes de métricas.

        Returns:
            (npc_count: int, event: str)
            event ∈ {"none", "advance", "rollback"}
        """
        if not self.enabled:
            return self.max_npc, "none"

        self._eps_at_stage += 1
        at_last_stage = self._stage_idx >= len(self.stages) - 1

        if self._stage_idx > 0 and self._eps_at_stage > self.min_eps:
            if self._is_collapsing(crash_rate, offroad_rate):
                self._rollback_counter += 1
            else:
                self._rollback_counter = max(0, self._rollback_counter - 1)

            if self._rollback_counter >= self.rollback_patience:
                self._stage_idx -= 1
                self._eps_at_stage = 0
                self._rollback_counter = 0
                return self.current_npc_count, "rollback"

        if not at_last_stage and self._eps_at_stage >= self.min_eps:
            if self._can_advance(crash_rate, offroad_rate, avg_reward):
                self._stage_idx += 1
                self._eps_at_stage = 0
                self._rollback_counter = 0
                return self.current_npc_count, "advance"

        return self.current_npc_count, "none"

    def _can_advance(
        self, crash_rate: float, offroad_rate: float, avg_reward: float
    ) -> bool:
        """
        Condiciones calibradas para avance gradual.

        Etapa 0 (0 NPCs):    dominar lane-keeping sin tráfico.
        Etapa 1 (25% NPCs):  sobrevivir con tráfico muy ligero.
        Etapa 2 (50% NPCs):  manejar tráfico moderado.
        Etapa 3 (75% NPCs):  gestionar tráfico denso antes de máxima dificultad.
        """
        idx = self._stage_idx
        if idx == 0:
            return offroad_rate < 0.20 and avg_reward > 0
        elif idx == 1:
            return crash_rate < 0.25 and offroad_rate < 0.15
        elif idx == 2:
            return crash_rate < 0.15
        elif idx == 3:
            return crash_rate < 0.10
        return crash_rate < 0.10

    def _is_collapsing(self, crash_rate: float, offroad_rate: float) -> bool:
        """
        El agente ha colapsado en la etapa actual y debe retroceder.
        Umbrales holgados para no hacer rollback por varianza normal.
        """
        idx = self._stage_idx
        if idx == 1:
            return crash_rate > 0.45 or offroad_rate > 0.40
        elif idx == 2:
            return crash_rate > 0.40
        elif idx == 3:
            return crash_rate > 0.35
        return crash_rate > 0.30