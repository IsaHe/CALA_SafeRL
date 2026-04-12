import numpy as np

class RunningMeanStd:
    """
    Normalización online de observaciones con media y varianza en ejecución.
 
    Implementa el algoritmo de Welford para varianza incremental estable.
    Se mantiene en CPU (numpy) para no añadir overhead en select_action.
 
    Uso:
        normalizer = RunningMeanStd(shape=(state_dim,))
        normalizer.update(obs_batch)          # actualiza estadísticas
        obs_norm = normalizer.normalize(obs)  # normaliza [-5, 5] aprox.
    """
 
    def __init__(self, shape: tuple, epsilon: float = 1e-8, clip: float = 5.0):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = epsilon
        self.epsilon = epsilon
        self.clip    = clip
 
    def update(self, x: np.ndarray):
        """Actualiza estadísticas con una observación o batch (shape: [..., dim])."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
 
        total_count  = self.count + batch_count
        delta        = batch_mean - self.mean
        new_mean     = self.mean + delta * batch_count / total_count
        m_a          = self.var * self.count
        m_b          = batch_var * batch_count
        new_var      = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
 
        self.mean  = new_mean
        self.var   = new_var
        self.count = total_count
 
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normaliza x a media≈0, std≈1, recortado a ±clip."""
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
 
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}
 
    def load_state_dict(self, d: dict):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]