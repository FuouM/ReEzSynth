from collections.abc import Callable

from .config import FastBlendConfig
from .engine import FastBlendEngine, FastBlendInput_Sequence


class FastBlendRunner:
    """Holds a :class:`FastBlendConfig` for scripts; prefer :class:`FastBlendEngine` for new code."""

    def __init__(self, config: FastBlendConfig | None = None) -> None:
        self.config = config or FastBlendConfig()
        self._engine = FastBlendEngine()

    def run(
        self,
        frames_guide: list,
        frames_style: list,
        progress_callback: Callable[[int, int], None] | None = None,
        backend: str = "auto",
    ) -> list:
        return self._engine.smooth_sequence(
            FastBlendInput_Sequence(frames_guide, frames_style),
            self.config,
            progress_callback,
            backend,
        )
