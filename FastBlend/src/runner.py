from typing import Callable, List, Optional

from .config import FastBlendConfig
from .engine import FastBlendEngine, FastBlendInput_Sequence


class FastBlendRunner:
    """Holds a :class:`FastBlendConfig` for scripts; prefer :class:`FastBlendEngine` for new code."""

    def __init__(self, config: Optional[FastBlendConfig] = None) -> None:
        self.config = config or FastBlendConfig()
        self._engine = FastBlendEngine()

    def run(
        self,
        frames_guide: List,
        frames_style: List,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        backend: str = "auto",
    ) -> List:
        return self._engine.smooth_sequence(
            FastBlendInput_Sequence(frames_guide, frames_style),
            self.config,
            progress_callback,
            backend,
        )
