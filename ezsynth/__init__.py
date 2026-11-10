"""Ezsynth: patch-based image and video stylization (EbSynth remake on PyTorch).

Public API (resolved lazily, so ``import ezsynth`` stays cheap)::

    from ezsynth import ImageSynth, RunConfig, load_guide

    synth = ImageSynth("style.png", RunConfig(backend="torch"))
    image, error = synth.run([load_guide("src_guide.png", "tgt_guide.png", 1.0)])
"""

from importlib import import_module

_EXPORTS = {
    "ImageSynth": "ezsynth.api",
    "RunConfig": "ezsynth.api",
    "Ezsynth": "ezsynth.api",
    "load_guide": "ezsynth.guide",
    "GuideObject": "ezsynth.guide",
    "Project": "ezsynth.project",
    "MainConfig": "ezsynth.config",
    "ProjectConfig": "ezsynth.config",
    "PrecomputationConfig": "ezsynth.config",
    "PipelineConfig": "ezsynth.config",
    "BlendingConfig": "ezsynth.config",
    "EbsynthParamsConfig": "ezsynth.config",
    "DebugConfig": "ezsynth.config",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
