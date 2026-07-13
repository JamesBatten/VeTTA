"""Shared typed settings primitives for ``vetta``."""

from .base import VettaSettings

_LAZY_EXPORTS = {
    "ChunkTreeBufferSettings": (".data", "ChunkTreeBufferSettings"),
    "InferVTAllSettings": (".inference", "InferVTAllSettings"),
    "InferencePipelineConfig": (".inference", "InferencePipelineConfig"),
    "PipelineBenchmarkConfig": (".integration", "PipelineBenchmarkConfig"),
    "PipelineValidation": (".integration", "PipelineValidation"),
    "TrainingBenchmarkConfig": (".integration", "TrainingBenchmarkConfig"),
    "TrainingValidation": (".integration", "TrainingValidation"),
    "VTInferNodeSettings": (".inference", "VTInferNodeSettings"),
    "VTInferTreeSettings": (".inference", "VTInferTreeSettings"),
    "TreeDatasetSettings": (".data", "TreeDatasetSettings"),
    "TreePipelineServerSettings": (".data", "TreePipelineServerSettings"),
    "TreePipelineSettings": (".data", "TreePipelineSettings"),
    "TreeRotateSettings": (".data", "TreeRotateSettings"),
    "TreeTranslateSettings": (".data", "TreeTranslateSettings"),
    "TreeZoomSettings": (".data", "TreeZoomSettings"),
    "TrainerCostFactorSettings": (".training", "TrainerCostFactorSettings"),
    "TrainerOptimizerSettings": (".training", "TrainerOptimizerSettings"),
    "TrainerSchedulerSettings": (".training", "TrainerSchedulerSettings"),
    "TrainerVesselTreeAutoencoderSettings": (".training", "TrainerVesselTreeAutoencoderSettings"),
    "VesselTreeAutoencoderConfig": (".model", "VesselTreeAutoencoderConfig"),
    "VesselTreeAutoencoderSettings": (".model", "VesselTreeAutoencoderSettings"),
    "default_trainer_config": (".training", "default_trainer_config"),
    "normalize_checkpoint_config": (".inference", "normalize_checkpoint_config"),
    "parse_checkpoint_model_settings": (".inference", "parse_checkpoint_model_settings"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "ChunkTreeBufferSettings",
    "InferVTAllSettings",
    "InferencePipelineConfig",
    "PipelineBenchmarkConfig",
    "PipelineValidation",
    "TrainingBenchmarkConfig",
    "TrainingValidation",
    "VTInferNodeSettings",
    "VTInferTreeSettings",
    "TreeDatasetSettings",
    "TreePipelineServerSettings",
    "TreePipelineSettings",
    "TreeRotateSettings",
    "TreeTranslateSettings",
    "TreeZoomSettings",
    "TrainerCostFactorSettings",
    "TrainerOptimizerSettings",
    "TrainerSchedulerSettings",
    "TrainerVesselTreeAutoencoderSettings",
    "VettaSettings",
    "VesselTreeAutoencoderConfig",
    "VesselTreeAutoencoderSettings",
    "default_trainer_config",
    "normalize_checkpoint_config",
    "parse_checkpoint_model_settings",
]
