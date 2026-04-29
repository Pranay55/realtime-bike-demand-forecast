from kedro.pipeline import Pipeline
from typing import Dict
from .pipelines.feature_eng import feat_eng_pipeline_training, feat_eng_pipeline_inference
from .pipelines.training import create_training_pipeline
from .pipelines.inference import create_inference_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    feature_eng_pipeline_training = feat_eng_pipeline_training()
    feature_eng_pipeline_inference = feat_eng_pipeline_inference()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()

    return {
        "__default__": feature_eng_pipeline_training + training_pipeline,
        "feature_eng": feature_eng_pipeline_training,
        "training": feature_eng_pipeline_training + training_pipeline,
        "inference": feature_eng_pipeline_inference + inference_pipeline
    }