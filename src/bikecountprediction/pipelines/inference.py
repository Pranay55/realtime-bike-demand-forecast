from kedro.pipeline import Pipeline, node
from .nodes import computeMetrics, drop_count, join_timestamps, load_model, predict

def create_inference_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_model,
                inputs=["params:training.model_type", "params:model_storage"],
                outputs="model"
            ),
            node(
                func=drop_count,
                inputs="final_features",
                outputs="final_features_wo_count"
            ),
            node(
                func=predict,
                inputs=["model", "final_features_wo_count"],
                outputs="predictions"
            ),
            node(
                func=join_timestamps,
                inputs=["predictions", "timestamps"],
                outputs="predictions_with_timestamps"
            )
        ]
    )