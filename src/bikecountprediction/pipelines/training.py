from kedro.pipeline import Pipeline, node
from .nodes import drop_count, make_target, predict, save_model, split_data, train_model, computeMetrics

def create_training_pipeline()->Pipeline:
    return Pipeline(
        [
            node(
                func = make_target,
                inputs=["final_features","params:training.target_params"],
                outputs="data_with_target",
            ),
            node(
                func=drop_count,
                inputs="data_with_target",
                outputs="data_wo_count"
            ),
            node(
                func= split_data,
                inputs=["data_wo_count","params:training"],
                outputs=["x_train","x_test","y_train","y_test"],
            ),
            node(
                func = train_model,
                inputs=["x_train","y_train","params:training"],
                outputs="trained_model"
            ),
            node(
                func=predict,
                inputs=["trained_model","x_test"],
                outputs="predictions"
            ),
            node(
                func=computeMetrics,
                inputs=["y_test","predictions"],
                outputs="evaluation_metrics"
            ),
            node(
                func = save_model,
                inputs=["trained_model","params:training.model_type","params:model_storage"],
                outputs=None
            )
        ]
    )