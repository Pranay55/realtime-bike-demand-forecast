from kedro.pipeline import Pipeline, node
from .nodes import add_count, drop_unnecessary_columns, get_feature, load_data, rename_columns

def create_feature_eng_pipeline()->Pipeline:
    return Pipeline(
        [
            node(
                func = rename_columns,
                inputs=["input_data","params:feature_engineering.rename_columns"],
                outputs="renamed_data",
            ),
            node(
                func = get_feature,
                inputs=["renamed_data","params:feature_engineering.lag_params","params:feature_engineering.rolling_params"],
                outputs=["features","timestamps"],
            ),
            node(
                func = drop_unnecessary_columns,
                inputs=["features","params:feature_engineering.drop_columns"],
                outputs="final_features",
            )
        ]
    )


def load_training_data()->Pipeline:
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="train_data",
                outputs=["input_data", "last_timestamp"]
            )
        ]
    )

def load_inference_data()->Pipeline:
    return Pipeline(
        [
             node(
                func=add_count,
                inputs="inference_data",
                outputs="raw_data_with_count"
            ),
            node(
                func=load_data,
                inputs="raw_data_with_count",
                outputs=["input_data", "last_timestamp"]
            )
        ]
    )


def feat_eng_pipeline_training()->Pipeline:
    return load_training_data() + create_feature_eng_pipeline()

def feat_eng_pipeline_inference()->Pipeline:
    return load_inference_data() + create_feature_eng_pipeline()