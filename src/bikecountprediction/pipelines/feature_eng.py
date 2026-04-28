from kedro.pipeline import Pipeline, node
from .nodes import drop_unnecessary_columns, get_feature, rename_columns

def create_feature_eng_pipeline()->Pipeline:
    return Pipeline(
        [
            node(
                func = rename_columns,
                inputs=["train_data","params:feature_engineering.rename_columns"],
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