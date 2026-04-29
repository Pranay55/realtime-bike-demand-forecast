from pathlib import Path
from time import sleep

import pandas as pd

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import yaml

def run_inference()->None:
    project_path = Path(__file__).resolve().parent.parent
    bootstrap_project(project_path)
    configure_project("bikecountprediction")

    param_path = project_path / "conf" / "base" / "parameters.yml"
    with open(param_path, "r") as f:
        params = yaml.safe_load(f)

    catalog_path = project_path / "conf" / "base" / "catalog.yml"
    with open(catalog_path, "r") as f:
        catalog = yaml.safe_load(f)

    
    runner_config = params['pipeline_runner']

    predictions_path = project_path / catalog['predictions_with_timestamps']['filepath']
    if predictions_path.exists():
        predictions_path.unlink()
        print("Cleared previous predictions.")

    inference_data_path = project_path / catalog['inference_data']['filepath']
    data = pd.read_csv(inference_data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.reset_index(drop=True)

    batch_size = runner_config['batch_size']
    first_timestamp = pd.to_datetime(runner_config['first_timestamp'])
    last_timestamp = pd.to_datetime(runner_config['last_timestamp'])
    num_steps = runner_config['num_steps_inference']
    interval_seconds = runner_config['inference_interval_seconds']

    first_idx = data[data['datetime']>=first_timestamp ].index[0]

    print(f"Starting inference from {first_timestamp}, number of steps: {num_steps}, batch size: {batch_size}")

    for step in range(num_steps):
        current_idx = first_idx + step
        batch_start = max(0,current_idx - batch_size + 1)
        batch_end = current_idx + 1

        batch = data.iloc[batch_start:batch_end].copy()

        batch_path = project_path / catalog['inference_batch']['filepath']
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch.to_csv(batch_path, index=False)

        with KedroSession.create(project_path = project_path) as session:
            session.run(pipeline_name="inference")
            
        print(f"[{step+1}/{num_steps}] Prediction saved")

        if step < num_steps - 1:
            sleep(interval_seconds)

if __name__ == "__main__":
    run_inference()