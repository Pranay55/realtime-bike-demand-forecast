from pathlib import Path
import joblib
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error 


def rename_columns(df:pd.DataFrame,renaming_dict:Dict[str,str])->pd.DataFrame:
    print("Renaming columns...")
    print(df.rename(columns = renaming_dict).head())
    return df.rename(columns = renaming_dict)

def get_feature(df:pd.DataFrame, lag_params: Dict[str,List[int]], rolling_params: Dict[str,List[int]])->Tuple[pd.DataFrame, pd.Series]:
    
    timestamps = pd.to_datetime(df['datetime'])
    df['hour'] = timestamps.dt.hour 
    df['weekday'] = timestamps.dt.weekday

    for feature, lags in lag_params.items():
        for lag in lags:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag).bfill()

    for stat, windows in rolling_params.items():
        for window in windows:
            if stat == "mean":
                df[f'rolling_mean_{window}'] = (df['bike_count'].shift(1).rolling(window).mean())
            
            elif stat == "std":
                df[f'rolling_std_{window}'] = (df['bike_count'].shift(1).rolling(window).std())

    
    df['diff_1'] = df['bike_count_lag_1'] - df['bike_count_lag_2']
    df['diff_2'] = df['bike_count_lag_2'] - df['bike_count_lag_3']
    df['momentum'] = df['bike_count_lag_1'] / (df['rolling_mean_3'] + 1e-5)
    df['acceleration'] = df['diff_1'] - df['diff_2']
    df['daily_pattern_diff'] = df['bike_count_lag_1'] - df['bike_count_lag_24']

    df['hour_workingday'] = df['hour'] * df['workingday']
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_weekend'] = df['hour'] * df['is_weekend']
    df['day_type'] = (df['holiday'] * 2 + df['workingday'])

    print(df.columns)
    print(timestamps)

    return df, timestamps

def make_target(df:pd.DataFrame, target_params: Dict[str,Any])->pd.DataFrame:
    df[target_params['new_target_name']] = (
        df[target_params['target_column']].shift(-target_params['shift_periods']).ffill()
    )
    return df

def drop_count(df:pd.DataFrame)->pd.DataFrame:
    return df.drop(columns=['bike_count'])

def drop_unnecessary_columns(df:pd.DataFrame, drop_params:List[str])->pd.DataFrame:
    print(df.drop(columns=drop_params).columns)
    print(df.drop(columns=drop_params).head())
    return df.drop(columns=drop_params)


def split_data(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame,pd.Series,pd.Series]:
    target_name = params['target_params']['new_target_name']
    features = [col for col in df.columns if col != target_name]
    x,y = df[features], df[target_name]
    train_size = int(params["train_fraction"] * len(df))
    x_train, x_test = x[:train_size],x[train_size:]
    y_train, y_test = y[:train_size],y[train_size:]
    print(f"Training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, {y_test.shape}")
    return x_train, x_test, y_train, y_test


def train_model(x_train:pd.DataFrame, y_train:pd.Series, params:Dict[str,Any])->Any:
    model_type = params['model_type'].lower().strip()
    model_params = params['model_params'][model_type]

    if model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**model_params)
    elif model_type =='lightgbm':
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**model_params)
    elif model_type == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(x_train, y_train)
    return model


def predict(model:Any, x_test:pd.DataFrame)->pd.DataFrame:
    y_pred = pd.DataFrame(model.predict(x_test), columns=['predicted_bike_count'])
    print(f"shape: {x_test.shape, y_pred.shape}")
    return y_pred


def computeMetrics(y_true: Union[np.ndarray,list], y_pred: Union[np.ndarray,list]) -> Dict[str,float]:
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = float(mean_absolute_error(y_true,y_pred))
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true + 1e-8))*100

    metrics = {
        "RMSE": float(round(rmse,2)),
        "MAE": float(round(mae,2)),
        "MAPE": float(round(mape,2))        
    }
    print(f"Evaluation Metrics: {metrics}")
    return metrics


def save_model(model:Any, model_type:str, model_storage:Dict[str,Any])->None:
    model_dir = Path(model_storage['path'])
    model_name = model_storage['name']
    model_type = model_type.lower().strip()

    if model_type == 'catboost':
        model.save_model(str(model_dir / f"{model_name}.cbm"))
    elif model_type == 'lightgbm':
        joblib.dump(model, model_dir / f"{model_name}.txt")
    elif model_type == 'xgboost':
        joblib.dump(model, model_dir / f"{model_name}.json")

    return None


def load_model(model_type:str, model_storage:Dict[str,Any])->Any:
    model_dir = Path(model_storage['path'])
    model_name = model_storage['name']
    model_type = model_type.lower().strip()

    if model_type in ['catboost', 'cb']:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(model_dir / f"{model_name}.cbm"))
    elif model_type in ['lightgbm', 'lgb']:
        model = joblib.load(model_dir / f"{model_name}.txt")
    elif model_type in ['xgboost', 'xgb']:
        model = joblib.load(model_dir / f"{model_name}.json")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model



def load_data(df:pd.DataFrame)->Tuple[pd.DataFrame,pd.Timestamp]:
    last_timestamp = pd.to_datetime(df['datetime']).iloc[-1]
    return df, last_timestamp

def add_count(df:pd.DataFrame)->pd.DataFrame:
    df['bike_count'] = df['registered'] + df['casual']
    return df

def join_timestamps(df:pd.DataFrame,timestamps:pd.DataFrame)->pd.DataFrame:
    df['datetime'] = timestamps
    print(df.tail(5))
    return df