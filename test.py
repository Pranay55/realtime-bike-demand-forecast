import pandas as pd

test = pd.read_csv("data/01_raw/test.csv")
pred = pd.read_csv("data/07_model_output/predictions_with_timestamps.csv")

print(test.shape, pred.shape)