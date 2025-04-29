"""
Script load csv file results from a classical computervision algorithm,
a heuristic method, and evaluate results according to the metrics

    Source Name : heuristic_crack_detection_result.py
    Contents    : Functions to load csv data and evaluate results
    Date        : 2025

 """
# %%
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.models.components.utils.evaluation import classify_metrics

# %% Load

# Define directories
data_dir        = r"C:\Users\lmohle\Documents\2_Coding\ml-crack-detection-van-gogh\data\impasto"
output_dir      = r"C:\Users\lmohle\Documents\2_Coding\data\output"

# Read input (only needed for true labels)
lightning_data = IMPASTO_DataModule(data_dir           = data_dir,
                                    batch_size         = 80,
                                    variant            = "512x512",
                                    crack              = "realBI"
                                    )

lightning_data.setup()
loader = lightning_data.test_dataloader()

# Read input
for i, (_, _, id) in enumerate(loader):
    break
y_true = id.numpy()

# Read output
file_name   = "christiaan_results.csv"
df_results  = pd.read_csv(os.path.join(output_dir, file_name))

# %% Get OOD scores from results .csv

# Using Sum(|Area * dHeight|) as OOD measure
df_results["ood"] = np.abs(df_results["Area_um2"] * df_results["HeightDifference_um"])
# df_results["ood"] = np.abs(df_results["Area_um2"])
# df_results["ood"] = np.abs(df_results["HeightDifference_um"])
df_summed = df_results.groupby("Number", as_index=False)["ood"].sum()

np_summed = df_summed.to_numpy()
np_summed[:,0] = np_summed[:,0] - 1

ood_scores = np.zeros((74))
for j, id in enumerate(np_summed[:,0]):
    ood_scores[int(id)] = np_summed[int(j),1]

# %% Apply evaluation metrics
classify_metrics(ood_scores, y_true)


# %%
