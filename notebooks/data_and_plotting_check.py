# %%
import sys
import torch
from pathlib import Path

# add main folder to working directory
wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from src.data.impasto_datamodule import IMPASTO_DataModule
from src.data.components.transforms import *
# %% Load the data
lightning_data = IMPASTO_DataModule(data_dir = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\data\impasto",
                                    batch_size = 32,
                                    rgb_transform = diffuser_to_grayscale(),
                                    height_transform = diffuser_normalize_height_idv()
                                    )
lightning_data.setup()
test_loader = lightning_data.test_dataloader()
# %% Run for 1 batch
avg_diff_gray   = torch.zeros(7)
avg_diff_height = torch.zeros(7)
for i, (gray, height, id) in enumerate(test_loader):
    avg_diff_gray[i]    = gray.max() - gray.min()
    avg_diff_height[i]  = height.max() - height.min()
print(f"mean diff gray: {torch.mean(avg_diff_gray):.2f}")
print(f"mean diff height: {torch.mean(avg_diff_height):.2f}")

# %%
for gray, height, id in test_loader:
    print(gray.min(), gray.max())


# %%
