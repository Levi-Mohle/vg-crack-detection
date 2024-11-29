import subprocess
from pathlib import Path

src_path  = Path.cwd() # / "src"

mode = "train" # ["train", "eval"]

model = "cae"
logger      = "mlflow"
experiment  = "impasto_cae"
max_epochs  = 1
data = "impasto"
debug = 'fdr'
device = "gpu" 

train_size = .1
test_size = .2

ckpt_path = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\logs\train\runs\2024-11-28_14-59-45_Good\checkpoints\last.ckpt"

if mode == "train":
    subprocess.run(["python", "train.py",
                    # f"data={data}",
                    # f"model={model}",
                    f"trainer={device}",
                    f"experiment={experiment}",
                    f"trainer.max_epochs={max_epochs}",
                    f"+trainer.limit_train_batches={train_size}",
                    f"+trainer.limit_test_batches={test_size}",
                    # f"debug={debug}",
                    # f"logger={logger}",
                    ],
                    cwd=src_path)
elif mode == "eval":
    subprocess.run(["python", "eval.py",
                    f"data={data}",
                    f"model={model}",
                    # f"logger={logger}", 
                    f"ckpt_path={ckpt_path}",
                    ],
                    cwd=src_path)