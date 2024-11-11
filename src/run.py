import subprocess
from pathlib import Path

src_path  = Path.cwd() / "src"

mode = "train" # ["train", "eval"]

model = "ddpm"
logger      = "mlflow"
experiment  = "mnist_realnvp"
max_epochs  = 1
data = "mnist_ad"
debug = 'fdr'

train_size = 0.05
test_size = 0.2

ckpt_path = None

if mode == "train":
    subprocess.run(["python", "train.py",
                    f"data={data}",
                    f"model={model}", 
                    # f"experiment={experiment}",
                    # f"trainer.max_epochs={max_epochs}",
                    # f"+trainer.limit_train_batches={train_size}",
                    # f"+trainer.limit_test_batches={test_size}",
                    f"debug={debug}",
                    # f"logger={logger}",
                    ],
                    cwd=src_path)
elif mode == "eval":
    subprocess.run(["python", "eval.py",
                    f"data={data}",
                    f"logger={logger}", 
                    f"ckpt_path={ckpt_path}",
                    ],
                    cwd=src_path)