import subprocess
from pathlib import Path

src_path  = Path.cwd() / "src"

mode = "train" # ["train", "eval"]

logger      = "mlflow"
experiment  = "mnist_realnvp"
max_epochs  = 1
data = "mnist_ad"
debug = 'fdr'

data_perc = 0.05

ckpt_path = None

if mode == "train":
    subprocess.run(["python", "train.py", 
                    f"experiment={experiment}",
                    f"trainer.max_epochs={max_epochs}",
                    f"+trainer.limit_train_batches={data_perc}",
                    # f"debug={debug}",
                    f"logger={logger}",
                    ],
                    cwd=src_path)
elif mode == "eval":
    subprocess.run(["python", "eval.py",
                    f"data={data}",
                    f"logger={logger}", 
                    f"ckpt_path={ckpt_path}",
                    ],
                    cwd=src_path)