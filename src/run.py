import subprocess
from pathlib import Path

src_path  = Path.cwd() / "src"

mode = "train" # ["train", "eval"]

logger      = "mlflow"
experiment  = "mnist_realnvp"
max_epochs  = 2
data = "mnist_ad"
debug = 'fdr'

ckpt_path = None

if mode == "train":
    subprocess.run(["python", "train.py", 
                    f"experiment={experiment}",
                    f"trainer.max_epochs={max_epochs}",
                    f"debug={debug}",
                    f"logger={logger}"], 
                    cwd=src_path)
elif mode == "eval":
    subprocess.run(["python", "eval.py",
                    f"data={data}",
                    f"logger={logger}", 
                    f"ckpt_path={ckpt_path}"],
                    cwd=src_path)