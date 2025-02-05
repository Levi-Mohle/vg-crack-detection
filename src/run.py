import subprocess
from pathlib import Path

src_path  = Path.cwd()  / "src"

mode = "train" # ["train", "eval"]

model = "flowmatching"
# model = "cae"
logger      = "mlflow"
experiment  = "impasto_flowmatching"
# experiment = "impasto_cae"
max_epochs  = 1
data = "impasto"
debug = 'fdr'
device = "cpu" 
batch_size = 32
variant = "32x32"

train_size = .01
val_size = .1
test_size = .1

ckpt_path = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\logs\train\runs\2025-01-08_17-04-12\checkpoints\last.ckpt"

if mode == "train":
    subprocess.run(["python", "train.py",
                    # f"data={data}",
                    # f"model={model}",
                    f"trainer={device}",
                    f"experiment={experiment}",
                    f"trainer.max_epochs={max_epochs}",
                    f'data.batch_size={batch_size}',
                    # f"+trainer.limit_val_batches={val_size}",
                    # f"+trainer.limit_train_batches={train_size}",
                    # f"+trainer.limit_test_batches={test_size}",
                    f"data.variant={variant}",
                    # f"debug={debug}",
                    # f"logger={logger}",
                    ],
                    cwd=src_path)
elif mode == "eval":
    subprocess.run(["python", "eval.py",
                    # f"data={data}",
                    # f"model={model}",
                    # f"logger={logger}", 
                    # f"data.num_workers={num_workers}",
                    f'data.batch_size={batch_size}',
                    f"ckpt_path={ckpt_path}",
                    f"+experiment={experiment}",
                    f"+trainer.limit_test_batches={test_size}"
                    ],
                    cwd=src_path)