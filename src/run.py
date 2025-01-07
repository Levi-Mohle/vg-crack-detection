import subprocess
from pathlib import Path

src_path  = Path.cwd()  / "src"

mode = "train" # ["train", "eval"]

model = "autoencoders/vq-f4"
logger      = "mlflow"
experiment  = "autoencoders/impasto_vq-f4"
max_epochs  = 1
data = "impasto"
debug = 'fdr'
device = "cpu" 
batch_size = 8
num_workers = 15

train_size = .05
val_size = .1
test_size = .05

ckpt_path = r"C:\Users\lmohle\Documents\2_Coding\lightning-hydra-template\logs\train\runs\2024-12-30_11-23-10_DDPM_2ch\checkpoints\last.ckpt"

if mode == "train":
    subprocess.run(["python", "train.py",
                    # f"data={data}",
                    # f"model={model}",
                    f"trainer={device}",
                    f"experiment={experiment}",
                    f"trainer.max_epochs={max_epochs}",
                    f"+trainer.limit_val_batches={val_size}",
                    f"+trainer.limit_train_batches={train_size}",
                    f"+trainer.limit_test_batches={test_size}",
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