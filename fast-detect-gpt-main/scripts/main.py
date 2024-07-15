import subprocess
from datetime import datetime

datasets = ["xsum", "squad", "writing"]
source_models = ["gpt2-xl", "opt-2.7b", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]
data_path = "../exp_maxlen150/data"

# 准备数据集
for D in datasets:
    for M in source_models:
        print(f"{datetime.now()}, Preparing dataset {D}_{M} ...")
        output_file = f"{data_path}/{D}_{M}"
        subprocess.run([
            "python", "data_builder.py",
            "--dataset", D,
            "--n_samples", "500",
            "--base_model_name", M,
            "--output_file", output_file
        ])