import json
import yaml
import sys
import os

# === 路径配置 ===
CONFIG_PATH = "/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/configs/gpt.yaml"
JSONL_PATH = "/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare.jsonl"

# === 动态引入 build_query 函数 ===
sys.path.append("/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench")
from data_utils import build_query

# === 加载 YAML 配置 ===
def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# === 读取 JSONL 第一条数据 ===
def load_first_jsonl_sample(path):
    with open(path, 'r') as f:
        first_line = f.readline()
        return json.loads(first_line)

# === 主流程 ===
if __name__ == "__main__":
    config = load_yaml(CONFIG_PATH)
    sample = load_first_jsonl_sample(JSONL_PATH)

    # 构建 query（使用 Directly 策略）
    result = build_query(sample, config, strategy='Directly')

    # 打印结果
    print("\n==== GENERATED QUERY ====\n")
    print(result['query'])

    print("\n==== GROUND TRUTH ANSWER ====\n")
    print(result['gt_content'])
