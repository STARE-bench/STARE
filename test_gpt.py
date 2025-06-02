import json
import yaml
import os
import time
from PIL import Image
from openai import OpenAI
import sys

# === 添加模块路径 ===
sys.path.append("/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench")

# === 导入组件 ===
from data_utils import build_query
from models.gpt import GPT_Model, create_message

# === 路径配置 ===
ROOT = "/mnt/petrelfs/gujiawei/stare_bench/release_stare"
JSONL_PATH = os.path.join(ROOT, "stare.jsonl")
CONFIG_PATH = os.path.join(ROOT, "stare-bench/configs/gpt.yaml")
IMAGE_ROOT = os.path.join(ROOT, "")

# === 加载 config.yaml ===
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# === 加载 stare.jsonl 的第一条样本 ===
def load_first_sample(jsonl_path):
    with open(jsonl_path, 'r') as f:
        line = f.readline()
        return json.loads(line)

# === 替换路径为 PIL.Image 对象 ===
def convert_image_paths_to_objects(sample, image_root):
    image_objs = []
    for img_path in sample['images']:
        full_path = os.path.join(image_root, img_path)
        try:
            image = Image.open(full_path).convert("RGB")
            image_objs.append(image)
        except Exception as e:
            print(f"❌ Failed to load image: {full_path}\nError: {e}")
            image_objs.append(None)
    sample['images'] = image_objs
    return sample

# === 主函数 ===
if __name__ == "__main__":
    # Step 1: Load config & raw sample
    config = load_yaml(CONFIG_PATH)
    raw_sample = load_first_sample(JSONL_PATH)
    print(f"✅ 加载第一条样本: {raw_sample['qid']}")
    print(f"\n📝 原始问题内容:\n{raw_sample['question']}")
    
    # Step 2: Load images as PIL.Image
    raw_sample = convert_image_paths_to_objects(raw_sample, IMAGE_ROOT)
    print(f"\n🖼️ 成功加载图像数: {len(raw_sample['images'])}")

    # Step 3: Build query with <image> placeholders
    query_sample = build_query(raw_sample, config, strategy="Directly")
    print(f"\n📦 构建后的 query:\n{query_sample['query']}")
    print(f"\n✅ 正确答案 (gt_content): {query_sample['gt_content']}")

    # Step 4: 构造 messages（可选打印）
    messages = create_message(query_sample)
    print("\n📨 构造的 messages（简略展示）:")
    for m in messages[0]['content']:
        if m['type'] == 'text':
            print("🔹 TEXT:", m['text'][:60].replace('\n', ' ') + '...')
        else:
            print("🖼️ IMAGE: [base64 image embedded]")

    # Step 5: 初始化 GPT 模型并调用
    openai_client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY
    gpt_model = GPT_Model(client=openai_client)

    print("\n🚀 调用 GPT 模型中...")
    start_time = time.time()
    response = gpt_model.get_response(query_sample)
    end_time = time.time()

    # Step 6: 打印结果
    print("\n=== 🧠 GPT 预测结果 ===\n")
    print(response)

    print("\n=== ✅ 正确答案 (gt_content) ===\n")
    print(query_sample['gt_content'])

    print(f"\n⏱️ 推理耗时: {end_time - start_time:.2f} 秒")
