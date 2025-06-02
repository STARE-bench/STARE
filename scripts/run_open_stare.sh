#!/bin/bash

# ========== ✅ CUDA 环境 ==========
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ========== ✅ 显卡选择（调试用，可选）==========
export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

# ========== ✅ 日志准备 ==========
LOG_DIR="/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen_run_$(date '+%Y%m%d_%H%M%S').log"
echo "日志将保存到: $LOG_FILE"

# ========== ✅ 推理命令启动 ==========
OMP_NUM_THREADS=8 srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=QwenInfer \
 -c 16 \
 -w SH-IDCA1404-10-140-54-106 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
nohup python /mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/generate_response.py \
 --dataset_name '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare.jsonl' \
 --split 'test' \
 --subject '2D_text_instruct_VSim' \
 --strategy 'Direct' \
 --config_path '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/configs/gpt.yaml' \
 --model_path '/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2-VL-2B-Instruct' \
 --output_path '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/results/stare_qwen_direct.jsonl' \
 --max_tokens 1024 \
 --temperature 0 \
 --save_every 10 \
 > "$LOG_FILE" 2>&1 &

echo "✅ Qwen 推理任务已启动！日志在: $LOG_FILE"
