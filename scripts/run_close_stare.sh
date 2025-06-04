#!/bin/bash

export OPENAI_API_KEY=

LOG_DIR="/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date '+%Y%m%d_%H%M%S').log"

python /mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/generate_response.py \
 --dataset_name '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare.jsonl' \
 --split 'test' \
 --subject '2D_text_instruct_VSim' \
 --strategy 'Direct' \
 --config_path '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/configs/gpt.yaml' \
 --model 'chatgpt-4o-latest' \
 --api_key $OPENAI_API_KEY \
 --output_path '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare-bench/results/stare_gpt4o_direct.json' \
 --max_tokens 1024 \
 --temperature 0 \
 --save_every 10 \
 > "$LOG_FILE" 2>&1

echo "✅ 日志保存在: $LOG_FILE"


































