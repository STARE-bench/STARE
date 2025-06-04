#!/bin/bash
 python generate_response.py \
 --dataset_name '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare.jsonl' \
 --split 'test' \
 --category '2D_text_instruct_VSim' \
 --strategy 'CoT' \
 --config_path 'configs/gpt.yaml' \
 --model_path 'path_to_your_local_model' \
 --output_path 'path_to_output_json_file' \
 --max_tokens 4096 \
 --temperature 0.7 \
 --save_every 20

































