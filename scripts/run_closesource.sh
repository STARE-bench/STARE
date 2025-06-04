#!/bin/bash
 python generate_response.py \
 --dataset_name '/mnt/petrelfs/gujiawei/stare_bench/release_stare/stare.jsonl' \
 --split 'test' \
 --category '2D_text_instruct_VSim' \
 --config_path 'configs/gpt.yaml' \
 --model 'remote-model-name' \
 --api_key '' \
 --output_path 'path_to_output_file_name.json' \
 --max_tokens 4096 \
 --temperature 0 \
 --save_every 20