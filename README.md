<p align="center">
  <img src="./images/stare_name.png" alt="STARE" width="300" />
</p>
<h1 align="center">
  Evaluating Multimodal Models on Visual Simulations
</h1>


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="https://stare-bench.github.io/" style="text-decoration: none; font-weight: bold;">[📖 ArXiv Paper]</a> •
    <a href="https://huggingface.co/datasets/kuvvi/STARE" style="text-decoration: none; font-weight: bold;"> [🤗 Data]</a> 
  </p>
</div>

<p align="center" width="80%">
  <img src="./images/overview.png" width="80%" height="70%">
</p>
<p align="center" style="font-size: 14px; color: gray;">
  <em>An overview of our <b>STARE</b>. </em>
</p>

## 💥 News 
- **[2025.6.6]** Our paper is now accessible at https://arxiv.org/abs/xxxx.

## 😳 STARE: Unfolding Spatial Cognition
<p align="center">
    <img src="images/vissim.png" width="90%"> <br>
  <b>Visual simulation of a cube net folding task reveals the challenges of spatial reasoning.</b> 
</p>


You can download both two datasets by the following command (Taking downloading math data as an example):

```python
from datasets import load_dataset

dataset = load_dataset("kuvvi/STARE", "folding_nets", split="test")
```



### Data Format

The dataset is provided in jsonl format and contains the following attributes:

```
{
    "pid": [string] Problem ID, e.g., “2d_va_vsim_001”,
    "question": [string] The question text,
    "answer": [string] The correct answer for the problem,
    "images": [list] , The images that problem needs.
    "other_info": [string] Additional information about this question，
    "category": [string] The category of the problem, e.g., “2D_text_instruction”,
}
```

## Requirements
```bash
git clone https://github.com/STARE-bench/STARE.git
cd STARE
git install -e .
```

## 📈 Evaluation

### Responses Generation
Our repository supports the evaluation of open source models such as Qwen2-VL, InternVL, LLaVA, and closed source models such as GPT, Gemini, Claude, etc. 
You can generate responses of these models by using the following commands:

Open-source Model:
```
 python generate_response.py \
 --dataset_name 'kuvvi/STARE' \
 --split 'test' \
 --category '2D_text_instruct_VSim' \
 --strategy 'CoT' \
 --config_path 'configs/gpt.yaml' \
 --model_path 'path_to_your_local_model' \
 --output_path 'path_to_output_json_file' \
 --max_tokens 4096 \
 --temperature 0.7 \
 --save_every 20
```

Close-source Model:

```
 python generate_response.py \
 --dataset_name 'kuvvi/STARE' \
 --split 'test' \
 --category '2D_text_instruct_VSim' \
 --config_path 'configs/gpt.yaml' \
 --model 'remote-model-name' \
 --api_key '' \
 --output_path 'path_to_output_file_name.json' \
 --max_tokens 4096 \
 --temperature 0 \
 --save_every 20
```

### Score Calculation

Finally, execute `python evaluation/calculate_acc.py` to calculate the final score based on the evaluation results. 
This step will compute overall accuracy as well as accuracy for each subject, category, and tasks.

## 📝Citation

If you find our benchmark useful in your research, please consider citing this BibTex:

```
```