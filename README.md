<p align="center">
  <img src="./images/stare_name.png" alt="STARE" width="300" />
</p>
<h1 align="center">
  Evaluating Multimodal Models on Visual Simulations
</h1>


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="https://arxiv.org/abs/2506.04633" style="text-decoration: none; font-weight: bold;">[🔗 ArXiv]</a> •
    <a href="https://arxiv.org/pdf/2506.04633" style="text-decoration: none; font-weight: bold;">[📑 Paper]</a> •
    <a href="https://huggingface.co/datasets/kuvvi/STARE" style="text-decoration: none; font-weight: bold;"> [🤗 Data]</a> 
  </p>
</div>

<p align="center" width="80%">
  <img src="./images/overview.png" width="80%" height="70%">
</p>
<p align="center" style="font-size: 14px; color: gray;">
  <em>An overview of our <b>STARE</b>. </em>
</p>



## 😳 STARE: Unfolding Spatial Cognition
STARE is structured to comprehensively cover spatial reasoning at multiple complexity levels, from basic geometric transformations (2D and 3D) to more integrated tasks (cube net folding and tangram puzzles) and real-world spatial reasoning scenarios (temporal frame and perspective reasoning). Each task is presented as a multiple-choice or yes/no question using carefully designed visual and textual prompts. In total, the dataset contains about 4K instances across different evaluation setups.
<p align="center">
    <img src="images/vissim.png" width="90%"> <br>
  <b>Visual simulation of a cube net folding task reveals the challenges of spatial reasoning.</b> 
</p>
Models exhibit significant variation in spatial reasoning performance across STARE tasks. Accuracy is highest on simple 2D transformations (up to 87.7%) but drops substantially for 3D tasks and multi-step reasoning (e.g., cube nets, tangrams), often nearing chance. Visual simulations generally improve performance, though inconsistently across models. The reasoning-optimized o1 model performs best overall with VisSim, yet still lags behind humans. Human participants consistently outperform models, confirming the complexity of STARE tasks.


## 📖 Dataset Usage

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
To set up the environment, it is highly recommended to use Python 3.10.

```bash
git clone https://github.com/STARE-bench/STARE.git
cd STARE
pip install -r requirements.txt
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
@misc{li2025unfoldingspatialcognitionevaluating,
      title={Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations}, 
      author={Linjie Li and Mahtab Bigverdi and Jiawei Gu and Zixian Ma and Yinuo Yang and Ziang Li and Yejin Choi and Ranjay Krishna},
      year={2025},
      eprint={2506.04633},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04633}, 
}
```
