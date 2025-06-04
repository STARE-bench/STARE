
# STARE: Evaluating Multimodal Models on Visual Simulations

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="https://stare-bench.github.io/" style="text-decoration: none; font-weight: bold;">🌻 Homepage</a> •
    <a href="https://huggingface.co/STARE-VisSim/" style="text-decoration: none; font-weight: bold;">🤗 Data</a> 
  </p>
</div>


<p align="center" width="80%">
  <img src="./images/stare_overview.png" width="80%" height="70%">
</p>
<p align="center" style="font-size: 14px; color: gray;">
  <em>An overview of our <b>STARE</b>. </em>
</p>

### Data Format

The dataset is provided in jsonl format and contains the following attributes:

```
{
    "pid": [string] Problem ID, e.g., “math_1”,
    "question": [string] The question text,
    "answer": [string] The correct answer for the problem,
    "images": [list] , The images that problem needs.
    "other_info": [string] Additional information about this question，
    "category": [string] The category of the problem, e.g., “2D_text_instruction”,
}
```

## Requirements
```bash
git clone https://github.com/Kuvvius/stare_open
cd stare_open
git install -e .
```

## 📈 Evaluation

### Responses Generation
Our repository supports the evaluation of open source models such as Qwen2-VL, InternVL, LLaVA, and closed source models such as GPT, Gemini, Claude, etc. 
You can generate responses of these models by using the following commands:
