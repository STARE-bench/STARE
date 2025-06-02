
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

## 📈 Evaluation

### Responses Generation
Our repository supports the evaluation of open source models such as Qwen2-VL, InternVL, LLaVA, and closed source models such as GPT, Gemini, Claude, etc. 
You can generate responses of these models by using the following commands:
