import yaml
import json


def load_yaml(file_path):
    """
    Load a YAML file and return the parsed dictionary.
    """
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    """
    Check if the response string is valid (non-empty and no error flag).
    """
    if isinstance(response, str):
        response = response.strip()
    if not response or "Response Error" in response:
        return False
    return True


def build_query(sample, config, strategy):
    """
    Construct a multiple-choice query. Inserts <image> placeholder if missing.
    Appends either chain-of-thought (CoT) or direct instruction from config.
    """
    question = sample['question']
    images = sample.get('images', [])
    answer = sample['answer'].strip().upper()

    # Ensure <image> placeholder if images are provided
    if "<image>" not in question and images:
        question += "\n<image>"

    # Fill the template with the question
    prompt_template = config['multi_choice_format']
    filled_prompt = prompt_template.format(question=question)

    # Append the selected strategy instruction
    if strategy == 'CoT':
        query = f"{filled_prompt}\n{config['Strategy_Instruction']['CoT']}"
    else:
        query = f"{filled_prompt}\n{config['Strategy_Instruction']['Directly']}"

    # Build the result dictionary
    res_dict = {
        'query': query,
        'gt_content': answer,
        **sample
    }
    return res_dict