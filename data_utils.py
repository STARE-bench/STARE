import yaml
import json


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True



def build_query(sample, config, strategy):
    """
    Build a multiple-choice query from simplified sample data structure.
    Assumes the question already contains <image> placeholders and options text.
    """
    question = sample['question']
    images = sample.get('images', [])
    answer = sample['answer'].strip().upper()
    res_dict = {}

    # 如果没有 <image>，但提供了图像，自动添加一个
    if "<image>" not in question and images:
        question += "\n<image>"

    # 构建 prompt（仅插入 question）
    prompt_template = config['multi_choice_format']
    filled_prompt = prompt_template.format(question=question)

    # 添加提示策略（CoT / Directly）
    if strategy == 'CoT':
        query = filled_prompt + "\n" + config['Strategy_Instruction']['CoT']
    else:
        query = filled_prompt + "\n" + config['Strategy_Instruction']['Directly']

    # 构造最终结果字典
    res_dict['query'] = query
    res_dict['gt_content'] = answer  # 直接返回字母答案
    res_dict.update(sample)
    return res_dict



# def build_query(sample, config, strategy):
#     """Build the text query by combining the context, question and options. The <image_n> token is still there"""
#     context = sample['context']
#     question = sample['question']
#     example = ""
#     res_dict = {}
#     if sample['type'].lower() == 'multiple choice':
#         options = sample['options']
#         start_chr = 'A'
#         for option in options:
#             example += f"{start_chr}: {option}\n"
#             start_chr = chr(ord(start_chr) + 1)
#         empty_prompt_sample_structure = config['multi_choice_format']
#         empty_prompt = empty_prompt_sample_structure.format(context=context, question=question, options=example)
#         if strategy == 'CoT':
#             res_dict['query'] = empty_prompt + config['Strategy_Instruction']['CoT']
#         else:
#             res_dict['query'] = empty_prompt + config['Strategy_Instruction']['Directly']

#         res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
#     else:
#         empty_prompt_sample_structure = config['open_ended_format']
#         empty_prompt = empty_prompt_sample_structure.format(context=context, question=question)
#         if strategy == 'CoT':
#             res_dict['query'] = empty_prompt + config['Strategy_Instruction']['CoT']
#         else:
#             res_dict['query'] = empty_prompt + config['Strategy_Instruction']['Directly']
#         res_dict['gt_content'] = sample['answer']

#     # append existing key and value in data
#     res_dict.update(sample)
#     return res_dict
