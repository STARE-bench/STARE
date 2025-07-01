import logging
import re
import base64
from io import BytesIO
import time
import os
from PIL import Image



from openai import OpenAI



def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



def create_message(sample):
    query = sample['query']
    all_contents = []

    # 拆分文本（以 <image> 分隔）
    split_text = re.split(r"<image>", query)

    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.append({"type": "text", "text": fragment})

        # 每段文字后接一张图（除最后一段）
        if i < len(sample['images']):
            image_entry = sample['images'][i]
            try:
                # 如果是路径字符串：拼接路径并打开
                if isinstance(image_entry, str):
                    image_path = os.path.join(IMAGE_ROOT, image_entry)
                    image = Image.open(image_path).convert("RGB")
                elif isinstance(image_entry, Image.Image):
                    image = image_entry
                else:
                    raise ValueError(f"Unsupported image type: {type(image_entry)}")

                img_base64 = encode_image_to_base64(image)
                all_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })

            except Exception as e:
                logging.error(f"❌ Failed to load/encode image at index {i}: {e}")

    messages = [
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages

    # for i, fragment in enumerate(split_text):
    #     if fragment.strip():
    #         all_contents.append({"type": "text", "text": fragment})
        
    #     # 每段文字后可能有一个图像（除了最后一段）
    #     if i < len(sample['images']):
    #         image_data = sample['images'][i]
    #         if image_data:
    #             img_base64 = encode_image_to_base64(image_data)
    #             all_contents.append({
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": f"data:image/png;base64,{img_base64}"
    #                 }
    #             })
    #         else:
    #             logging.error(f"Missing image for <image> at position {i}")

    # messages = [
    #     {
    #         "role": "user",
    #         "content": all_contents
    #     }
    # ]
    # return messages


# build gpt class
class GPT_Model:
    def __init__(
            self,
            client: OpenAI,
            model="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1024,
            retry_attempts = 5
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts

    def get_response(self, sample):
        attempt = 0
        messages = create_message(sample)

        while attempt < self.retry_attempts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")

                if 'error' in str(e) and 'message' in str(e):
                    error_message = str(e)
                    if 'The server had an error processing your request.' in error_message:
                        sleep_time = 30
                        logging.error(f"Server error, retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    elif 'Please try again in ' in error_message:
                        sleep_time = float(error_message.split('Please try again in ')[1].split('s.')[0])
                        logging.error(f"Rate limit exceeded, retrying in {sleep_time * 2}s...")
                        time.sleep(sleep_time * 2)
                    elif 'RESOURCE_EXHAUSTED' in error_message:
                        sleep_time = 30
                        logging.error(f"Gemini rate limit, retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print("Unknown error, skipping this request.")
                        break
                attempt += 1

        return None
