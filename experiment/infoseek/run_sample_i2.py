import json
import os
import argparse

from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

def load_image_wrapper(image_url):
    cnt = 0
    while cnt < 3:
        try:
            print('attempting to load image...')
            img = load_image(image_url)
            return img
        except:
            cnt += 1
            print(f'attempt ({cnt+1}/3) failed')
    return None

def query_with_image(
        model,
        processor,
        image_url,
        screenshot_url,
        query_text,
        use_screenshot=False,
        exp_dir='experiment/infoseek/',
        sys_msg_filename=None,
    ):
    with open(exp_dir + sys_msg_filename, 'r') as f:
        query_system_msg = f.read()

    query_image = load_image_wrapper(image_url)
    if use_screenshot: 
        context_text = ("In the screenshot, the large image on the left is the query image for a reverse image search. "
                        "The smaller images on the right and their titles are the top hits from the search. ")
        screenshot_image = load_image_wrapper(screenshot_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image"},
                    {'type': 'text', 'text': context_text},
                    {"type": "image"},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        messages_record = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image", 'url': image_url},
                    {'type': 'text', 'text': context_text},
                    {"type": "image", 'url': screenshot_url},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[query_image, screenshot_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image"},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        messages_record = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image", 'url': image_url},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[query_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts, messages_record


def main(args):

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
    ).to(DEVICE)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('local_data/infoseek_data.json', 'r') as f:
        data = json.load(f)
    samples = [_ for cat_data in data.values() for _ in cat_data]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)


    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_url = f"https://anonymous.4open.science/api/repo/rir_data/file/infoseek/{sample['image_id']}"
        screenshot_url = f"https://anonymous.4open.science/api/repo/rir_data/file/screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response, messages_record = query_with_image(
            model,
            processor,
            image_url,
            screenshot_url,
            query_text,
            use_screenshot=args.use_screenshot,
            exp_dir='experiment/infoseek/',
            sys_msg_filename=args.sys_msg_filename,
        )

        pred = response[0].split('Assistant: ')[-1]
        if isinstance(sample['answer'], list):
            answer_in_pred = any(_.lower() in pred.lower() for _ in sample['answer'])
        else:
            answer_in_pred = sample['answer'].lower() in pred.lower()

        logs.append(
            {
                'idx': idx,
                'answer_in_pred': answer_in_pred,
                'data_id': sample['data_id'],
                'image_id': sample['image_id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'pred': pred,
                'answer_eval': sample['answer_eval'],
                'full_messages_record': str(messages_record),
                'full_response': str(response),
            }
        )

        output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename', type=str, required=True)
    # screenshot
    argparser.add_argument('--use_screenshot', type=int, required=True)
    # addtional
    argparser.add_argument('--idx_offset', type=int, required=True)

    args = argparser.parse_args()
    main(args)