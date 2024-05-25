import json
import os
import argparse
import requests
import torch

from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

def query_with_image(
        model,
        processor,
        image_url,
        screenshot_url,
        query_text,
        use_screenshot=False,
        exp_dir='experiment/snake/',
        sys_msg_filename=None,
    ):
    with open(exp_dir + sys_msg_filename, 'r') as f:
        query_system_msg = f.read()

    query_image = load_image(image_url)
    if use_screenshot: 
        context_text = ("In the screenshot, the large image on the left is the query image for a reverse image search. "
                        "The smaller images on the right and their titles are the top hits from the search. ")
        screenshot_image = load_image(screenshot_url)
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
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[screenshot_image, query_image], return_tensors="pt")
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
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[query_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def main(args):

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
    ).to(DEVICE)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('local_data/snakeclef_data.json', 'r') as f:
        samples = json.load(f)[:30]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)


    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_url = f"https://anonymous.4open.science/api/repo/rir_data/file/snake/{sample['image_path']}"
        screenshot_url = f"https://anonymous.4open.science/api/repo/rir_data/file/snake/{sample['image_path'].split('/')[-1]}-search_result.png"
        if args.metric == 'recall':
            query_text = "What is the binomial name of the snake in the image? Please respond with the binomial name only."
        elif args.metric == 'em':
            query_text = "What is the binomial name of the snake in the image (e.g. Psammophis namibensis, Lampropeltis annulata)? Please respond with the binomial name only."
        response = query_with_image(
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
        if pred[-1] == '.': pred = pred[:-1]
        binomial_em = sample['binomial_name'].lower() == pred.lower()
        genus_em = sample['binomial_name'].split(' ')[0].lower() == pred.split(' ')[0].lower()
        binomial_recall = sample['binomial_name'].lower() in pred.lower()
        genus_recall = sample['binomial_name'].split(' ')[0].lower() in pred.lower()

        if args.metric == 'em':
            logs.append(
                {
                    'idx': idx,
                    'binomial_em': binomial_em,
                    'genus_em': genus_em,
                    'observation_id': sample['observation_id'],
                    'image_path': sample['image_path'],
                    'question': query_text,
                    'answer': sample['binomial_name'],
                    'pred': pred,
                    'full_response': str(response),
                }
            )
        elif args.metric == 'recall':
            logs.append(
                {
                    'idx': idx,
                    'binomial_recall': binomial_recall,
                    'genus_recall': genus_recall,
                    'observation_id': sample['observation_id'],
                    'image_path': sample['image_path'],
                    'question': query_text,
                    'answer': sample['binomial_name'],
                    'pred': pred,
                    'full_response': str(response),
                }
            )


        output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, default='local_data/snake/snake_i2_300')
    argparser.add_argument('--log_name', type=str, default='logs_snake_i2_300')
    argparser.add_argument('--sys_msg_filename', type=str, default='query_system_with_screenshot.jinja2')
    # screenshot
    argparser.add_argument('--use_screenshot', type=int, default=0)
    # addtional
    argparser.add_argument('--idx_offset', type=int, default=0)
    argparser.add_argument('--metric', type=str, choices=['em', 'recall'])

    args = argparser.parse_args()
    main(args)