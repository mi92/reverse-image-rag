import json
import os
import argparse

from rir_api import RIR_API
from PIL import Image
from tqdm import tqdm
from transformers.image_utils import load_image


def main(args):

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)
    with open('cred.txt', 'r') as f:
        openai_api_key = f.read()
    api = RIR_API(openai_api_key)

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
        image = load_image(f"https://raw.githubusercontent.com/liamjxu/rir_data/main/infoseek/{sample['image_id']}")
        width, height = image.size
        image_url = f"https://raw.githubusercontent.com/liamjxu/rir_data/main/infoseek/{sample['image_id']}"
        query_text = sample['question']
        response = api.query_with_image(
            image_url,
            query_text,
            delay=3,
            show_result=False,
            use_screenshot=args.use_screenshot,
            screenshot_dir=args.screenshot_dir,
            model_name=args.model,
            exp_dir='experiment/infoseek/',
            sys_msg_filename=args.sys_msg_filename,
            generate_new_screenshot=args.generate_new_screenshot
        )
        
        # gpt4o-2024-05-13 returns a tuple
        try:
            if isinstance(response, tuple):
                response = response[0]
            pred = response.choices[0].message.content
        except:
            raise ValueError(f"Error in response: {response}")

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
                'image_width': width,
                'image_height': height,
                'question': sample['question'],
                'answer': sample['answer'],
                'pred': pred,
                'answer_eval': sample['answer_eval'],
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
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
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--sys_msg_filename', type=str)
    # screenshot
    argparser.add_argument('--use_screenshot', type=int, required=True)
    argparser.add_argument('--generate_new_screenshot', type=int, required=True)
    argparser.add_argument('--screenshot_dir', type=str, required=True)
    # addtional
    argparser.add_argument('--idx_offset', type=int, required=True)

    args = argparser.parse_args()
    main(args)