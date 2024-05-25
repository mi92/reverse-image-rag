import json
import os
import argparse

from rir_api import RIR_API
from PIL import Image
from tqdm import tqdm
from transformers.image_utils import load_image


def main(args):

    with open('cred.txt', 'r') as f:
        openai_api_key = f.read()
    api = RIR_API(openai_api_key)

    # load sample data
    with open('local_data/snakeclef_data.json', 'r') as f:
        samples = json.load(f)[:30]

    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image = load_image(f"https://anonymous.4open.science/api/repo/rir_data/file/snake/{sample['image_path']}")
        width, height = image.size
        image_url = f"https://anonymous.4open.science/api/repo/rir_data/file/snake/{sample['image_path']}"
        query_text = "What is the binomial name of the snake in the image (e.g. Psammophis namibensis, Lampropeltis annulata)? Please respond with the binomial name only."
        response = api.query_with_image(
            image_url,
            query_text,
            delay=3,
            show_result=False,
            use_screenshot=args.use_screenshot,
            screenshot_dir=args.screenshot_dir,
            model_name=args.model,
            exp_dir=args.exp_dir,
            sys_msg_filename=args.sys_msg_filename,
            generate_new_screenshot=args.generate_new_screenshot,
        )

        try:
            if isinstance(response, tuple):
                response = response[0]
            pred = response.choices[0].message.content
        except:
            raise ValueError(f"Error in response: {response}")
        binomial_em = sample['binomial_name'].lower() == pred.lower()
        genus_em = sample['binomial_name'].split(' ')[0].lower() == pred.split(' ')[0].lower()

        logs.append(
            {
                'idx': idx,
                'binomial_em': binomial_em,
                'genus_em': genus_em,
                'observation_id': sample['observation_id'],
                'image_path': sample['image_path'],
                'image_width': width,
                'image_height': height,
                'question': query_text,
                'answer': sample['binomial_name'],
                'pred': pred,
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'full_response': str(response),
            }
        )

        output_dir = f'{args.output_root}/'
        output_filename = f'{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.log_name}.json'
        output_path = output_dir + output_filename
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--exp_dir', type=str, default='experiment/snake/')
    argparser.add_argument('--output_root', type=str, default='infoseek_gpt4turbo_rir_1650')
    argparser.add_argument('--log_name', type=str, default='logs_infoseek_gpt4turbo_rir_1650')
    argparser.add_argument('--model', type=str, default='gpt-4-turbo-2024-04-09', choices=['gpt-4-turbo-2024-04-09', 'gpt-4-1106-vision-preview','gpt-4o-2024-05-13'])
    argparser.add_argument('--sys_msg_filename', type=str)
    # screenshot
    argparser.add_argument('--use_screenshot', type=int, default=0)
    argparser.add_argument('--generate_new_screenshot', type=int, default=0)
    argparser.add_argument('--screenshot_dir', type=str, default='local_data/snake/snake_screenshot/')
    # additional
    argparser.add_argument('--idx_offset', type=int, default=0)

    args = argparser.parse_args()
    main(args)