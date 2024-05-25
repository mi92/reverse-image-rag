from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI
from tqdm import tqdm
import json
import argparse

# Your OpenAI API key
with open('cred.txt', 'r') as f:
    openai_api_key = f.read()
client = OpenAI(api_key=openai_api_key)

def evaluate_response(template, question, gold_answer, answer_eval, model_answer):
    """
    Uses GPT-4-turbo to evaluate if a given response is correct compared to a gold answer.
    
    :param template: The template prompt to use.
    :param question: The question.
    :param gold_answer: The correct (gold) answer.
    :param model_answer: The response to evaluate.
    :return: GPT-4-turbo's judgment on the correctness of the response.
    """
    prompt = template.render({
        'question': question,
        'gold_answer': gold_answer,
        'answer_eval': f"range: {answer_eval[0]['range']}" if isinstance(answer_eval[0], dict) else answer_eval,
        'model_answer': model_answer
    })

    with open('experiment/infoseek/judge_system.jinja2', 'r') as f:
        system_prompt = f.read()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
    )
    messages=[
        {'role': 'system', 'content': system_prompt},
        {"role": "user", "content": prompt}
    ],

    # Extracting and returning the AI's judgment
    return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens, prompt, messages


def main(args):
    # Set up the environment
    env = Environment(
        loader=FileSystemLoader(searchpath="./"),  # Directory containing your .jinja2 file
        autoescape=select_autoescape(['html', 'xml'])  # Autoescape for safety if generating HTML/XML
    )

    # Load the template
    template = env.get_template('experiment/infoseek/judge_answer.jinja2')

    with open(f'{args.output_root}/{args.exp_name}.json', 'r') as f:
        data = json.load(f)
    with open(f'{args.output_root}/samples.json', 'r') as f:
        samples = json.load(f)
    if args.skip_440:
        samples = samples[440:]
    
    judged_logs = []
    for entry, sample in tqdm(zip(data, samples)):
        pred = entry['pred']
        output, usage_completion, usage_prompt, prompt, messages = evaluate_response(template, sample['question'], sample['answer'], sample['answer_eval'], pred)

        entry['judge_text'] = output
        entry['judge_correct'] = True if output.strip().lower() == 'yes' else False
        entry['judge_completion_tokens'] = usage_completion
        entry['judge_prompt_tokens'] = usage_prompt
        entry['judge_prompt'] = prompt
        entry['judge_full_message'] = messages
        judged_logs.append(entry)
    
        with open(f'{args.output_root}/{args.model}_judged_{args.exp_name}.json', 'w') as f:
            json.dump(judged_logs, f, indent=4)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--exp_name', type=str, required=True)
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--skip_440', type=int, default=0)
    args = argparser.parse_args()
    main(args)