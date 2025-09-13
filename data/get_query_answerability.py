import openai
import json
import argparse
import torch
from utils import *
from gpt_utils import *
from prompts import get_relevance_prompt


def get_judgment_gpt(model_id, prompt, token_usage):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # Count token usage
    input_token_count = count_tokens(model_id, prompt)
    token_usage["input_tokens"] += input_token_count

    generation = response.choices[0].message.content.strip(" \n")
    output_token_count = count_tokens(model_id, generation)
    token_usage["output_tokens"] += output_token_count

    estimate_cost(model_id, token_usage)
    return generation, input_token_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", required=True, help="input jsonl file")
    parser.add_argument("--output_file", "-o", required=True, help="output jsonl file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"☑️ Using device: {device}")

    load_api_key()
    gpt_token_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0
    }
    model_id = "o4-mini"
    
    processed_ids = set()
    try:
        with open(args.output_file, "r", encoding="utf-8") as out_file:
            for line in out_file:
                try:
                    item = json.loads(line)
                    if "id" in item:
                        processed_ids.add(item["id"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass  # Output file doesn’t exist yet

    print(f"! Found {len(processed_ids)} already processed IDs.")

    with open(args.input_file, 'r', encoding='utf-8') as infile, open(args.output_file, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            data_id = data.get("id", "")

            if data_id in processed_ids:
                print(f"⏭️ Skipping ID {data_id} (already processed).")
                continue

            report = data.get("report_src", "")
            query = data.get("question_src", "")
            print(f"☑️ Query: {query}")

            # Formulate as prompt template
            prompt_template = get_relevance_prompt.replace("{{query}}", query).replace("{{report}}", report)
            print("\n")
            print("=" * 30, " ✏️ Prompt ✏️ ", "=" * 30)
            print(prompt_template)

            generation, token_length = get_judgment_gpt(model_id, prompt_template, gpt_token_usage)
            print(f"📄 Prompt token length: {token_length}")
            print("=" * 30, " 🤖 Response 🤖 ", "=" * 30)
            print(generation)
            
            data[f"gpt4o_relevance"] = generation
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
