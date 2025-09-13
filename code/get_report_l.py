import json
import openai
import argparse
from gpt_utils import *
from utils import make_prompt
from prompts import get_report_prompt


def run_gpt(model_id, prompt, token_usage):
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
    parser.add_argument("--language", "-l", required=True, help="language of the report")
    args = parser.parse_args()

    # === GPT utils ===
    load_api_key()
    gpt_token_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0
    }

    model_id = "o3"
    processed_ids = set()
    try:
        with open(args.output_file, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    with open(args.input_file, 'r', encoding='utf-8') as infile, open(args.output_file, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            input_id = data.get("id", None)

            if input_id in processed_ids:
                print(f"✅ Skipping ID {input_id} (already processed)")
                continue

            question = data.get("question_src", "")
            context_title = data.get("pos_context_title_src", "")
            context = data.get("pos_context_src", "")

            prompt, id_to_source = make_prompt(
                question=question,
                context_title=context_title,
                context=context,
                prompt_template=get_report_prompt,
                language=args.language,
            )
            print("⛓️ Doc ID-Source: ", id_to_source)
            print("=" * 30, " ✏️ Prompt ✏️ ", "=" * 30)
            print(prompt)

            generation, input_token_count = run_gpt(model_id, prompt, gpt_token_usage)
            data["report_src"] = generation
            data["id_to_source"] = id_to_source
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            print(f"📄 Prompt token length: {input_token_count}")
            print("=" * 30, " 🤖 Response 🤖 ", "=" * 30)
            print(generation)
            print("\n")
