import re
import torch
import openai
import json
import argparse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from utils import *
from gpt_utils import *
from prompts import get_claim_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification



def get_entailment_score(nli_model, nli_tokenizer, premise, hypothesis):
    input_data = nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = nli_model(input_data["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    print(f"Prediction → {prediction}")
    return prediction


def get_judgment_qwen(model, tokenizer, prompt, prefix="Response: "):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    tokenized = tokenizer(prompt, return_tensors="pt")
    token_length = tokenized["input_ids"].shape[-1]
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    generation = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    if prefix in generation:
        try: clean_generation = generation.split(prefix, 1)[-1].strip()
        except: clean_generation = generation.strip()
    else: clean_generation = generation
    return clean_generation, token_length


def get_judgment_gemini(client, prompt):
    model_name = "gemini-2.5-pro"

    request_body = {
        "instances": [{"content": prompt}],
        "parameters": {
            "temperature": 0.0,
            "maxOutputTokens": 128
        }
    }
    response = client.models().predict(name=model_name, body=request_body).execute()
    generation = response["predictions"][0]["content"].strip() if response.get("predictions") else ""
    token_length = len(prompt.split())
    return generation, token_length


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
    parser.add_argument("--mode", "-m", required=True, help="mode")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"☑️ Using device: {device}")
    print(f"☑️ Mode: {args.mode}")

    load_api_key()
    gpt_token_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0
    }

    # Load model once
    if args.mode == "nli":
        model_id = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        nli_tokenizer = AutoTokenizer.from_pretrained(model_id)
        nli_tokenizer.pad_token = nli_tokenizer.eos_token
    
    elif args.mode == "qwen":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-32B",
            torch_dtype="auto",
            device_map="auto"
        )
    
    elif args.mode == "gemini":
        creds = service_account.Credentials.from_service_account_file(
            "service_account.json"
        )
        client = build("gemini", "v1alpha", credentials=creds)
    
    elif args.mode == "gpto4":
        model_id = "o4-mini"
    
    else:
        raise ValueError(f"! Invalid mode: {args.mode}")
    

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

            # Get document context & claim
            doc_context = data.get("pos_context_en", [])
            report = data.get("report_en", "")
            query = data.get("question_en", "")
            print(f"☑️ Query: {query}")

            # Remove citation from claims
            all_claims, _ = split_by_claims(report)

            generations = []
            claim_w_citations = []
            for claim in all_claims:
                context_to_check = []
                matches = re.findall(r'\[(\d+)\]', claim)
                for match in matches:
                    try:
                        context_to_check.append(doc_context[int(match)-1])
                        claim_w_citations.append(claim)
                    except:
                        pass
                if len(context_to_check) == 1:
                    context = context_to_check[0]
                    # Formulate as prompt template
                    prompt_template = get_claim_prompt.replace("{{query}}", query).replace("{{context}}", context).replace("{{claim}}", claim)
                    print("\n")
                    print("=" * 30, " ✏️ Prompt ✏️ ", "=" * 30)
                    print(prompt_template)

                    if args.mode == "nli":
                        prediction = get_entailment_score(nli_model, nli_tokenizer, context, claim)
                        generations.append(prediction)
                        print("=" * 30, " 🤖 Entailment Response 🤖 ", "=" * 30)
                        print(prediction)
                    
                    elif args.mode == "qwen":
                        generation, token_length = get_judgment_qwen(model, tokenizer, prompt_template)
                        generations.append(generation)
                        print(f"📄 Prompt token length: {token_length}")
                        print("=" * 30, " 🤖 Response 🤖 ", "=" * 30)
                        print(generation)
                    
                    elif args.mode == "gemini":
                        generation, token_length = get_judgment_gemini(client, prompt_template)
                        generations.append(generation)
                        print(f"📄 Prompt token length: {token_length}")
                        print("=" * 30, " 🤖 Response 🤖 ", "=" * 30)
                        print(generation)
                    
                    elif args.mode == "gpto4":
                        generation, token_length = get_judgment_gpt(model_id, prompt_template, gpt_token_usage)
                        generations.append(generation)
                        print(f"📄 Prompt token length: {token_length}")
                        print("=" * 30, " 🤖 Response 🤖 ", "=" * 30)
                        print(generation)
                
                    else:
                        raise ValueError(f"! Invalid mode: {args.mode}")
                else:
                    print(f"! Skipping claim for no/multiple citations: {claim}")
                    continue
            
            data["all_claims"] = all_claims
            data["claims_w_citations"] = claim_w_citations
            data[f"{args.mode}_judgment"] = generations
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
