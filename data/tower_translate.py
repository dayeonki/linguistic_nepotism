import os
import re
import json
import argparse
import torch
from transformers import pipeline


def tower_postprocess(text):
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, re.DOTALL)
    if match:
        model_response = match.group(1).strip()
    else:
        model_reponse = ""
    return model_response


def tower_translate(input_path, output_path, target_lang, dataset):
    pipe = pipeline(
        "text-generation", 
        model="Unbabel/TowerInstruct-7B-v0.2", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    print("Target language:", target_lang)
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                try:
                    item = json.loads(line)
                    processed_ids.add(item["id"])
                except Exception:
                    continue
        print(f"! Loaded {len(processed_ids)} already processed items. Skipping those.")
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if data["id"] in processed_ids:
                continue
            
            q = data.get('question_en', '').strip()
            pos_titles = data.get("pos_context_title_en", [])
            pos_contexts = data.get("pos_context_en", [])

            pos_title_generations, pos_context_generations = [], []

            # Question
            q_msg = [
                {"role": "user", "content": f"Translate the following text from English into {target_lang}.\nEnglish: {q}\n{target_lang}:"},
            ]
            q_prompt = pipe.tokenizer.apply_chat_template(q_msg, tokenize=False, add_generation_prompt=True)
            q_output = pipe(q_prompt, max_new_tokens=256, do_sample=False)
            q_generation = tower_postprocess(q_output[0]["generated_text"])

            # Positive title + context
            for pos_title in pos_titles:
                pos_title_msg = [
                    {"role": "user", "content": f"Translate the following text from English into {target_lang}.\nEnglish: {pos_title}\n{target_lang}:"},
                ]
                pos_title_prompt = pipe.tokenizer.apply_chat_template(pos_title_msg, tokenize=False, add_generation_prompt=True)
                pos_title_output = pipe(pos_title_prompt, max_new_tokens=256, do_sample=False)
                pos_title_generation = tower_postprocess(pos_title_output[0]["generated_text"])
                pos_title_generations.append(pos_title_generation)
            
            for pos_context in pos_contexts:
                pos_context_msg = [
                    {"role": "user", "content": f"Translate the following text from English into {target_lang}.\nEnglish: {pos_context}\n{target_lang}:"},
                ]
                pos_context_prompt = pipe.tokenizer.apply_chat_template(pos_context_msg, tokenize=False, add_generation_prompt=True)
                pos_context_output = pipe(pos_context_prompt, max_new_tokens=1024, do_sample=False)
                pos_context_generation = tower_postprocess(pos_context_output[0]["generated_text"])
                pos_context_generations.append(pos_context_generation)
            
            data['question_src'] = q_generation
            data['pos_context_title_src'] = pos_title_generations
            data['pos_context_src'] = pos_context_generations

            # Negative title + context
            if dataset == "miracl":
                neg_titles = data.get("neg_context_title_en", [])
                neg_contexts = data.get("neg_context_en", [])

                neg_title_generations, neg_context_generations = [], []
                for neg_title in neg_titles:
                    neg_title_msg = [
                        {"role": "user", "content": f"Translate the following text from English into {target_lang}.\nEnglish: {neg_title}\n{target_lang}:"},
                    ]
                    neg_title_prompt = pipe.tokenizer.apply_chat_template(neg_title_msg, tokenize=False, add_generation_prompt=True)
                    neg_title_output = pipe(neg_title_prompt, max_new_tokens=256, do_sample=False)
                    neg_title_generation = tower_postprocess(neg_title_output[0]["generated_text"])
                    neg_title_generations.append(neg_title_generation)
                
                for neg_context in neg_contexts:
                    neg_context_msg = [
                        {"role": "user", "content": f"Translate the following text from English into {target_lang}.\nEnglish: {neg_context}\n{target_lang}:"},
                    ]
                    neg_context_prompt = pipe.tokenizer.apply_chat_template(neg_context_msg, tokenize=False, add_generation_prompt=True)
                    neg_context_output = pipe(neg_context_prompt, max_new_tokens=1024, do_sample=False)
                    neg_context_generation = tower_postprocess(neg_context_output[0]["generated_text"])
                    neg_context_generations.append(neg_context_generation)
                
                data['neg_context_title_src'] = neg_title_generations
                data['neg_context_src'] = neg_context_generations
            
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", required=True, help="jsonl file with the claims")
    parser.add_argument("--output_file", "-o", required=True, help="file to write the results to")
    parser.add_argument("--target_lang", '-l', default='Arabic', help="target language")
    parser.add_argument("--dataset", '-d', default='eli5', help="dataset to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🤖 Using Device: ", device)

    tower_translate(args.input_file, args.output_file, args.target_lang, args.dataset)
