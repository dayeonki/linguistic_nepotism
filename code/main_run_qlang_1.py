import re
import json
import torch
import os
import argparse
import torch.nn.functional as F
from prompts import guess_citation_prompt
from utils import make_prompt_main_run
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor


def get_existing_ids(output_file):
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'id' in data:
                            existing_ids.add(data['id'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    return existing_ids


class CitationLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for allowed tokens
        mask = torch.full_like(scores, fill_value=float('-inf')) 
        mask[:, self.allowed_token_ids] = 0
        return scores + mask


# Targeted probability 
def get_next_token_probs(model_name, prompt, target_tokens):
    messages = [{"role": "user", "content": prompt}]
    if "qwen" in model_name.lower():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to(device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    next_token_logits = logits[0, -1]  # get logits for next token
    probs = torch.softmax(next_token_logits, dim=-1)

    # Map target strings to token ids and get probabilities
    token_probs = {}
    for tok in target_tokens:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        token_probs[tok] = probs[tok_id].item()

    return token_probs


# Natural distribution
def get_next_token_distribution(model_name, prompt):
    messages = [{"role": "user", "content": prompt}]
    if "qwen" in model_name.lower():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to(device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    next_token_logits = logits[0, -1]  # [vocab_size]
    probs = torch.softmax(next_token_logits, dim=-1)
    token_probs = [(tokenizer.decode([i]), probs[i].item()) for i in range(len(probs))]
    token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)

    return token_probs


# Constrained decoding + probabilities for each allowed token
def get_constrained_decoding(prompt, target_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]

    allowed_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    processor = CitationLogitsProcessor(allowed_token_ids)
    constrained_logits = processor(input_ids, logits)
    probs = F.softmax(constrained_logits, dim=-1)

    allowed_probs = {token: probs[0, token_id].item() for token, token_id in zip(target_tokens, allowed_token_ids)}
    return allowed_probs


def get_entropy(model_name, prompt):
    messages = [{"role": "user", "content": prompt}]
    
    if "qwen" in model_name.lower():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to(device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

    if "aya" in model_name.lower():
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :].to(torch.float32)  # convert to float32

            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * log_probs, dim=-1)
    else:
        with torch.no_grad(): # Get logits
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
            probs = F.softmax(logits, dim=-1)

        # Compute entropy H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return entropy.item()



def get_results(model_name, doc_claim, doc_context, query, input_data, lang_to_change, baseline, target_tokens, id_to_source, output_file):   
    with open(output_file, 'a') as fout:
        if doc_claim:
            for claim in doc_claim:
                if lang_to_change == "bn":
                    claim = claim.replace("।", ".")
                elif lang_to_change == "zh":
                    claim = claim.replace("。", ".")
                else:
                    pass

                correct_claim_num = get_claim_num(claim)
                if correct_claim_num is None:
                    continue
                else:
                    if baseline is False:
                        change_context = input_data["pos_context_src"][int(correct_claim_num)-1]
                        change_title = input_data["pos_context_title_src"][int(correct_claim_num)-1]

                        target_doc_id = int(correct_claim_num)
                        target_en_title = input_data["pos_context_title_en"][int(correct_claim_num)-1]
                        target_en_context = input_data["pos_context_en"][int(correct_claim_num)-1]

                        old_doc_block = f"Document ID: {target_doc_id}\nTitle: {change_title}\nContent: {change_context}"
                        new_doc_block = f"Document ID: {target_doc_id}\nTitle: {target_en_title}\nContent: {target_en_context}"
                        doc_context_modified = doc_context.replace(old_doc_block, new_doc_block)
                    else:
                        doc_context_modified = doc_context
                    
                # Change context to to lang_to_change accordingly
                print("🎯 Target tokens: ", target_tokens)
                print("⛓️ Doc ID-Source: ", id_to_source)

                claim = claim.replace(f" [{correct_claim_num}].", "").replace(f"[{correct_claim_num}].", "").replace(f"[{correct_claim_num}]", "")
                prompt_template = guess_citation_prompt.replace("{{context}}", doc_context_modified).replace("{{query}}", query).replace("{{sentence}}", claim)
                print("\n", "="*30, " ✍️ Prompt Template ✍️ ", "="*30)
                print(prompt_template)

                try:
                    print(f"✅ Correct Claim to be cited: [{correct_claim_num}]")
                    token_probs = get_next_token_probs(model_name, prompt_template, target_tokens)
                    print("\n", "="*30, " 💯 Targeted Token Probability 💯 ", "="*30)
                    print(token_probs)

                    print("\n", "="*30, " 💯 Ordered Targeted Token Probability 💯 ", "="*30)
                    sorted_token_probs = sorted(token_probs.items(), key=lambda item: item[1], reverse=True)
                    sorted_token_probs = dict(sorted_token_probs)
                    print(sorted_token_probs)

                    constrained_tokens = get_constrained_decoding(prompt_template, target_tokens)
                    print("\n", "="*30, " 🪢 Constrained Token Probability 🪢 ", "="*30)
                    print(constrained_tokens)

                    print("\n", "="*30, " 🪢 Order Constrained Token Probability 🪢 ", "="*30)
                    sorted_constrained_token_probs = sorted(constrained_tokens.items(), key=lambda item: item[1], reverse=True)
                    sorted_constrained_token_probs = dict(sorted_constrained_token_probs)
                    print(sorted_constrained_token_probs)

                    print("\n", "="*30, " 🌀 Entropy (lower=more confident) 🌀 ", "="*30)
                    entropy = get_entropy(model_name, prompt_template)
                    print(entropy)

                    top_tokens = get_next_token_distribution(model_name, prompt_template)
                    print("\n", "="*30, " 📈 Next Token Distribution 📈 ", "="*30)
                    for tok, prob in top_tokens[:10]:
                        print(f"{repr(tok)}: {prob:.4f}")
                    print("\n", "-"*150, "\n")
                except:
                    token_probs = ""
                    sorted_token_probs = ""
                    constrained_tokens = ""
                    sorted_constrained_token_probs = ""
                    entropy = ""
                    top_tokens = ""
                
                result = {
                    "id": input_data.get("id", ""),
                    "correct_claim_num": correct_claim_num,
                    "sorted_token_probs": sorted_token_probs,  # sorted dict by prob
                    "sorted_constrained_token_probs": sorted_constrained_token_probs,
                    "entropy": entropy,
                    "claim": claim,
                    "top_next_tokens": top_tokens[:10],
                    "question_src": query,
                    "report_src": input_data.get("report_src", ""),
                    "id_to_source": input_data.get("id_to_source", ""),
                    "all_claims": input_data.get("all_claims", ""),
                    "claims_w_citation": input_data.get("claims_w_citation", ""),
                    "claims_retained": input_data.get("claims_retained", ""),
                    "prompt": prompt_template,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            pass


def get_claim_num(claim):
    matches = re.findall(r'\[([^\]]+)\]', claim)
    return matches[-1] if matches else None


def get_prefix_decoding(model_name, input_file, output_file, lang_to_change, baseline):
    existing_ids = get_existing_ids(output_file)
    with open(input_file, 'r') as fin:
        for input_line in fin:
            input_data = json.loads(input_line)
            query = input_data.get("question_src", "")
            claims = input_data.get("claims_retained", [])
            id_to_source = input_data.get("id_to_source", {})

            if input_data.get("id") in existing_ids:
                print(f"! Skipping ID: {input_data.get('id')} (already processed)")
                continue

            target_tokens = [str(i+1) for i in range(len(input_data["pos_context_src"]))]
                
            prompt = make_prompt_main_run(
                context_title=input_data["pos_context_title_src"],
                context=input_data["pos_context_src"],
            )

            # Get next token prediction - get results()
            get_results(
                model_name=model_name, 
                doc_claim=claims, 
                doc_context=prompt,
                query=query, 
                input_data=input_data,
                lang_to_change=lang_to_change,
                baseline=baseline,
                target_tokens=target_tokens,
                id_to_source=id_to_source,
                output_file=output_file,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", "-m", required=True, help="Huggingface model id")
    parser.add_argument("--input_file", "-i", required=True, help="jsonl file with the claims (reports/each number)")
    parser.add_argument("--output_file", "-o", required=True, help="file to write the results to")
    parser.add_argument("--lang_to_change", "-l", default='ar', help="lang to change for main run")
    parser.add_argument("--baseline", action="store_true", help="baseline or not (if True, use the pos_context_src, else use the pos_context_en)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🤖 Using Device: ", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    get_prefix_decoding(args.model_id, args.input_file, args.output_file, args.lang_to_change, args.baseline)
