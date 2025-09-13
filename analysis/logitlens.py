import os
import re
import gc
import torch
import json
import argparse
from nnsight import LanguageModel
from transformers import AutoTokenizer
from utils import get_claim_num, remove_citations, make_prompt_main_run


guess_citation_prompt = """Information: {{context}}
---
Using the above information, the response is the answer to the query or task: "{{query}}" in a single sentence.
You MUST cite the most relevant document by including only its Source ID in brackets at the end of the sentence (e.g., [Source ID]).
Do NOT include any additional words inside or outside the brackets.
Please output ONLY the number of the Source ID that is most relevant to the sentence.

Response: {{sentence}} ["""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, required=True, help="model name")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="input file")
    parser.add_argument("--src_input_file", "-s", type=str, required=True, help="source input file")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="output file")
    parser.add_argument("--lang_to_change", "-l", type=str, required=True, help="lang to change")
    args = parser.parse_args()

    lang_to_change = args.lang_to_change

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LanguageModel(args.model_name, device_map="auto", dispatch=True, low_cpu_mem_usage=True)
    print(model)

    if "gemma" in args.model_name.lower():
        num_layers = len(model.model.language_model.layers)
    else:
        num_layers = len(model.model.layers)
    all_first_hit_layers = []

    print("Number of layers: ", num_layers)

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as fout:
            for line in fout:
                data = json.loads(line)
                if "id" in data:
                    processed_ids.add(data["id"])
    print(f"! Found {len(processed_ids)} already processed request IDs.")

    with open(args.input_file, 'r') as fin, open(args.src_input_file, 'r') as src_fin, open(args.output_file, 'a') as fout:
        for input_line, src_input_line in zip(fin, src_fin):
            input_data = json.loads(input_line)
            src_input_data = json.loads(src_input_line)

            id = input_data.get("id")
            if id in processed_ids:
                print(f"⏭️ Skipping already processed id: {id}")
                continue
                
            print(f"🚀 Processing id: {id}")
            query = input_data.get("question_en", "")
            claims = input_data.get("claims_retained", [])
            context = input_data.get("pos_context_en", [])

            correct_layer_nums_by_id = {
                str(idx+1): [] for idx in range(len(context))
            }
            print("🧺 Correct layer nums by id: ", correct_layer_nums_by_id)

            doc_context = make_prompt_main_run(
                context_title=input_data["pos_context_title_en"],
                context=input_data["pos_context_en"],
            )

            all_decoded_pred_dict = {
                "claim " + str(idx+1): [] for idx in range(len(claims))
            }

            i = 1
            for claim in claims:
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
                    if lang_to_change != "en":
                        change_context = src_input_data["pos_context_src"][int(correct_claim_num)-1]
                        change_title = src_input_data["pos_context_title_src"][int(correct_claim_num)-1]
                        
                        # Find the specific document block to replace (to avoid replacing all identical titles)
                        target_doc_id = int(correct_claim_num)
                        target_en_title = input_data["pos_context_title_en"][int(correct_claim_num)-1]
                        target_en_context = input_data["pos_context_en"][int(correct_claim_num)-1]
                        
                        # Create the old and new document blocks
                        old_doc_block = f"Document ID: {target_doc_id}\nTitle: {target_en_title}\nContent: {target_en_context}"
                        new_doc_block = f"Document ID: {target_doc_id}\nTitle: {change_title}\nContent: {change_context}"
                        
                        # Replace only the specific document block
                        doc_context_modified = doc_context.replace(old_doc_block, new_doc_block)
                    else:
                        doc_context_modified = doc_context

                claim_wo_citations = remove_citations(claim)
                print("✍️ Claim: ", claim_wo_citations)

                filled_prompt = guess_citation_prompt.replace("{{context}}", doc_context_modified).replace("{{query}}", query).replace("{{sentence}}", claim_wo_citations)
                print("=" * 30, " ✏️ Prompt ✏️ ", "=" * 30)
                print(filled_prompt)

                # tokenize and locate position of "[" token
                inputs = tokenizer(filled_prompt, return_tensors="pt").to(device)
                input_ids = inputs.input_ids[0]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                bracket_idx = max(i for i, tok in enumerate(tokens) if "[" in tok)
                correct_token_id = tokenizer.encode(correct_claim_num, add_special_tokens=False)[0]
                print(f"Correct citation ID: {correct_claim_num}, Encoded token ID: {correct_token_id}")
                
                # trace through layers to find the first layer where the correct token appears
                probs_layers = []
                with torch.inference_mode(): # or torch.no_grad() for older pytorch
                    with model.trace() as tracer:
                        with tracer.invoke(filled_prompt) as invoker:
                            if "gemma" in args.model_name.lower():
                                for layer_idx, layer in enumerate(model.model.language_model.layers):
                                    # Get logits and softmax probabilities
                                    layer_logits = model.lm_head(model.model.language_model.norm(layer.output[0]))
                                    target_logits = layer_logits[0, bracket_idx]
                                    probs = torch.nn.functional.softmax(target_logits.cpu(), dim=-1).save()
                                    probs_layers.append(probs)
                            else:
                                for layer_idx, layer in enumerate(model.model.layers):
                                    # Get logits and softmax probabilities
                                    layer_logits = model.lm_head(model.model.norm(layer.output[0]))
                                    target_logits = layer_logits[0, bracket_idx]
                                    probs = torch.nn.functional.softmax(target_logits.cpu(), dim=-1).save()
                                    probs_layers.append(probs)

                # Combine probabilities and find predicted token per layer
                probs_tensor = torch.stack([p.value for p in probs_layers])
                pred_token_ids = torch.argmax(probs_tensor, dim=-1).tolist()

                del probs_layers, probs, tokens, probs_tensor, filled_prompt, invoker, tracer
                torch.cuda.empty_cache()
                gc.collect()

                # Search for first layer where prediction matches correct token
                found = False
                all_decoded_pred = []
                for layer_idx, pred_token_id in enumerate(pred_token_ids):
                    decoded_pred = tokenizer.decode([pred_token_id])
                    all_decoded_pred.append(decoded_pred)
                    decoded_correct = tokenizer.decode([correct_token_id])
                    is_match = pred_token_id == correct_token_id

                    print(f"〽️ Layer {layer_idx}: Predicted {pred_token_id} ({decoded_pred}), "
                          f"Correct {correct_token_id} ({decoded_correct}), Match: {is_match}")

                    if is_match:
                        print("✅ Found match at layer:", layer_idx)
                        correct_layer_nums_by_id[correct_claim_num].append(layer_idx)
                        all_first_hit_layers.append(layer_idx)
                        found = True

                if not found:
                    print("❌ No match found in any layer.")
                    correct_layer_nums_by_id[correct_claim_num].append(-1)
                    all_first_hit_layers.append(-1)
                
                all_decoded_pred_dict["claim " + str(i)] = all_decoded_pred
                all_decoded_pred_dict["correct claim " + str(i)] = correct_claim_num
                del decoded_pred, decoded_correct, pred_token_ids
                torch.cuda.empty_cache()
                gc.collect()
                
                print("🧺 Filled correct layer nums by id: ", correct_layer_nums_by_id)
                print("🧺 Filled all decoded predictions for the claim: ", all_decoded_pred)
                print("🧺 Filled dictionary of decoded predictions for the claim: ", all_decoded_pred_dict)
                print("="*100)
                i += 1
        
            input_data["correct_layer_num"] = correct_layer_nums_by_id
            input_data["all_decoded_pred_dict"] = all_decoded_pred_dict
            fout.write(json.dumps(input_data, ensure_ascii=False) + "\n")

    valid_layers = [l for l in all_first_hit_layers if l >= 0]
    print("List of valid layers: ", valid_layers)
