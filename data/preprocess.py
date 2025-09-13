import json
import random


def select_subset(input_path, output_path, seed=25):
    random.seed(seed)
    print("Using random seed: ", seed)
    examples = []

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            examples.append(data)
        selected_examples = random.sample(examples, 270)

        for selected_example in selected_examples:
            outfile.write(json.dumps(selected_example, ensure_ascii=False) + "\n")


def preprocess_miracl(input_path, output_path, seed=25):
    random.seed(seed)
    print("Using random seed: ", seed)
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        examples = []
        for line in infile:
            data = json.loads(line)
            context_ids = [ctx['docid'] for ctx in data['pos_context']]
            context_titles = [ctx.get('title', '') for ctx in data['pos_context']]
            context_texts = [ctx['text'] for ctx in data['pos_context']]

            n_context_ids = [ctx['docid'] for ctx in data['neg_context']]
            n_context_titles = [ctx.get('title', '') for ctx in data['neg_context']]
            n_context_texts = [ctx['text'] for ctx in data['neg_context']]
            print(len(context_ids), len(n_context_ids))

            if len(context_ids) == 1 and len(n_context_ids) >= 1:
                example = {
                    "id": data["query_id"],
                    "question_en": data["query"],
                    "pos_context_id": context_ids,
                    "pos_context_title_en": context_titles,
                    "pos_context_en": context_texts,
                    "neg_context_id": n_context_ids,
                    "neg_context_title_en": n_context_titles,
                    "neg_context_en": n_context_texts,
                }
                outfile.write(json.dumps(example, ensure_ascii=False) + "\n")
                examples.append(example)
            else:
                continue


def preprocess_eli5(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        data = json.load(infile)
        for item in data:
            example = {
                "id": item["question_id"],
                "question_en": item["question"],
                "pos_context_id": [it["doc_id"] for it in item["docs"]],
                "pos_context_title_en": [it["title"] for it in item["docs"]],
                "post_context_en": [it["text"] for it in item["docs"]]
            }
            outfile.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = f"miracl/raw/en.jsonl"
    output_path = f"miracl/en_n1.jsonl"
    selected_path = f"miracl/en_n2_270.jsonl"
    preprocess_miracl(input_path, output_path)
    select_subset(output_path, selected_path)

    input_path = f"eli5/en_raw.json"
    output_path = f"eli5/en_processed.jsonl"
    preprocess_eli5(input_path, output_path)
