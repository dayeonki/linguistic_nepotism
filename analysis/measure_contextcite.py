import os
import re
import json
import csv


def get_claim_num(claim):
    matches = re.findall(r'\[([^\]]+)\]', claim)
    return matches[-1] if matches else None


def get_contextcite(input_dir, output_csv, dataset):
    results = []
    for filename in os.listdir(input_dir):

        hit1, hit3, score1, score3 = 0, 0, 0, 0
        total = 0
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(input_dir, filename)
        language = filepath.replace(".jsonl", "")[-2:]
        print("Language: ", language)
        report_file = f"../code/report/{language}/{dataset}_report_ensemble_single_tok.jsonl"
        en_report_file = f"../code/report/en/{dataset}_report_ensemble_single_tok.jsonl"
        with open(report_file, "r") as freport:
            report_data_dict = {json.loads(line)["id"]: json.loads(line) for line in freport}
        with open(en_report_file, "r") as fenreport:
            en_report_data_dict = {json.loads(line)["id"]: json.loads(line) for line in fenreport}

        with open(filepath, "r") as f:
            for input_line in f:
                input_data = json.loads(input_line)
                report_data = report_data_dict.get(input_data["id"])
                en_report_data = en_report_data_dict.get(input_data["id"])
                if report_data is None or en_report_data is None:
                    continue
                total += 1
                
                context_cite = input_data.get("context_cite", {})
                top_1_score = context_cite[0]["Score"]
                top_1_text = context_cite[0]["Source"].replace("Content: ", "").replace("Title: ", "")
                top_3 = context_cite[:3]
                top_3_score, top_3_text = [], []
                for each in top_3:
                    top_3_score.append(each["Score"])
                    top_3_text.append(each["Source"].replace("Content: ", "").replace("Title: ", ""))

                pos_context_en = report_data.get("pos_context_en", [])
                pos_context_src = report_data.get("pos_context_src", [])
                pos_title_en = report_data.get("pos_context_title_en", [])
                pos_title_src = report_data.get("pos_context_title_src", [])
                claims = en_report_data.get("claims_retained", [])
                
                if input_data["id"] == report_data["id"]:
                    claim = input_data["claim"]
                    if claim in claims:
                        correct_claim_num = get_claim_num(claim)
                        # Compute top1
                        if language != "en":
                            if top_1_text in pos_context_src[int(correct_claim_num)-1] or top_1_text in pos_title_src[int(correct_claim_num)-1]:
                                hit1 += 1
                                score1 += top_1_score
                            else:
                                pass
                        elif language == "en":
                            if top_1_text in pos_context_en[int(correct_claim_num)-1] or top_1_text in pos_title_en[int(correct_claim_num)-1]:
                                hit1 += 1
                                score1 += top_1_score
                            else:
                                pass
                        else:
                            raise ValueError("! Wrong language")
                        
                        # Compute top3 (hit3: only increment by 1 if any of top3 are in the correct context)
                        if language != "en":
                            context_list = pos_context_src[int(correct_claim_num)-1]
                            title_list = pos_title_src[int(correct_claim_num)-1]
                        else:
                            context_list = pos_context_en[int(correct_claim_num)-1]
                            title_list = pos_title_en[int(correct_claim_num)-1]

                        found = False
                        for i in range(len(top_3_score)):
                            each_score = top_3_score[i]
                            each_text = top_3_text[i]
                            if each_text in context_list or each_text in title_list:
                                if not found:
                                    hit3 += 1
                                    score3 += each_score
                                    found = True
        results.append({
            "filename": filename.replace(".jsonl", ""),
            "hit@1": round((hit1 / total), 3) if hit1 is not None else None,
            "hit@3":  round((hit3 / total), 3) if hit3 is not None else None,
            "score@1": round((score1 / total), 3) if score1 is not None else None,
            "score@3": round((score3 / total), 3) if score3 is not None else None,
        })

        print(f"Filename: {filename}")
        print(f"Hit @ 1 acc.: {hit1 / total}")
        print(f"Hit @ 3 acc.: {hit3 / total}")
        print(f"Score @ 1 avg.: {score1 / total}")
        print(f"Score @ 3 avg.: {score3 / total}")
        print("\n", "="*60, "\n")

        if output_csv:
            with open(output_csv, "w", newline="") as csvfile:
                fieldnames = ["filename", "hit@1", "hit@3", "score@1", "score@3"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)



if __name__ == "__main__":
    models = ["llama8b"]
    datasets = ["eli5"]

    for model in models:
        for dataset in datasets:
            input_dir = f"res_contextcite/{model}/{dataset}"
            output_csv = f"vis_contextcite/{dataset}_{model}.csv"
            get_contextcite(input_dir, output_csv, dataset)
