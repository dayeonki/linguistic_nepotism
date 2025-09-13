import json


languages = ["bn", "zh"]

for language in languages:
    print("========== ", language, " ==========")
    files_and_fields = [
        (f"report/{language}/miracl_n1_report_gpto4.jsonl", "gpto4_judgment"),
        (f"report/{language}/miracl_n1_report_qwen.jsonl", "qwen_judgment"),
        (f"report/{language}/miracl_n1_report_gemini.jsonl", "gemini_judgment"),
    ]
    nli_file = f"report/{language}/miracl_n1_report_nli.jsonl"
    save_file = f"report/{language}/miracl_n1_report_ensemble.jsonl"

    judgment_data = {}
    nli_data = {}

    for path, field in files_and_fields:
        with open(path, 'r') as f:
            for line in f:
                ex = json.loads(line)
                id_ = ex["id"]
                judgments = ex.get(field, [])
                if not judgments:
                    continue
                if id_ not in judgment_data:
                    judgment_data[id_] = []
                judgment_data[id_].append(judgments)

    with open(nli_file, 'r') as f:
        for line in f:
            ex = json.loads(line)
            nli_data[ex["id"]] = ex["nli_judgment"]

    results = {}
    total_compared = 0
    total_claims_retained = 0
    total_claims_considered = 0
    total_nli_filtered = 0
    total_nli_retained = 0
    for id_, judgments_list in judgment_data.items():
        if len(judgments_list) != 3:
            continue  # Skip if not all 3 judgments are available

        total_compared += 1
        num_claims = len(judgments_list[0])
        retained_indexes = []

        for i in range(num_claims):
            votes = [jl[i] for jl in judgments_list]
            true_count = votes.count("True")
            if true_count >= 2:
                retained_indexes.append(i)

        # Apply NLI filtering
        filtered_indexes = []
        nli_judgments_for_retained = []
        for idx in retained_indexes:
            try:
                nli = nli_data[id_][idx]
                if max(nli, key=nli.get) != "contradiction":
                    filtered_indexes.append(idx)
                    nli_judgments_for_retained.append(nli)
                else:
                    total_nli_filtered += 1
            except:
                print(f"Error: {id_} {idx}")
                total_nli_filtered += 1
                continue

        results[id_] = {
            "total_claims": num_claims,
            "retained_indexes": filtered_indexes,
            "nli_judgments": nli_judgments_for_retained
        }
        total_claims_retained += len(retained_indexes)
        total_claims_considered += num_claims
        total_nli_retained += len(filtered_indexes)


    with open(save_file, "w") as out_f:
        with open(files_and_fields[0][0], "r") as base_f:
            for line in base_f:
                ex = json.loads(line)
                ex.pop("gpto4_judgment", None)
                id_ = ex["id"]
                retained_claims = []
                if id_ in results:
                    retained_claims = [
                        ex["claims_w_citations"][idx]
                        for idx in results[id_]["retained_indexes"]
                    ]
                ex["claims_retained"] = retained_claims
                out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")


    print("========== Majority Voting ==========")
    print(f"Total claims considered: {total_claims_considered}")
    print(f"Total claims retained: {total_claims_retained}")
    print(f"Retain rate (before NLI): {total_claims_retained / total_claims_considered:.2%}")

    print("\n========== NLI Filtering ==========")
    print(f"Total claims removed due to contradiction: {total_nli_filtered}")
    print(f"Total claims retained after NLI filtering (Final claims): {total_nli_retained}")
    if total_claims_retained > 0:
        print(f"Retain rate (after NLI): {total_nli_retained / total_claims_retained:.2%}")
