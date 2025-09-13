import json
import argparse
import pandas as pd
import numpy as np
from glob import glob
import re
import os

# Accuracy
def get_acc_natural(input_file, constrained=False):
    total_count, correct_count = 0, 0
    with open(input_file, 'r') as fin:
        for input_line in fin:
            input_data = json.loads(input_line)
            correct_claim_num = input_data.get("correct_claim_num")
            if constrained:
                sorted_token_probs = input_data.get("sorted_constrained_token_probs")
            else:
                sorted_token_probs = input_data.get("sorted_token_probs")
            try:
                (first_key, first_value), *rest = sorted_token_probs.items()
                highest_token = first_key
                total_count += 1
                if str(correct_claim_num) == str(highest_token):
                    correct_count += 1
            except:
                pass
    accuracy = correct_count / total_count
    return accuracy, correct_count, total_count


# Probability
def get_probability_natural(input_file, constrained=False):
    correct_probability, correct_count = 0.0, 0
    with open(input_file, 'r') as fin:
        for input_line in fin:
            input_data = json.loads(input_line)
            correct_claim_num = input_data.get("correct_claim_num")
            if constrained:
                sorted_token_probs = input_data.get("sorted_constrained_token_probs")
            else:
                sorted_token_probs = input_data.get("sorted_token_probs")
            
            try:
                (first_key, first_value), *rest = sorted_token_probs.items()
                highest_token = first_key
                if str(correct_claim_num) == str(highest_token):
                    correct_probability += first_value
                    correct_count += 1
            except:
                pass
    probability = correct_probability / correct_count
    return probability


# Entropy
def get_entropy(input_file):
    entropys = []
    with open(input_file, 'r') as fin:
        for input_line in fin:
            input_data = json.loads(input_line)
            entropy_val = input_data.get("entropy", 0.0)
            if type(entropy_val) is not str:
                entropy = round(float(entropy_val), 4)
            entropys.append(entropy)
    avg_entropy = sum(entropys) / len(entropys)
    return avg_entropy


# Accuracy by position bin
def get_accuracy_by_doc_position(input_file, num_bins=5, constrained=False):
    positions = []  # (relative_position, is_correct)
    with open(input_file, 'r') as fin:
        for input_line in fin:
            input_data = json.loads(input_line)
            correct_claim_num = input_data.get("correct_claim_num")
            if constrained:
                sorted_token_probs = input_data.get("sorted_constrained_token_probs")
            else:
                sorted_token_probs = input_data.get("sorted_token_probs")
            if not sorted_token_probs:
                continue
            try:
                (first_key, first_value), *rest = sorted_token_probs.items()
                if str(first_key) == str(correct_claim_num):
                    if int(correct_claim_num) == 1:
                        rel_pos = 0.0
                    else:
                        rel_pos = int(correct_claim_num) / len(sorted_token_probs)
                    is_correct = True
                elif str(first_key) != str(correct_claim_num):
                    if int(correct_claim_num) == 1:
                        rel_pos = 0.0
                    else:
                        rel_pos = int(correct_claim_num) / len(sorted_token_probs)
                    is_correct = False
            except: pass
            positions.append((rel_pos, is_correct))
    
    # Bin the positions
    if not positions:
        return [], []
    rel_positions = [p for p, _ in positions]
    corrects = [c for _, c in positions]
    bins = np.linspace(0, 1, num_bins + 1)

    if not rel_positions:
        return [], bins
    bin_indices = np.digitize(rel_positions, bins, right=False)
    
    # Ensure bin_indices is always a list of ints
    if isinstance(bin_indices, int):
        bin_indices = [bin_indices]
    elif isinstance(bin_indices, np.generic):
        bin_indices = [int(bin_indices)]
    elif isinstance(bin_indices, np.ndarray):
        bin_indices = bin_indices.tolist()
    else:
        bin_indices = list(bin_indices)
    corrects = list(corrects)
    acc_per_bin = []

    for i in range(1, num_bins + 1):
        bin_corrects = [is_corr for is_corr, b in zip(corrects, bin_indices) if b == i]
        acc = np.mean(bin_corrects) if bin_corrects else float('nan')
        acc_per_bin.append(acc)

    return acc_per_bin, bins


def get_confidence_interval(acc, n, alpha=0.05):
    if n == 0:
        return (float('nan'), float('nan'))
    z = 1.96  # for 95% CI
    se = np.sqrt(acc * (1 - acc) / n)
    return (acc - z * se, acc + z * se)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="input file to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="output file to save")
    args = parser.parse_args()

    examples = []
    for input_file in glob(f"{args.input_dir}/*"):
        print(f"📂 Input File: {input_file}")
        # Accuracy
        acc_natural, correct_count_natural, total_count_natural = get_acc_natural(input_file, constrained=False)
        acc_constrained, correct_count_constrained, total_count_constrained = get_acc_natural(input_file, constrained=True)

        # Probability
        prob_natural = get_probability_natural(input_file, constrained=False)
        prob_constrained = get_probability_natural(input_file, constrained=True)

        # Entropy
        avg_entropy = get_entropy(input_file)

        # Accuracy by position bin
        acc_per_bin_natural, bins = get_accuracy_by_doc_position(input_file, num_bins=6, constrained=False)
        acc_per_bin_constrained, bins = get_accuracy_by_doc_position(input_file, num_bins=6, constrained=True)
        print(bins)

        ci_natural = get_confidence_interval(acc_natural, total_count_natural)
        ci_constrained = get_confidence_interval(acc_constrained, total_count_constrained)

        example = {
            "file": input_file.replace("../code/result/", "").replace("../code/result_qlang/", "").replace("../code/result_qlang2/", "").replace("../code/result_qlang3/", ""),
            "acc_natural": round(acc_natural, 4),
            "correct_count_natural": correct_count_natural,
            "total_count_natural": total_count_natural,
            "prob_natural": round(prob_natural, 4),

            "acc_constrained": round(acc_constrained, 4),
            "correct_count_constrained": correct_count_constrained,
            "total_count_constrained": total_count_constrained,
            "prob_constrained": round(prob_constrained, 4),

            "avg_entropy": round(float(avg_entropy), 4),

            "acc_per_bin_natural": [float(round(acc, 4)) for acc in acc_per_bin_natural],
            "acc_per_bin_constrained": [float(round(acc, 4)) for acc in acc_per_bin_constrained],
            "acc_natural_ci_low": round(ci_natural[0], 4),
            "acc_natural_ci_high": round(ci_natural[1], 4),
            "acc_constrained_ci_low": round(ci_constrained[0], 4),
            "acc_constrained_ci_high": round(ci_constrained[1], 4),
        }
        examples.append(example)

    def sort_key(row):
        fname = row["file"]
        base = os.path.basename(fname)
        m = re.match(r"(eli5)_(.*?)(_baseline)?\.jsonl", base)
        if not m:
            return (99, fname)
        prefix, model, en = m.groups()
        if prefix == "eli5" and en:
            order = 0
        elif prefix == "eli5" and not en:
            order = 1
        else:
            order = 99
        return (model, order)

    df = pd.DataFrame(examples)
    df["sort_key"] = df.apply(sort_key, axis=1)
    df = df.sort_values(by="sort_key")
    df = df.drop(columns=["sort_key"])
    df.to_csv(args.output_file, index=False)
