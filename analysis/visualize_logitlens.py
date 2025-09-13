import json
import matplotlib.pyplot as plt
import numpy as np
import os

def is_valid_doc_number(tok, total_context_len):
    try:
        num = int(tok)
        return 1 <= num <= total_context_len
    except ValueError:
        return False

def count_all_tgt_en_predictions(input_file):
    tgt_counts = []
    en_counts = []

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            input_data = json.loads(line)
            pos_context_en = input_data.get("pos_context_en", [])
            total_context_len = len(pos_context_en)
            all_decoded_pred_dict = input_data.get("all_decoded_pred_dict", {})

            for k, v in all_decoded_pred_dict.items():
                if k.startswith("claim "):
                    claim_id = k.split(" ")[1]
                    correct_key = f"correct claim {claim_id}"
                    correct_num_str = str(all_decoded_pred_dict.get(correct_key, ""))

                    for layer_idx, tok in enumerate(v):
                        if is_valid_doc_number(tok, total_context_len):
                            # Extend counts arrays if needed
                            if layer_idx >= len(tgt_counts):
                                tgt_counts.extend([0] * (layer_idx + 1 - len(tgt_counts)))
                                en_counts.extend([0] * (layer_idx + 1 - len(en_counts)))
                            if tok == correct_num_str:
                                # Increment for all later layers as well
                                for future_idx in range(layer_idx, len(tgt_counts)):
                                    tgt_counts[future_idx] += 1
                            else:
                                for future_idx in range(layer_idx, len(en_counts)):
                                    en_counts[future_idx] += 1

    return tgt_counts, en_counts

if __name__ == "__main__":
    languages = ["ar", "bn", "es", "fr", "ko", "ru", "sw", "zh"]  # 8 langs
    models = ["llama8b", "aya8b", "qwen8b", "qwen14b", "gemma27b", "llama70b"]
    model_map = {
        "llama8b": "LLaMA-3.1 8B",
        "aya8b": "Aya23 8B",
        "qwen8b": "Qwen-3 8B",
        "qwen14b": "Qwen-3 14B",
        "gemma12b": "Gemma-3 12B",
        "gemma27b": "Gemma-3 27B",
        "llama70b": "LLaMA-3.3 70B"
    }
    datasets = ["eli5"]

    for dataset in datasets:
        for model in models:
            base_path = f"{model}/{dataset}"
            model_name = model_map[model]

            fig, axs = plt.subplots(2, 4, figsize=(14, 4), sharex=True, sharey=True)
            num_layers_to_show = 14

            for i, lang in enumerate(languages):
                input_file = os.path.join(base_path, f"{dataset}_{model}_{lang}.jsonl")
                tgt_counts, en_counts = count_all_tgt_en_predictions(input_file)

                # Slice to last 15 layers (handle if less than 15 layers available)
                if len(tgt_counts) >= num_layers_to_show:
                    tgt_slice = tgt_counts[-num_layers_to_show:]
                    en_slice = en_counts[-num_layers_to_show:]
                else:
                    tgt_slice = tgt_counts
                    en_slice = en_counts

                layers = range(len(tgt_slice))
                # Create x labels as negative indices: -15 ... -1 or shorter if less layers
                x_labels = [f"-{num_layers_to_show - i}" for i in range(len(tgt_slice))]

                ax = axs[i // 4, i % 4]

                ax.plot(layers, tgt_slice, label="Correct (XX)", marker='o', color=(130/255, 100/255, 25/255))
                ax.plot(layers, en_slice, label="Wrong (en)", marker='x', color=(76/255, 212/255, 217/255))

                ax.set_title(f"{model_name} ({lang})", fontsize=10)
                ax.grid(True, linestyle=':')

                ax.set_xticks(layers)
                ax.set_xticklabels(x_labels, rotation=45)

                # if i // 4 == 1:  # bottom row: show x label
                #     ax.set_xlabel("Last layer index")

                if i % 4 == 0:  # first col: show y label
                    ax.set_ylabel("Count")
            
            # Add a single legend for the entire figure at the bottom center with two columns
            handles, labels = axs[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))

            plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust rect to leave space for legend
            plt.savefig(f"vis_logitlens/{model}_{dataset}.png", dpi=300, bbox_inches='tight')
            plt.show()
