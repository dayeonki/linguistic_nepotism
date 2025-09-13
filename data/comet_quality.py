import json
from comet import download_model, load_from_checkpoint


model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)


def get_comet_qe_score(src, mt):
    if not isinstance(src, str) or not isinstance(mt, str):
        print(f"[WARN] Skipping non-string input: src={src}, mt={mt}")
        return 0.0
    data = [{"src": src, "mt": mt}]
    score = model.predict(data, batch_size=1, gpus=1)
    return score["system_score"]


def comet_eli5(input_path, output_path):
    all_scores_c, all_scores_t, all_scores_q = [], [], []
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            item = json.loads(line)

            # === Process context titles (list) ===
            titles_en = item.get("pos_context_title_en", [])
            titles_src = item.get("pos_context_title_src", [])
            title_scores = [
                get_comet_qe_score(title_en, title_src)
                for title_en, title_src in zip(titles_en, titles_src)
            ]
            avg_title_score = sum(title_scores) / len(title_scores) if title_scores else 0.0

            # === Process contexts (list) ===
            contexts_en = item.get("pos_context_en", [])
            contexts_src = item.get("pos_context_src", [])
            context_scores = [
                get_comet_qe_score(context_en, context_src)
                for context_en, context_src in zip(contexts_en, contexts_src)
            ]
            avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0.0

            # === Process question (single string) ===
            question_en = item.get("question_en", "")
            question_src = item.get("question_src", "")
            question_score = get_comet_qe_score(question_en, question_src)

            # Store scores
            all_scores_t.append(avg_title_score)
            all_scores_c.append(avg_context_score)
            all_scores_q.append(question_score)

            item["comet_t"] = title_scores
            item["comet_c"] = context_scores
            item["comet_q"] = question_score

            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Context Score: {avg_context_score}")
            print(f"Title Score: {avg_title_score}")
            print(f"Question Score: {question_score}")

    # === Print Final Averages ===
    print("\n========= Average Scores =========")
    print("Context Score: ", sum(all_scores_c) / len(all_scores_c))
    print("Title Score: ", sum(all_scores_t) / len(all_scores_t))
    print("Question Score: ", sum(all_scores_q) / len(all_scores_q))


def comet_miracl(input_path, output_path):
    scores_c, scores_t, scores_q, scores_nt, scores_nc = [], [], [], [], []

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            item = json.loads(line)

            # === Process positive titles (list) ===
            pos_titles_en = item.get("pos_context_title_en", [])
            pos_titles_src = item.get("pos_context_title_src", [])
            pos_title_scores = [
                get_comet_qe_score(en, src)
                for en, src in zip(pos_titles_en, pos_titles_src)
            ]
            avg_pos_title_score = sum(pos_title_scores) / len(pos_title_scores) if pos_title_scores else 0.0

            # === Process positive contexts (list) ===
            pos_contexts_en = item.get("pos_context_en", [])
            pos_contexts_src = item.get("pos_context_src", [])
            pos_context_scores = [
                get_comet_qe_score(en, src)
                for en, src in zip(pos_contexts_en, pos_contexts_src)
            ]
            avg_pos_context_score = sum(pos_context_scores) / len(pos_context_scores) if pos_context_scores else 0.0

            # === Process negative titles (list) ===
            neg_titles_en = item.get("neg_context_title_en", [])
            neg_titles_src = item.get("neg_context_title_src", [])
            neg_title_scores = [
                get_comet_qe_score(en, src)
                for en, src in zip(neg_titles_en, neg_titles_src)
            ]
            avg_neg_title_score = sum(neg_title_scores) / len(neg_title_scores) if neg_title_scores else 0.0

            # === Process negative contexts (list) ===
            neg_contexts_en = item.get("neg_context_en", [])
            neg_contexts_src = item.get("neg_context_src", [])
            neg_context_scores = [
                get_comet_qe_score(en, src)
                for en, src in zip(neg_contexts_en, neg_contexts_src)
            ]
            avg_neg_context_score = sum(neg_context_scores) / len(neg_context_scores) if neg_context_scores else 0.0

            # === Process question (single string) ===
            question_en = item.get("question_en", "")
            question_src = item.get("question_src", "")
            question_score = get_comet_qe_score(question_en, question_src)

            # === Save aggregate scores ===
            scores_t.append(avg_pos_title_score)
            scores_c.append(avg_pos_context_score)
            scores_nt.append(avg_neg_title_score)
            scores_nc.append(avg_neg_context_score)
            scores_q.append(question_score)

            item["comet_t"] = pos_title_scores
            item["comet_c"] = pos_context_scores
            item["comet_q"] = question_score
            item["comet_nt"] = neg_title_scores
            item["comet_nc"] = neg_context_scores

            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Context Score: {avg_pos_context_score}")
            print(f"Title Score: {avg_pos_title_score}")
            print(f"Question Score: {question_score}")
            print(f"Neg Context Score: {avg_neg_context_score}")
            print(f"Neg Title Score: {avg_neg_title_score}")

    # === Print Final Averages ===
    print("\n========= Average Scores =========")
    print("Pos Context Score: ", sum(scores_c) / len(scores_c))
    print("Pos Title Score: ", sum(scores_t) / len(scores_t))
    print("Neg Context Score: ", sum(scores_nc) / len(scores_nc))
    print("Neg Title Score: ", sum(scores_nt) / len(scores_nt))
    print("Question Score: ", sum(scores_q) / len(scores_q))



if __name__ == "__main__":
    languages = ["ar", "bn", "es", "fr", "ru", "ko", "sw", "zh"]

    for language in languages:
        print("="*30, "MIRACL n1", "="*30)
        input_path = f"miracl/{language}_n1.jsonl"
        output_path = f"miracl/{language}_n1_comet.jsonl"
        comet_miracl(input_path, output_path)

        print("="*30, "ELI5", "="*30)
        input_path = f"eli5/{language}.jsonl"
        output_path = f"eli5/{language}_comet.jsonl"
        comet_eli5(input_path, output_path)
