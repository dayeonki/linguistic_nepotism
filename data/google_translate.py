import json
import time
import os
from deep_translator import GoogleTranslator

MAX_CHARS = 2000

def split_text_preserving_words(text, max_len=MAX_CHARS):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_len:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def safe_translate(text, translator):
    if len(text.strip()) == 0:
        return ""
    if len(text) <= MAX_CHARS:
        return translator.translate(text)

    chunks = split_text_preserving_words(text)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)


def translate_miracl(input_path, output_path, source_lang='en', target_lang=None):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
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
        print(f"Loaded {len(processed_ids)} already processed items. Skipping those.")

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if data["id"] in processed_ids:
                continue
            
            try:
                # Translate question
                q = data.get('question_en', '').strip()
                data['question_src'] = safe_translate(q, translator)
                print(f"question → {data['question_src'][:60]}...")
                time.sleep(1)

                # Translate titles - positive
                pos_titles = data.get("pos_context_title_en", [])
                title_list = [safe_translate(title.strip(), translator) for title in pos_titles]
                data['pos_context_title_src'] = title_list
                print(f"title[0] → {title_list[0][:60]}..." if title_list else "no title")
                time.sleep(1)

                # Translate contexts - positive
                pos_contexts = data.get("pos_context_en", [])
                context_list = [safe_translate(ctx.strip(), translator) for ctx in pos_contexts]
                data['pos_context_src'] = context_list
                print(f"context[0] → {context_list[0][:60]}..." if context_list else "no context")
                time.sleep(1)

                # Translate titles - negative
                neg_titles = data.get("neg_context_title_en", [])
                title_list = [safe_translate(title.strip(), translator) for title in neg_titles]
                data['neg_context_title_src'] = title_list
                print(f"title[0] → {title_list[0][:60]}..." if title_list else "no title")
                time.sleep(1)

                # Translate contexts - negative
                neg_contexts = data.get("neg_context_en", [])
                context_list = [safe_translate(ctx.strip(), translator) for ctx in neg_contexts]
                data['neg_context_src'] = context_list
                print(f"context[0] → {context_list[0][:60]}..." if context_list else "no context")
                time.sleep(1)
            except:
                data['question_src'] = "NOT TRANSLATED"
                data['pos_context_title_src'] = "NOT TRANSLATED"
                data['pos_context_src'] = "NOT TRANSLATED"
                data['neg_context_title_src'] = "NOT TRANSLATED"
                data['neg_context_src'] = "NOT TRANSLATED"

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


def translate_eli5(input_path, output_path, source_lang='en', target_lang=None):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
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
        print(f"Loaded {len(processed_ids)} already processed items. Skipping those.")
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if data["id"] in processed_ids:
                continue
            
            try:
                # Translate question
                q = data.get('question_en', '').strip()
                data['question_src'] = safe_translate(q, translator)
                print(f"question → {data['question_src'][:60]}...")

                # Translate titles - positive
                pos_titles = data.get("pos_context_title_en", [])
                title_list = [safe_translate(title.strip(), translator) for title in pos_titles]
                data['pos_context_title_src'] = title_list
                print(f"title[0] → {title_list[0][:60]}..." if title_list else "no title")

                # Translate contexts - positive
                pos_contexts = data.get("pos_context_en", [])
                context_list = [safe_translate(ctx.strip(), translator) for ctx in pos_contexts]
                data['pos_context_src'] = context_list
                print(f"context[0] → {context_list[0][:60]}..." if context_list else "no context")
            except:
                data['question_src'] = "NOT TRANSLATED"
                data['pos_context_title_src'] = "NOT TRANSLATED"
                data['pos_context_src'] = "NOT TRANSLATED"

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # ELI5
    languages = ["ar", "bn", "es", "fr", "ru", "ko", "sw", "zh"]
    for language in languages:
        input_path = f"eli5/en_processed.jsonl"
        output_path = f"eli5/{language}.jsonl"
        if language == "zh":
            language = "zh-CN"
        else: pass
        translate_eli5(input_path=input_path, output_path=output_path, target_lang=language)

    # MIRACL n1
    languages = ["ko"]
    for language in languages:
        input_path = f"miracl/en_n1.jsonl"
        output_path = f"miracl/{language}_n1.jsonl"
        translate_miracl(input_path=input_path, output_path=output_path, target_lang=language)
