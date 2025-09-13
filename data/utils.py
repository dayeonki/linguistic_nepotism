import csv
import json


def csv_to_jsonl(csv_path, jsonl_path):
    with open(csv_path, 'r', encoding='utf-8') as csv_file, open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            json_line = json.dumps(row, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


def remove_duplicates(input_path, output_path):
    seen_ids = set()
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                entry = json.loads(line)
                entry_id = str(entry.get("id"))
                if entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue



if __name__ == "__main__":
    csv_to_jsonl('data.csv', 'data.jsonl')
