#!/usr/bin/env python3
import json


for lang in ["fr"]:
    INPUT_FILE = f"multi_token/{lang}.jsonl"
    OUTPUT_FILE = f"single_token/{lang}.jsonl"
    MAX_LENGTH = 9

    filtered_count = 0
    total_count = 0
    filtered_out_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                total_count += 1
                
                # Check if pos_context_id exists and is a list
                if 'pos_context_id' in data and isinstance(data['pos_context_id'], list):
                    # Check if length is <= MAX_LENGTH
                    if len(data['pos_context_id']) <= MAX_LENGTH:
                        outfile.write(line + '\n')
                        filtered_count += 1
                    else:
                        filtered_out_count += 1
                        print(f"Filtered out row {line_num}: pos_context_id length = {len(data['pos_context_id'])} (max allowed: {MAX_LENGTH})")
                else:
                    filtered_out_count += 1
                    print(f"Filtered out row {line_num}: missing or invalid pos_context_id")
                        
            except json.JSONDecodeError:
                print(f"Filtered out row {line_num}: JSON decode error")
                continue
    
    print(f"Total rows processed: {total_count}")
    print(f"Rows kept (pos_context_id length <= {MAX_LENGTH}): {filtered_count}")
    print(f"Rows filtered out: {filtered_out_count}")
    print(f"Output saved to: {OUTPUT_FILE}")
