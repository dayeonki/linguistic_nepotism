import json


def compute_averages(jsonl_file):
    total_query_words = 0
    total_title_words = 0
    total_context_words = 0
    total_contexts = 0
    num_questions = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            num_questions += 1

            # Query words
            query_words = len(data.get("question_en", "").split())
            total_query_words += query_words

            # Positive context titles and texts
            titles = data.get("pos_context_title_en", [])
            contexts = data.get("pos_context_en", [])

            # Total number of contexts
            num_contexts = len(contexts)
            total_contexts += num_contexts

            # Count words in titles and contexts
            title_word_count = sum(len(title.split()) for title in titles)
            context_word_count = sum(len(context.split()) for context in contexts)

            total_title_words += title_word_count
            total_context_words += context_word_count

    avg_query_words = total_query_words / num_questions if num_questions else 0
    avg_title_words = total_title_words / total_contexts if total_contexts else 0
    avg_context_words = total_context_words / total_contexts if total_contexts else 0
    avg_contexts_per_question = total_contexts / num_questions if num_questions else 0

    print(f"Average # of words (query): {avg_query_words:.2f}")
    print(f"Average # of words (title): {avg_title_words:.2f}")
    print(f"Average # of words (context): {avg_context_words:.2f}")
    print(f"Average # of contexts per question: {avg_contexts_per_question:.2f}")

compute_averages("eli5/en_processed.jsonl")
