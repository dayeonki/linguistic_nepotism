import os
import openai
import tiktoken
from dotenv import load_dotenv


def load_api_key():
    try:
        load_dotenv(dotenv_path='.env')
        openai.api_key = os.environ['OPENAI_API_KEY']
        print("OpenAI API key successfully loaded!")
    except KeyError:
        print("Error: OPENAI_API_KEY environment variable is not set.")
    except openai.error.AuthenticationError:
        print("Error: Incorrect OpenAI API key.")


def count_tokens(model, text):
    # https://github.com/openai/tiktoken/blob/main/tiktoken/model.py#L87
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        if "o3" in model:
            # print(f"[Warning] No tokenizer found for model '{model}', using 'o200k_base' as fallback.")
            encoding = tiktoken.get_encoding("o200k_base")
        elif "4.1" in model:
            # print(f"[Warning] No tokenizer found for model '{model}', using 'gpt-4o' as fallback.")
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif "o4" in model:
            # print(f"[Warning] No tokenizer found for model '{model}', using 'gpt-4o' as fallback.")
            encoding = tiktoken.encoding_for_model("gpt-4o")
        else:
            pass
    token_count = len(encoding.encode(text))
    return token_count


def estimate_cost(model, token_usage):
    model = model.lower()
    if 'gpt-4o' in model:
        _estimate_cost_4o(token_usage)
    elif 'o3' in model or 'o3' in model:
        _estimate_cost_o3(token_usage)
    elif 'gpt-4.1' in model:
        _estimate_cost_41(token_usage)
    elif 'o4' in model:
        _estimate_cost_o4(token_usage)
    else:
        print(f"No cost estimator defined for model: {model}")


def _estimate_cost_41(token_usage):
    input_cost = (token_usage["input_tokens"] / 1_000_000) * 2.00
    cached_input_cost = (token_usage["cached_input_tokens"] / 1_000_000) * 0.5
    output_cost = (token_usage["output_tokens"] / 1_000_000) * 8.00
    total_cost = input_cost + cached_input_cost + output_cost

    print("\n", "="*30, " 💸 Estimated Cost (gpt-4.1) 💸 ", "="*30)
    print(f"1️⃣ Input Cost: ${input_cost:.4f}")
    print(f"2️⃣ Output Cost: ${output_cost:.4f}")
    print(f"💰 Total Cost: ${total_cost:.4f}\n")


def _estimate_cost_4o(token_usage):
    input_cost = (token_usage["input_tokens"] / 1_000_000) * 2.50
    cached_input_cost = (token_usage["cached_input_tokens"] / 1_000_000) * 1.25
    output_cost = (token_usage["output_tokens"] / 1_000_000) * 10.00
    total_cost = input_cost + cached_input_cost + output_cost

    print("\n", "="*30, " 💸 Estimated Cost (gpt-4o) 💸 ", "="*30)
    print(f"1️⃣ Input Cost: ${input_cost:.4f}")
    print(f"2️⃣ Output Cost: ${output_cost:.4f}")
    print(f"💰 Total Cost: ${total_cost:.4f}\n")


def _estimate_cost_o3(token_usage):
    input_cost = (token_usage["input_tokens"] / 1_000_000) * 1.10
    cached_input_cost = (token_usage["cached_input_tokens"] / 1_000_000) * 0.55
    output_cost = (token_usage["output_tokens"] / 1_000_000) * 4.40
    total_cost = input_cost + cached_input_cost + output_cost

    print("\n", "="*30, " 💸 Estimated Cost (gpt-o3) 💸 ", "="*30)
    print(f"1️⃣ Input Cost: ${input_cost:.4f}")
    print(f"2️⃣ Output Cost: ${output_cost:.4f}")
    print(f"💰 Total Cost: ${total_cost:.4f}\n")


def _estimate_cost_o4(token_usage):
    input_cost = (token_usage["input_tokens"] / 1_000_000) * 1.10
    cached_input_cost = (token_usage["cached_input_tokens"] / 1_000_000) * 0.275
    output_cost = (token_usage["output_tokens"] / 1_000_000) * 4.40
    total_cost = input_cost + cached_input_cost + output_cost

    print("\n", "="*30, " 💸 Estimated Cost (gpt-4.1) 💸 ", "="*30)
    print(f"1️⃣ Input Cost: ${input_cost:.4f}")
    print(f"2️⃣ Output Cost: ${output_cost:.4f}")
    print(f"💰 Total Cost: ${total_cost:.4f}\n")
