from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import json
import string
import os
import numpy as np
import gc
import pandas as pd

df = pd.read_csv("translationese_opensubtitles_clean.csv")
df["context"] = df["context"].apply(json.loads)

# --- Prompt construction ---

def make_prompt(english, context_lines, n_context):
    """Build a translation prompt with N preceding context lines."""
    if n_context == 0:
        return f"Översätt följande mening till svenska:\n{english}\nÖversättning: "
    ctx = " ".join(context_lines[-n_context:])
    return (f"Översätt följande mening till svenska med hänsyn till kontexten:\n"
            f"Kontext: {ctx}\nMening: {english}\nÖversättning: ")


def get_texts_for_setup(df, n_context):
    """Return (human, opus, gpt) text tuples for a given context level."""
    human_texts, opus_texts, gpt_texts = [], [], []
    for _, row in df.iterrows():
        prompt = make_prompt(row["english_source"], row["context"], n_context)
        human_texts.append(prompt + row["swedish_human"])
        opus_texts.append(prompt + row["swedish_opus"])
        gpt_texts.append(prompt + row["swedish_gpt"])
    return human_texts, opus_texts, gpt_texts


# --- Model utilities ---

def get_chat_template(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        return None

    model_dir = config._name_or_path if os.path.isdir(config._name_or_path) else None
    possible_files = ["generation_config.json", "config.json", "tokenizer_config.json"]

    chat_template = None
    if model_dir is not None:
        for fname in possible_files:
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for key in ["prompt_template", "system_prompt", "chat_template"]:
                    if key in data:
                        chat_template = data[key]
                        break
            if chat_template:
                break
    return chat_template


def strip_trailing_punct(text):
    return text.rstrip(string.punctuation + " ")


# --- Scoring ---

def compute_triplet_scores(texts_a, texts_b, texts_c, tokenizer, model, chat_template=None):
    """Compute NLL and perplexity for triplets of (human, opus, gpt) sentences."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for a, b, c in zip(texts_a, texts_b, texts_c):
        triple = []
        for text in (a, b, c):
            text = text.strip()
            stripped_text = strip_trailing_punct(text).strip()
            formatted_text = chat_template(stripped_text) if chat_template is not None else stripped_text

            encoding = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding["input_ids"].to(model.device)
            attention_mask = encoding["attention_mask"].to(model.device)

            if chat_template is not None:
                sentence_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
                start_idx = (input_ids[0] == sentence_ids[0][0]).nonzero(as_tuple=True)[0].item()
                end_idx = start_idx + sentence_ids.size(1)
            else:
                start_idx, end_idx = 0, input_ids.size(1)

            labels = input_ids.clone()
            labels[:start_idx] = -100
            labels[end_idx:] = -100
            labels[input_ids == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_nll = loss.item() * num_tokens
            mean_nll = total_nll / num_tokens
            ppl = torch.exp(torch.tensor(mean_nll)).item()

            triple.append({
                "sentence": text,
                "formatted_text": formatted_text,
                "total_nll": total_nll,
                "mean_nll": mean_nll,
                "ppl": ppl,
            })

        results.append({"correct": triple[0], "opus": triple[1], "gpt": triple[2]})

    return results


def evaluate_results(results):
    human_nlls, opus_nlls, gpt_nlls = [], [], []
    correct_guesses = 0
    correct_guesses_gpt = 0

    for pair in results:
        human_nlls.append(pair["correct"]["mean_nll"])
        opus_nlls.append(pair["opus"]["mean_nll"])
        gpt_nlls.append(pair["gpt"]["mean_nll"])

        if pair["correct"]["mean_nll"] < pair["opus"]["mean_nll"]:
            correct_guesses += 1
        if pair["correct"]["mean_nll"] < pair["gpt"]["mean_nll"]:
            correct_guesses_gpt += 1

    acc = correct_guesses / len(results)
    acc_gpt = correct_guesses_gpt / len(results)

    human_nlls = np.array(human_nlls)
    opus_nlls = np.array(opus_nlls)
    gpt_nlls = np.array(gpt_nlls)
    delta_nlls = np.mean((opus_nlls - human_nlls) / human_nlls)
    delta_nlls_gpt = np.mean((gpt_nlls - human_nlls) / human_nlls)

    print(f"Accuracy vs OPUS: {acc:.4f}")
    print(f"Relative ΔNLL (OPUS vs Human): {delta_nlls*100:.2f}%")
    print(f"Accuracy vs GPT: {acc_gpt:.4f}")
    print(f"Relative ΔNLL (GPT vs Human): {delta_nlls_gpt*100:.2f}%")
    print(f'{acc:.4f} & {delta_nlls*100:.2f} & {acc_gpt:.4f} & {delta_nlls_gpt*100:.2f}')

    return {
        "accuracy opus": acc,
        "delta nll opus": delta_nlls,
        "accuracy gpt": acc_gpt,
        "delta nll gpt": delta_nlls_gpt,
    }


# --- Models and evaluation ---

models = [

    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'AI-Sweden-Models/Llama-3-8B',
    'AI-Sweden-Models/Llama-3-8B-instruct',

    'utter-project/EuroLLM-1.7B',
    'utter-project/EuroLLM-1.7B-Instruct',
    'utter-project/EuroLLM-9B',
    'utter-project/EuroLLM-9B-Instruct',

    'google/gemma-3-270m',
    'google/gemma-3-270m-it',
    'google/gemma-3-1b-pt',
    'google/gemma-3-1b-it',
    'google/gemma-3-4b-pt',
    'google/gemma-3-4b-it',
    'google/gemma-3-12b-pt',
    'google/gemma-3-12b-it',
]

# None = raw sentences (no prompt), 0-5 = context lines, 10 = full context
context_levels = [None, 0, 1, 2, 3, 4, 5, 10]

all_results = {}

for m in models:
    print("=" * 60)
    print("Model:", m)
    tokenizer = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True).to("cuda").eval()

    chat_template_str = get_chat_template(m)
    if chat_template_str is not None:
        chat_template_fn = lambda s: chat_template_str.format(text=s)
    else:
        chat_template_fn = None

    model_results = {}
    for n_ctx in context_levels:
        setup_name = "no_prompt" if n_ctx is None else f"context_{n_ctx}"
        if n_ctx is None:
            print("--- No prompt (raw sentences) ---")
            human_texts = df["swedish_human"].tolist()
            opus_texts = df["swedish_opus"].tolist()
            gpt_texts = df["swedish_gpt"].tolist()
        else:
            print(f"--- context={n_ctx} ---")
            human_texts, opus_texts, gpt_texts = get_texts_for_setup(df, n_ctx)

        results = compute_triplet_scores(human_texts, opus_texts, gpt_texts, tokenizer, model, chat_template_fn)
        model_results[setup_name] = evaluate_results(results)

    all_results[m] = model_results

    del model
    gc.collect()
    torch.cuda.empty_cache()


with open("translationese_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, default=str)
