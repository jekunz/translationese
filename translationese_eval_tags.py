from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import json
import string
import os
import ast
import re
import numpy as np
import gc
import pandas as pd
from collections import Counter


def parse_tags(s):
    """Parse error_tags into a list of strings."""
    return re.findall(r"'(\w+)'", s)


df = pd.read_csv("translationese_opensubtitles_clean.csv")
df["context"] = df["context"].apply(ast.literal_eval)
df["error_tags"] = df["error_tags"].apply(parse_tags)


# --- Prompt construction ---

def make_prompt(english, context_lines, n_context):
    if n_context == 0:
        return f"Översätt följande mening till svenska:\n{english}\nÖversättning: "
    ctx = " ".join(context_lines[-n_context:])
    return (f"Översätt följande mening till svenska med hänsyn till kontexten:\n"
            f"Kontext: {ctx}\nMening: {english}\nÖversättning: ")


def get_texts_for_setup(df, n_context):
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

            triple.append({"mean_nll": mean_nll})

        results.append({"correct": triple[0], "opus": triple[1], "gpt": triple[2]})

    return results


def analyze_error_tags(results, error_tags_list):
    """Count opus-preferred vs human-preferred samples per error tag."""
    opus_preferred = Counter()
    human_preferred = Counter()

    for pair, tags in zip(results, error_tags_list):
        if pair["opus"]["mean_nll"] < pair["correct"]["mean_nll"]:
            opus_preferred.update(tags)
        else:
            human_preferred.update(tags)

    all_tags = sorted(set(opus_preferred.keys()) | set(human_preferred.keys()))
    tag_results = {}
    for tag in all_tags:
        op = opus_preferred[tag]
        hp = human_preferred[tag]
        tag_results[tag] = {
            "opus_preferred": op,
            "human_preferred": hp,
            "total": op + hp,
            "opus_preferred_pct": op / (op + hp) * 100 if (op + hp) > 0 else 0,
        }

    return tag_results


# Only best models and setups ---

models = {
    "AI-Sweden LLaMA base": "AI-Sweden-Models/Llama-3-8B",
    "EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    "Gemma-12B-it": "google/gemma-3-12b-it",
}

# (i) no prompt, (ii) prompt with 10-sentence context
setups = {
    "no_prompt": None,
    "context_10": 10,
}

error_tags_list = df["error_tags"].tolist()
all_results = {}

for model_label, model_id in models.items():
    print("=" * 60)
    print(f"Model: {model_label} ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda").eval()

    chat_template_str = get_chat_template(model_id)
    if chat_template_str is not None:
        chat_template_fn = lambda s: chat_template_str.format(text=s)
    else:
        chat_template_fn = None

    model_results = {}
    for setup_name, n_ctx in setups.items():
        if n_ctx is None:
            print(f"--- {setup_name} (raw sentences) ---")
            human_texts = df["swedish_human"].tolist()
            opus_texts = df["swedish_opus"].tolist()
            gpt_texts = df["swedish_gpt"].tolist()
        else:
            print(f"--- {setup_name} ---")
            human_texts, opus_texts, gpt_texts = get_texts_for_setup(df, n_ctx)

        results = compute_triplet_scores(human_texts, opus_texts, gpt_texts, tokenizer, model, chat_template_fn)
        tag_analysis = analyze_error_tags(results, error_tags_list)
        model_results[setup_name] = tag_analysis

        for tag, stats in sorted(tag_analysis.items()):
            print(f"  {tag}: opus>{stats['opus_preferred']}, human>{stats['human_preferred']}, "
                  f"total={stats['total']}, opus_pct={stats['opus_preferred_pct']:.1f}%")

    all_results[model_label] = model_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

with open("translationese_tag_analysis.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
