
# A Dataset for Probing Translationese Preferences in English-to-Swedish Translation

Intended for modelling implicit preferences for idiomatic language in a minimal-pair setup. 

## Dataset Description

This dataset contains 600 English sentences from [OpenSubtitles](https://aclanthology.org/L16-1147.pdf) paired with three Swedish translations of varying quality:

| Column | Description |
|---|---|
| `english_source` | Original English sentence |
| `swedish_opus` | Machine translation by [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-sv) |
| `swedish_human` | Human-corrected translation, revised to be more natural and idiomatic |
| `swedish_gpt` | Machine translation by GPT-4 |
| `error_tags` | Error categories for the OPUS translation (see below) |
| `gpt_acceptable` | Whether the GPT translation was judged equally acceptable as the human one |
| `context` | 10 preceding subtitle lines from the English source |

### Error Tags

Each sample is annotated with one or more error tags categorizing the translationese present in the OPUS translation. See our paper (coming soon) for detailed descriptions.

## Files

| File | Description |
|---|---|
| `translationese_opensubtitles_clean.csv` | Release version of the dataset. Also available [on HuggingFace](https://huggingface.co/datasets/liu-nlp/translationese-opensubtitles). |
| `translationese_opensubtitles_annotated.csv` | Extended version with detailed error descriptions and comments by the annotators. |
| `translationese_eval_csv.py` | Minimal-pair evaluation script for the LREC paper. |

## Citation

Coming soon. 
