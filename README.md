
# A Dataset for Probing Translationese Preferences in English-to-Swedish Translation

Intended for modelling implicit preferences for idiomatic language in a minimal-pair setup. 

## Dataset Description

This dataset contains 600 English sentences from English [OpenSubtitles](https://aclanthology.org/L16-1147.pdf) paired with three Swedish translations of varying quality: 
- swedish_opus is a translation by [OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-sv).
- swedish_gpt is a translation by GPT-4.
- swedish_human is a translation written by a human annotator, with the intention to explicitly make the OPUS translation more natural and idiomatic. 

Each sample also includes:
- 10 sentences context from the source
- A human annotator's judgment whether the GPT translation is equally acceptable as the human translation.
- Error tags for the OPUS translation. See our paper (coming soon) for a detailed description of the tags.

All sentences are annotated with error tags categorizing the type of translationese present in the OPUS translation.

## Files

- 'translationese_opensubtitles_clean.csv': Release version of the dataset. Also available [on HuggingFace](https://huggingface.co/datasets/liu-nlp/translationese-opensubtitles). 
- 'translationese_opensubtitles_annotated.csv': Also includes detailed error descriptions by the annotators.
- 'translationese_eval.py': Minimal pair evaluations of models for the LREC paper. 

## Citation

Coming soon. 
