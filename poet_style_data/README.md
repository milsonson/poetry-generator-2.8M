# Poet Style Data

This folder keeps the poet-conditioned fine-tuning data separate from the
project's original training files.

## Layout

- `raw/chinese-poetry-npm/`: downloaded `chinese-poetry` npm package and extracted JSON data.
- `processed/poet_style_corpus.txt`: simplified Chinese, one training sample per line, formatted as `作者李白。诗句正文`.
- `processed/poet_style_records.jsonl`: structured records with poet, source, title/rhythmic metadata, and cleaned text.
- `processed/per_poet/`: per-poet text files.
- `processed/poet_style_stats.json`: counts by poet and source.
- `processed_vocab_compatible/`: records that can be encoded by the current root `vocab.json`.
- `training_full/`: train/validation tensors and a new vocab built from the full processed style corpus.

## Regenerate

```bash
poet_style_data/tools/opencc_venv/bin/python build_poet_style_data.py
.venv/bin/python prepare_data.py --local-txt poet_style_data/processed/poet_style_corpus.txt --sample-size 0 --output-dir poet_style_data/training_full
```

To inspect what can be encoded by the current checkpoint vocabulary:

```bash
poet_style_data/tools/opencc_venv/bin/python build_poet_style_data.py --output-dir poet_style_data/processed_vocab_compatible --vocab vocab.json
```
