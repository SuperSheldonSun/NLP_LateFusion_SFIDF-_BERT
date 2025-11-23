## Personalized News Recommendation on MIND-small

This project implements multiple document representations and recommendation modes on `MIND-small`, together with a unified evaluation toolkit (Hit@K, nDCG@K, MRR). Supported ranking systems:

- **TF-IDF baseline**
- **SF-IDF** (WordNet synsets only)
- **SF-IDF+** (synsets + NER entities)
- **BERT-only** (Sentence-BERT)
- **Late Fusion** (BERT + SF-IDF+)

All ranking outputs share the TSV format:

```text
user_id \t session_id \t doc_id \t score \t label
```

### 1. Environment

- **Python**: 3.9+ recommended  
- Install dependencies:

```bash
pip install numpy scipy scikit-learn sentence-transformers nltk spacy pyyaml
python -m spacy download en_core_web_sm
```

- Download NLTK resources (run once in Python):

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
```

Directory layout assumption:

- Project root: `NLP_Project`
- Data: `data/MINDsmall_train`, `data/MINDsmall_dev`

### 2. Configuration

Main configuration file: `config.yaml`. Key fields:

- **data.root_dir**: project root (default `.`)  
- **paths.cached_dir**: cache directory for docs/sessions (default `cached`)  
- **paths.vectors_dir**: directory for SF-IDF / BERT vectors (default `vectors`)  
- **paths.outputs_dir**: ranking outputs (default `outputs`)  
- **sfidf.alpha**: synset/entity weight for SF-IDF+  
- **fusion.lambda_fusion**: Late Fusion mixing weight λ  
- **eval.k_list**: K values for Hit@K / nDCG@K (e.g., `[5, 10]`)

All scripts honor `--config config.yaml`, and CLI arguments override config values.

### 3. End-to-end pipeline

Assume your current working directory is `NLP_Project`.

#### 3.1 Optional preprocessing & caching

```bash
python scripts/prepare_data.py --config config.yaml
```

Reads `news.tsv` and `behaviors.tsv`, normalizes documents/sessions, and caches them under `cached/` for faster debugging.

#### 3.2 Build SF-IDF / SF-IDF+ vectors

```bash
python scripts/build_sfidf.py --config config.yaml
```

What it does:

- Iterate over every news article in `train + dev`
- Use NLTK + WordNet for synset extraction and spaCy (with fallback) for NER
- Collect corpus-level `df_synset` and `df_entity`
- Build per-document vectors:
  - SF-IDF (synset-only)
  - SF-IDF+ (synset + entity)
- Save vectors to `vectors/sfidf_vectors.pkl`, `vectors/sfidf_entity_vectors.pkl`, `vectors/sfidf_plus_vectors.pkl`

#### 3.3 Build BERT document vectors

```bash
python scripts/build_bert.py --config config.yaml
```

- Use `SentenceTransformer("all-MiniLM-L6-v2")`
- Encode every news article (train + dev)
- L2-normalize and save to `vectors/bert_vectors.pkl`

#### 3.4 Run ranking for different modes

Entry point: `scripts/run_rank.py`. Key arguments:

- `--mode`: `tfidf`, `sfidf`, `sfidf_plus`, `bert`, `late_fusion`
- `--split`: `train` or `dev`

Examples (recommend running on the `dev` split at minimum):

- **TF-IDF baseline**

```bash
python scripts/run_rank.py --config config.yaml --mode tfidf --split dev
```

- **SF-IDF (synset only)**

```bash
python scripts/run_rank.py --config config.yaml --mode sfidf --split dev
```

- **SF-IDF+ (synset + entity)**

```bash
python scripts/run_rank.py --config config.yaml --mode sfidf_plus --split dev
```

- **BERT-only**

```bash
python scripts/run_rank.py --config config.yaml --mode bert --split dev
```

- **Late Fusion (BERT + SF-IDF+)**

```bash
python scripts/run_rank.py --config config.yaml --mode late_fusion --split dev --lambda_fusion 0.5
```

Each run writes a TSV file under `outputs/`, for example:

- `outputs/tfidf_dev_rank.tsv`
- `outputs/sfidf_dev_rank.tsv`
- `outputs/sfidf_plus_dev_rank.tsv`
- `outputs/bert_dev_rank.tsv`
- `outputs/late_fusion_dev_rank.tsv`

Line format: `user_id \t session_id \t doc_id \t score \t label`

### 4. Evaluation (Hit@K / nDCG@K / MRR)

Use `eval.py` to score any ranking file.

Example for the TF-IDF baseline on the `dev` split:

```bash
python eval.py --config config.yaml --input outputs/tfidf_dev_rank.tsv --mode tfidf --output_json metrics_tfidf_dev.json
```

Likewise for other modes:

```bash
python eval.py --config config.yaml --input outputs/sfidf_dev_rank.tsv --mode sfidf --output_json metrics_sfidf_dev.json
python eval.py --config config.yaml --input outputs/sfidf_plus_dev_rank.tsv --mode sfidf_plus --output_json metrics_sfidf_plus_dev.json
python eval.py --config config.yaml --input outputs/bert_dev_rank.tsv --mode bert --output_json metrics_bert_dev.json
python eval.py --config config.yaml --input outputs/late_fusion_dev_rank.tsv --mode late_fusion --output_json metrics_late_fusion_dev.json
```

Console output example:

```text
Results for file: outputs/tfidf_dev_rank.tsv
Mode: tfidf
Hit@5: 0.xxx
Hit@10: 0.xxx
nDCG@5: 0.xxx
nDCG@10: 0.xxx
MRR: 0.xxx
```

The optional JSON file stores the metrics for later comparison.

### 5. Suggested summary table

For the `dev` split, you can summarize results as:

| Mode       | Hit@5 | Hit@10 | nDCG@5 | nDCG@10 | MRR  |
|------------|-------|--------|--------|---------|------|
| TF-IDF     |       |        |        |         |      |
| SF-IDF     |       |        |        |         |      |
| SF-IDF+    |       |        |        |         |      |
| BERT-only  |       |        |        |         |      |
| Late Fusion|       |        |        |         |      |

Populate the numbers using the corresponding `metrics_*.json`.

### 6. Late Fusion grid search (α, λ)

Default parameters are baked in (α / λ = 0.3, 0.5, 0.7; baseline JSONs: `outputs/metrics_tfidf_dev.json`, `outputs/metrics_sfidf_plus_dev.json`; results saved to `outputs/late_fusion_grid/grid_results_dev.json`), so simply run:

```bash
python scripts/grid_search_late_fusion.py
```

To override any setting:

```bash
python scripts/grid_search_late_fusion.py \
  --split dev \
  --alpha_values 0.2 0.4 0.6 \
  --lambda_values 0.4 0.6 0.8 \
  --baseline_metrics custom_metrics.json \
  --results_json my_grid.json
```

- `alpha_values`: synset/entity weighting candidates for SF-IDF+
- `lambda_values`: BERT vs SF-IDF+ weights inside Late Fusion
- Each combination produces `outputs/late_fusion_grid/*.tsv` and immediately runs `eval.evaluate`
- `baseline_metrics` (optional): JSON files generated via `eval.py`; used to compare against Late Fusion in the printed table
- `results_json` (optional): dump the full grid-search results for later analysis



