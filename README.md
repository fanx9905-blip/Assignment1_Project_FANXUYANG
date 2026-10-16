# Assignment1_Project_FANXUYANG

**Language Models, Embeddings, and Representations**  
**Name: FAN XUYANG · Student ID: A0351187B**

## Start here

1. **[Report](output/pdf/DSA4213_Assignment1_Report_FANXUYANG.pdf)** — main findings, comparisons and limitations (6 main pages + 1 source page).
2. **[Experiment notebook](Assignment_code.ipynb)** — all model definitions, experiments and saved outputs for Parts I–III.
3. **Reproduction instructions** — environment, data processing, configurations and seeds are provided below.

Part I compares n-gram, RNN, LSTM and causal Transformer models on *Pride and Prejudice*. Part II compares trainable and frozen LSTM input embeddings. Part III evaluates frozen representations on sentiment classification. Random-frozen inputs, TF-IDF and paired statistics provide exploratory supplementary comparisons.

## Repository guide

```text
Assignment_code.ipynb       Main experiment source and saved outputs
output/pdf/                PDF report only
artifacts/                 Saved models, metrics, predictions and figures
scripts/                   Data download and result verification
downstream_data/            Sentiment dataset and source metadata
external_embeddings/       GloVe source metadata and local vector cache
requirements*.txt          Experiment dependencies
```

The original novel is included as `pride_and_prejudice_gutenberg_1342.txt`. Start with the report and notebook; reproduction instructions are provided below.

## Setup and verification

Recorded environment: Python 3.11.15, PyTorch 2.14.0+cpu, Windows, four PyTorch CPU threads. Run commands from the repository root, using an environment with `requirements.txt` installed:

```bash
python -m pip install -r requirements.txt
python scripts/download_part2_glove.py
python scripts/download_part3_sentiment.py
python scripts/verify_part1.py
python scripts/verify_part2.py
python scripts/verify_part3.py
python scripts/verify_supplementary_experiments.py
```

Downloads prepare missing external inputs. The four verification scripts restore saved models and reconstruct results without retraining or replacing evidence. All four passed after script relocation. Launch the notebook with `python -m notebook Assignment_code.ipynb`, or open it in VS Code with the same Python environment.

**Run All retrains models and overwrites artifacts.** The supplementary random-frozen model uses a cached run when available; fresh predictions can invalidate supplementary-plan hashes. The completed 51-cell sequential audit restored LM/Skip-gram checkpoints, so it does not certify a fresh end-to-end training run. Preserve submitted results before retraining.

For an isolated sequential replay, run `python scripts/audit_notebook_execution.py`. Its copied workspace and audit record are written under `.execution_audit/`, which is ignored by Git.

## Data processing and experiment settings

The included novel has SHA-256 `a5666f87abf2cbfdaa27ea8c73bd284da9649b9a2ab27b4e6c8f6aeab1bd1c88`. Preserve its source/license notice when sharing. Preprocessing removes front matter, illustrations and chapter headings, then lowercases and regex-tokenizes words and punctuation. Full paragraphs are split chronologically at approximately 80/10/10% of tokens. The training-only vocabulary (minimum frequency 2) contains 3,620 entries, including PAD=0, UNK=1, BOS=2, EOS=3. Inputs are BOS+body; targets are body+EOS, excluding padding from CE. PPL=exp(CE). No neural paragraph truncation is used. Train/validation/test targets total 118,401 / 14,649 / 14,811.

UCI data are parsed on physical LF because IMDb contains sentence-internal NEL characters. Deduplication uses original tokens before UNK mapping. Removing 21 duplicates leaves 2,979 sentences; seed-42 source+label stratification gives 1,787 / 596 / 596 train/validation/test rows. Original token keys are disjoint. Encoded collisions and approximately 21% downstream UNK remain limitations.

| Setting | Configuration |
|---|---|
| n-gram | Smoothed unigram/bigram/trigram; nine validation candidates; selected alpha=0.001, weights 0.3/0.5/0.2 |
| Neural training | Seed 42, Adam 0.003, batch 8, gradient clip 1, max 8 epochs, patience 3; checkpoint selected by minimum validation CE |
| RNN/LSTM | One layer, embedding 100, hidden 128, dropout 0.2; packed full paragraphs |
| Transformer | Embedding 100 projected to 128; one pre-LN causal block, 4 heads, FFN 256, sinusoidal positions; attention dropout 0, other dropout 0.2 |
| Skip-gram | Training paragraphs only; 100d, window 5, 5 replacement negatives from frequency^0.75, 5 epochs, batch 1024, Adam 0.003, seed 42 |
| Frozen LSTM inputs | Entire 362,000-parameter input table frozen, including special/fallback rows; 584,740 other trainable parameters; seeded N(0,1) fallback, PAD zero |
| Sentiment | Frozen encoders, mean 128d body states excluding BOS/EOS/PAD; StandardScaler + logistic regression, L-BFGS/max_iter 2000, seed 42; C={0.1,1,10} by validation macro-F1, smaller C on ties; refit train+validation |
| TF-IDF | Original downstream tokens, 1–2 grams, min_df=2, max_features=50000, sublinear TF, L2; same C grid/splits, selected C=10; vectorizer refitted with classifier on train+validation |
| Generation | elizabeth was / mr darcy / she could not; seeds 42/43/44; 50 new tokens, temperature 0.8, top-k 50; PAD/BOS/EOS blocked, UNK allowed |
| Statistics | 10,000 paired sentence bootstrap resamples, seed 4213; exact McNemar, Holm adjustment across six pairs |

Inference uses `artifacts/inference_workload.json`: 128 test prefixes, up to 32 history tokens, batch 1, full vocabulary, eight warm-ups, median of three repeats, no recurrent/KV caching. Tensor construction is included, checkpoint loading excluded. Measurements span sessions and are not generation throughput.

## Saved evidence

| File type | Contents and purpose |
|---|---|
| `.json` | Structured text containing configurations, splits, metrics, histories, predictions, generation samples or source/checksum records; readable by both Python and people |
| `.csv` | Summary tables for spreadsheets |
| `.pt` | PyTorch model weights or embedding tensors used to restore trained models |
| `.npz` | NumPy arrays containing features or numeric classifier state |
| `.png` | Experiment plots |

JSON files in `artifacts/` are experimental evidence and some are runtime inputs; retain them. For example, `part1_comparison.json` supports the model comparison, `part3_test_predictions.json` records sentence-level predictions, and `part3_uncertainty.json` records statistical results.

## Results at a glance

| Comparison | Test result |
|---|---|
| Part I perplexity | n-gram 109.20; RNN 83.28; LSTM 75.66; Transformer 96.34 |
| Frozen-input LSTM perplexity | Skip-gram 83.81; GloVe 83.11; random-frozen 85.90 |
| Sentiment macro-F1 | Random/trainable LM 0.6325; Skip-gram 0.6357; GloVe 0.6861; TF-IDF 0.8254 |

These are single-seed observations. The supplementary experiments were designed after inspecting original test results. The report discusses domain shift, OOV, unequal model/input settings, timing limits and uncertainty.

## Sources and tool assistance

Data: Jane Austen, [Project Gutenberg ebook 1342](https://www.gutenberg.org/ebooks/1342); Kotzias (2015), [UCI Sentiment Labelled Sentences](https://doi.org/10.24432/C57604), CC BY 4.0. Public vectors: [Stanford GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) through [gensim-data](https://github.com/piskvorky/gensim-data); source/license/checksum metadata accompany the downloads. Skip-gram is trained locally with PyTorch. Libraries include PyTorch, NumPy, Matplotlib, scikit-learn, SciPy, nbformat and nbclient; optional PDF tools are ReportLab/PyMuPDF.

OpenAI Codex assisted with code and PDF preparation. I am responsible for the experimental design and conclusions and have checked them.

The large GloVe gzip is downloaded separately; saved project checkpoints are retained for verification.
