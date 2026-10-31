# Fake News Detection using NLP

A Streamlit educational prototype for classifying labeled news text using TF-IDF and Logistic Regression.

## Important note about the supplied starter data

The supplied `data/dataset.csv` contains 500 rows and these 8 columns:
`article_id, source, word_count, sentiment_score, exclamation_count, all_caps_pct, label, political_lean`.

It does **not** contain article headlines or article text. The app can train a metadata-only baseline on this file, but that is not an NLP model. For NLP training, upload a labeled CSV with a text column such as `news_text`, `text`, `content`, `article`, `headline`, or `title`, plus a label column (for example `FAKE`/`REAL`).

A starter schema is available at `data/text_dataset_template.csv`.

## Features

- Upload up to 20 CSV files, maximum 100 MB each.
- Inspect and combine uploaded labeled datasets.
- Automatically recognize common text column names.
- NLP mode: TF-IDF (unigrams and bigrams) + Logistic Regression.
- Metadata baseline when text is absent.
- Stratified train/test split and accuracy, macro precision, recall, F1, classification report and confusion matrix.
- Single text prediction and batch CSV prediction after training an NLP model.
- Model card with intended use, limitations and human-review safeguards.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Dataset guidance

Use documented, lawfully obtained datasets. Keep source/provenance and original labels. LIAR uses claim-level labels and FakeNewsNet contains article/social-context data; they do not necessarily share identical label definitions or schemas. Map labels only with a documented, defensible rule. Do not invent labels or present synthetic generated rows as verified news.

Do not merge datasets blindly. Deduplicate records, inspect class balance, preserve a `data_origin` field, and check for source/event leakage before reporting model performance. A random split can overestimate generalization if near-duplicate articles or the same publishers occur in both train and test sets.

## Responsible-use model card

**Intended use:** classroom demonstration and exploratory research into patterns in labeled news text.

**Not intended for:** establishing factual truth, automated censorship, political-content enforcement, or decisions about individuals or publishers.

**Limitations:** dataset labels may be noisy, politically or geographically biased, incomplete, outdated, and inconsistent across sources. TF-IDF captures word patterns rather than verifying claims. Performance may degrade on new events, languages, publishers, and formats. Confidence is not a truth probability.

**Safeguards:** require human review; provide context and provenance; audit false positives and false negatives; evaluate performance across sources and topics; never use political leaning as evidence that an item is true or false; do not make consequential decisions solely from model output.

## Project structure

```text
fake_news_detection_nlp/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── data/
│   ├── dataset.csv
│   ├── data_dictionary.csv
│   └── text_dataset_template.csv
├── models/
└── src/
    ├── __init__.py
    └── model_utils.py
```
