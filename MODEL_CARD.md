# Model Card: Fake News Detection NLP Prototype

- **Model:** TF-IDF + Logistic Regression
- **Task:** Supervised classification of labeled news text
- **Input:** Headline/article text and dataset-provided label
- **Output:** One of the classes learned from the training data
- **Training:** User-supplied, labeled CSV; stratified holdout evaluation
- **Intended use:** Education and exploratory research only

## Critical limitations
This model does not verify factual claims. It learns associations between text patterns and training labels. Labels from LIAR, FakeNewsNet, and other sources have different definitions and units of analysis; they must not be treated as interchangeable without careful mapping. The supplied 500-row starter CSV contains metadata only, so training on it uses a metadata baseline, not NLP.

The model may reflect source, topic, time-period, language, and political biases in the data. Confidence scores are not calibrated probabilities that a story is true.

## Safeguards
- Require a human to review any prediction.
- Do not use for censorship or automatic political-content labeling.
- Do not treat political leaning as evidence of truth.
- Record dataset origin and label mapping.
- Report class-wise metrics and investigate false positives/negatives.
- Test on temporally and source-separated data where possible.
