from pathlib import Path
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.model_utils import (
    MAX_FILES, MAX_FILE_BYTES, combine_uploaded_csvs, normalize_labels,
    detect_text_columns, prepare_training_frame, train_classifier,
    evaluate_classifier, predict_texts, label_to_display
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"

st.set_page_config(page_title="Fake News Detection | NLP", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection using NLP")
st.caption("An educational machine-learning prototype for analyzing text patterns—not a fact-checking authority.")

with st.sidebar:
    st.header("Project controls")
    st.info("Upload up to 20 CSV files. Each file can be at most 100 MB.")
    st.caption("Supported training format: CSV with a label column and preferably a headline/article text column.")

tab_overview, tab_train, tab_predict, tab_batch, tab_card = st.tabs(
    ["Overview", "Dataset & Training", "Single Prediction", "Batch Prediction", "Model Card"]
)

def read_csv_uploads(uploaded_files):
    if not uploaded_files:
        return None
    for f in uploaded_files:
        if f.size > MAX_FILE_BYTES:
            raise ValueError(f"{f.name} exceeds the 100 MB per-file limit.")
        if not f.name.lower().endswith(".csv"):
            raise ValueError(f"{f.name}: only CSV files are supported for training.")
    return combine_uploaded_csvs(uploaded_files)

def get_current_model():
    model = st.session_state.get("trained_model")
    info = st.session_state.get("model_info")
    if model is not None:
        return model, info
    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            return saved["model"], saved.get("info", {})
        except Exception:
            return None, None
    return None, None

with tab_overview:
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current bundled records", "500")
    c2.metric("Bundled columns", "8")
    c3.metric("Upload limit", "20 files")
    st.markdown("""
    **Workflow:** upload a labeled dataset → inspect columns → train a model → evaluate it on a held-out test set → analyze text or batch CSV files.

    The supplied starter dataset contains metadata fields (`word_count`, `sentiment_score`,
    `exclamation_count`, `all_caps_pct`, etc.) but **does not contain article text**.
    The app therefore supports two modes:
    - **NLP mode:** when a text column such as `news_text`, `text`, `content`, `article`, `headline`, or `title` exists, TF-IDF features are used.
    - **Metadata baseline:** when no text column exists, a tabular baseline is trained. This is not an NLP model and cannot learn from article wording.
    """)
    if DEFAULT_DATA.exists():
        starter = pd.read_csv(DEFAULT_DATA)
        st.write("Bundled dataset preview")
        st.dataframe(starter.head(10), use_container_width=True)
        st.write("Label distribution")
        if "label" in starter.columns:
            st.bar_chart(starter["label"].astype(str).value_counts())

with tab_train:
    st.subheader("1. Load a labeled dataset")
    uploaded = st.file_uploader(
        "Upload CSV dataset(s)", type=["csv"], accept_multiple_files=True,
        help="Maximum 20 files; 100 MB per file."
    )
    if uploaded and len(uploaded) > MAX_FILES:
        st.error(f"Please upload no more than {MAX_FILES} files.")
        data = None
    else:
        try:
            data = read_csv_uploads(uploaded) if uploaded else pd.read_csv(DEFAULT_DATA)
        except Exception as exc:
            st.error(str(exc))
            data = None

    if data is not None:
        st.session_state["active_data"] = data
        st.success(f"Loaded {len(data):,} rows and {len(data.columns)} columns.")
        st.dataframe(data.head(8), use_container_width=True)
        label_options = list(data.columns)
        default_label_index = label_options.index("label") if "label" in label_options else 0
        label_col = st.selectbox("Select target/label column", label_options, index=default_label_index)
        text_candidates = detect_text_columns(data, exclude=[label_col])
        text_choice = st.selectbox(
            "Text column (choose 'None' for metadata-only baseline)",
            ["None"] + text_candidates,
            index=1 if text_candidates else 0
        )
        st.caption("For NLP, select a column containing actual headlines or article text. Do not select article_id or metadata.")
        test_size = st.slider("Test set percentage", min_value=10, max_value=40, value=20, step=5)
        if st.button("Train and evaluate model", type="primary"):
            with st.spinner("Preparing data and training model..."):
                try:
                    frame, labels, mode, used_features = prepare_training_frame(
                        data, label_col, None if text_choice == "None" else text_choice
                    )
                    model, split, metrics, report, cm = train_classifier(
                        frame, labels, mode, test_size / 100
                    )
                    st.session_state["trained_model"] = model
                    st.session_state["model_info"] = {
                        "mode": mode, "text_column": text_choice if text_choice != "None" else None,
                        "label_column": label_col, "features": used_features,
                        "rows_used": int(len(labels)), "metrics": metrics
                    }
                    joblib.dump({"model": model, "info": st.session_state["model_info"]}, MODEL_PATH)
                    st.session_state["evaluation"] = (report, cm, list(split))
                    st.success(f"Training complete. Mode: {mode}. Model saved to models/fake_news_model.joblib.")
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    if "model_info" in st.session_state:
        info = st.session_state["model_info"]
        st.divider()
        st.subheader("2. Evaluation")
        if info["mode"] == "NLP":
            st.success("NLP mode: TF-IDF text features were included.")
        else:
            st.warning("Metadata-only baseline: no article text was available. Add a text column to train a true NLP model.")
        cols = st.columns(4)
        for col, key in zip(cols, ["accuracy", "precision_macro", "recall_macro", "f1_macro"]):
            col.metric(key.replace("_", " ").title(), f"{info['metrics'].get(key, 0):.3f}")
        if "evaluation" in st.session_state:
            report, cm, class_names = st.session_state["evaluation"]
            st.text("Classification report\n" + report)
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_title("Confusion matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
            ax.set_yticks(range(len(class_names)), class_names)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            st.pyplot(fig)
            plt.close(fig)

with tab_predict:
    st.subheader("Analyze a news item")
    model, info = get_current_model()
    if model is None:
        st.info("Train a model in the Dataset & Training tab first.")
    else:
        if info.get("mode") != "NLP":
            st.warning("The currently loaded model is metadata-only. Train an NLP model with a real text column to enable text prediction.")
        else:
            text = st.text_area("Paste a headline or article text", height=180, placeholder="Paste the news text here...")
            if st.button("Analyze text", type="primary", disabled=not text.strip()):
                pred, confidence = predict_texts(model, pd.DataFrame({"__text__": [text]}))
                st.metric("Model output", label_to_display(pred[0]))
                st.metric("Model confidence", f"{confidence[0]:.1%}" if confidence is not None else "Unavailable")
                st.caption("Confidence is the model's estimated probability, not the probability that the claim is true.")

with tab_batch:
    st.subheader("Batch prediction")
    st.write("Upload a CSV containing a text column. The model adds `predicted_label` and, when available, `model_confidence`.")
    batch_file = st.file_uploader("Choose batch CSV", type=["csv"], key="batch_file")
    model, info = get_current_model()
    if batch_file:
        if batch_file.size > MAX_FILE_BYTES:
            st.error("File exceeds 100 MB.")
        elif model is None or info.get("mode") != "NLP":
            st.warning("Train an NLP model before batch text prediction.")
        else:
            batch = pd.read_csv(batch_file)
            text_cols = detect_text_columns(batch)
            if not text_cols:
                st.error("No recognized text column found. Expected one of: news_text, text, content, article, headline, title.")
            else:
                selected = st.selectbox("Text column", text_cols)
                if st.button("Predict batch"):
                    preds, conf = predict_texts(model, pd.DataFrame({"__text__": batch[selected].fillna("").astype(str)}))
                    batch["predicted_label"] = preds
                    if conf is not None:
                        batch["model_confidence"] = conf
                    st.dataframe(batch.head(30), use_container_width=True)
                    st.download_button("Download predictions CSV", batch.to_csv(index=False).encode("utf-8"), "news_predictions.csv", "text/csv")

with tab_card:
    st.subheader("Model card and responsible use")
    st.markdown("""
    **Intended use:** educational experimentation and research into text classification patterns.

    **Not intended for:** determining truth, censoring content, making decisions about people,
    ranking political viewpoints, or automatically labeling specific political content as false.

    **Limitations**
    - Dataset labels can be incomplete, biased, inconsistent, or outdated.
    - A classifier learns correlations in its training data, not factual verification.
    - Performance can fall on new events, regions, languages, publishers, and writing styles.
    - Political-lean metadata must not be treated as evidence of truth or falsehood.
    - Confidence scores are not calibrated truth probabilities.

    **Safeguards:** show predictions as model assessments, retain source/provenance information,
    evaluate errors across groups and sources, and require independent human review before
    any consequential use. Never use this model as the sole basis for moderation or publication decisions.
    """)
