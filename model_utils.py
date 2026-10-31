import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

MAX_FILES = 20
MAX_FILE_BYTES = 100 * 1024 * 1024

TEXT_NAMES = ["news_text", "article_text", "text", "content", "article", "headline", "title", "news"]

def combine_uploaded_csvs(files):
    frames = []
    for f in files:
        f.seek(0)
        frames.append(pd.read_csv(f))
    if not frames:
        raise ValueError("No CSV files were supplied.")
    return pd.concat(frames, ignore_index=True, sort=False)

def normalize_labels(series):
    raw = series.astype(str).str.strip().str.lower()
    mapping = {
        "fake": "FAKE", "false": "FAKE", "0": "FAKE", "1": "REAL",
        "real": "REAL", "true": "REAL", "genuine": "REAL", "mostly-true": "REAL",
        "pants-fire": "FAKE", "barely-true": "FAKE", "half-true": "REAL"
    }
    return raw.map(lambda x: mapping.get(x, x.upper()))

def detect_text_columns(df, exclude=None):
    exclude = set(exclude or [])
    lower_map = {str(c).lower(): c for c in df.columns}
    found = []
    for name in TEXT_NAMES:
        if name in lower_map and lower_map[name] not in exclude:
            found.append(lower_map[name])
    return found

def prepare_training_frame(df, label_col, text_col=None):
    if label_col not in df:
        raise ValueError("Selected label column does not exist.")
    y = normalize_labels(df[label_col])
    valid = y.notna() & ~y.isin(["", "NAN", "NONE"])
    work = df.loc[valid].copy()
    y = y.loc[valid]
    if y.nunique() < 2:
        raise ValueError("At least two distinct label classes are required.")
    if text_col:
        if text_col not in work:
            raise ValueError("Selected text column does not exist.")
        X = pd.DataFrame({"__text__": work[text_col].fillna("").astype(str)})
        X = X[X["__text__"].str.strip().ne("")]
        y = y.loc[X.index]
        if y.nunique() < 2:
            raise ValueError("After removing empty text, at least two classes must remain.")
        return X, y, "NLP", [text_col]
    excluded = {label_col}
    excluded.update(c for c in work.columns if str(c).lower() in {"article_id", "id", "timestamp", "date"})
    X = work.drop(columns=list(excluded), errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found.")
    # Drop columns that are entirely empty or unique identifiers.
    X = X.dropna(axis=1, how="all")
    for col in list(X.columns):
        if X[col].nunique(dropna=True) <= 1:
            X = X.drop(columns=[col])
    if X.shape[1] == 0:
        raise ValueError("No informative metadata columns found.")
    return X, y, "Metadata baseline", list(X.columns)

def build_pipeline(X, mode):
    if mode == "NLP":
        pre = ColumnTransformer([
            ("text", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1, strip_accents="unicode"), "__text__")
        ])
    else:
        numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers = []
        if numeric:
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
            transformers.append(("num", num_pipe, numeric))
        if categorical:
            cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
            transformers.append(("cat", cat_pipe, categorical))
        pre = ColumnTransformer(transformers)
    return Pipeline([("features", pre), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))])

def train_classifier(X, y, mode, test_size=0.2):
    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = build_pipeline(X_train, mode)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f)
    }
    classes = list(model.named_steps["classifier"].classes_)
    return model, (X_test, y_test), metrics, classification_report(y_test, pred, zero_division=0), confusion_matrix(y_test, pred, labels=classes)

def predict_texts(model, frame):
    pred = model.predict(frame)
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(frame).max(axis=1)
    return pred, confidence

def label_to_display(value):
    return str(value)
