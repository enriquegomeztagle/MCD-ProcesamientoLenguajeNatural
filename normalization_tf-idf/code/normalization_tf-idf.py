# %%

# %%
import re
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from contraction_fix import fix
import spacy

import random
import datasets as hf_datasets
import sklearn

random.seed(42)
np.random.seed(42)

try:
    from langdetect import detect as _detect_lang
except Exception:
    _detect_lang = None

# %%
nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "textcat"])

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#([\wáéíóúüñÁÉÍÓÚÜÑ]+)")
NON_ALNUM_RE = re.compile(r"[^0-9A-Za-záéíóúüñÁÉÍÓÚÜÑ’']+")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

EMOJI_POS = {
    "flexed_biceps",
    "clapping_hands",
    "ok_hand",
    "oncoming_fist",
    "party_popper",
    "green_heart",
    "smiling_face_with_sunglasses",
    "rocket",
    "trophy",
    "raised_fist",
}
EMOJI_NEG = {
    "pensive_face",
}
EMOJI_NEU = {
    "lion_face",
    "smirking_face",
    "smiling_face_with_open_mouth_&_cold_sweat",
    "winking_face",
    "hot_pepper",
    "backhand_index_pointing_right",
}

# %%
EN_SW = set(ENGLISH_STOP_WORDS)
ES_SW = {
    "de",
    "la",
    "que",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "más",
    "pero",
    "sus",
    "le",
    "ya",
    "o",
    "este",
    "sí",
    "porque",
    "esta",
}
IT_SW = {
    "di",
    "a",
    "da",
    "in",
    "che",
    "la",
    "e",
    "il",
    "le",
    "i",
    "un",
    "una",
    "per",
    "con",
    "non",
    "su",
    "al",
    "lo",
    "gli",
    "del",
    "della",
    "dei",
    "delle",
}
FR_SW = {
    "de",
    "la",
    "le",
    "les",
    "des",
    "et",
    "à",
    "a",
    "en",
    "un",
    "une",
    "pour",
    "avec",
    "pas",
    "sur",
    "au",
    "aux",
    "du",
    "dans",
    "ce",
    "cet",
    "cette",
    "ces",
}
STOPWORDS_ALL = EN_SW | ES_SW | IT_SW | FR_SW | {"amp", "rt"}


def _split_hashtag_token(tok: str) -> str:
    t = CAMEL_RE.sub(" ", tok)
    t = re.sub(r"gp$", " gp", t, flags=re.IGNORECASE)
    return t.lower()


def _detect_language(text: str) -> str:
    if _detect_lang is None:
        return "en"
    try:
        return _detect_lang(text)
    except Exception:
        return "en"


def clean_and_expand(text: str) -> str:
    t = (text or "").lower()
    t = fix(t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(lambda m: " " + _split_hashtag_token(m.group(1)) + " ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    raw_tokens = t.split()
    mapped = []
    for w in raw_tokens:
        if w in EMOJI_POS:
            mapped.append("emopos")
        elif w in EMOJI_NEG:
            mapped.append("emoneg")
        else:
            mapped.append(w)
    tokens = [w for w in mapped if len(w) > 1 and w not in STOPWORDS_ALL]
    return " ".join(tokens)


def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    out = []
    for tok in doc:
        lemma = tok.lemma_.lower()
        if (
            len(lemma) > 1
            and any(c.isalpha() for c in lemma)
            and lemma not in STOPWORDS_ALL
        ):
            out.append(lemma)
    return " ".join(out)


def normalize(text: str) -> str:
    cleaned = clean_and_expand(text)
    lang = _detect_language(text)
    if lang == "en":
        return lemmatize_text(cleaned)
    return cleaned


# %%
ds = load_dataset("Malekith/twitter_f1")
df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)[["text", "label"]]
df["clean"] = df["text"].astype(str).apply(normalize)
df.head(10)

# %%
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    lowercase=False,
    strip_accents=None,
)
X = vectorizer.fit_transform(df["clean"])
features = vectorizer.get_feature_names_out()
print("\n" + "-" * 50)
print("TF-IDF MATRIX INFO")
print("-" * 50)
print("\nTF-IDF matrix shape:", X.shape)
print("Vocabulary size:", len(features))

from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

import pandas as pd


def sparse_view(i):
    row = X.getrow(i)
    nz = row.nonzero()[1]
    return pd.DataFrame({"feature": features[nz], "tfidf": row.data}).sort_values(
        "tfidf", ascending=False
    )


sample_count = min(3, X.shape[0])
sample_sparse = pd.concat(
    [sparse_view(i).assign(doc_id=i) for i in range(sample_count)], ignore_index=True
)
print("\n" + "-" * 50)
print("SPARSE SAMPLE VECTORS")
print("-" * 50)
sample_sparse.to_csv(output_dir / "tfidf_sample_sparse.csv", index=False)
print("Saved outputs/tfidf_sample_sparse.csv (non-zero TF-IDF entries for sample docs)")
print(sample_sparse.head(10))

dense_rows = min(5, X.shape[0])
dense_slice = pd.DataFrame(X[:dense_rows].toarray(), columns=features)
print("\n" + "-" * 50)
print("DENSE SAMPLE VECTORS")
print("-" * 50)
dense_slice.to_csv(output_dir / "tfidf_sample_dense_first5.csv", index=False)
print(
    "Saved outputs/tfidf_sample_dense_first5.csv (dense TF-IDF matrix for first 5 docs)"
)
print(dense_slice.head(3))

pd.DataFrame({"term": features}).to_csv(output_dir / "vocabulary.csv", index=False)
df[["label", "clean"]].to_csv(output_dir / "clean_texts.csv", index=False)

print(
    "Saved sample/vector artifacts to outputs/:",
    "tfidf_sample_sparse.csv,",
    "tfidf_sample_dense_first5.csv,",
    "vocabulary.csv,",
    "clean_texts.csv",
)


# %%
def top_terms(row_idx, k=10):
    row = X.getrow(row_idx)
    nz = row.nonzero()[1]
    vals = row.data
    order = np.argsort(-vals)[:k]
    return [(features[nz[i]], float(vals[i])) for i in order]


print("\n" + "-" * 50)
print("TOP TERMS PER DOCUMENT")
print("-" * 50)
for i in range(3):
    print(f"\nDoc {i} top terms:", top_terms(i, 10))

print("\n" + "-" * 50)
print("CORPUS STATS")
print("-" * 50)
print("\nDocuments:", len(df))
print("Unique terms (vocab size):", len(features))
