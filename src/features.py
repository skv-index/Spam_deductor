import re
import pandas as pd
from .preprocess import URL_RE, PHONE_RE, MONEY_RE, SHORTENER_RE, URGENCY_WORDS

def extract_features(df: pd.DataFrame, text_col="message") -> pd.DataFrame:
    out = df.copy()
    txt = out[text_col].fillna("").astype(str)
    out["msg_length"] = txt.str.len()
    out["word_count"] = txt.str.split().str.len()
    out["url_count"] = txt.apply(lambda x: len(URL_RE.findall(x)))
    out["has_url"] = (out["url_count"] > 0).astype(int)
    out["has_shortener"] = txt.str.contains(SHORTENER_RE, na=False).astype(int)
    out["has_phone"] = txt.str.contains(PHONE_RE, na=False).astype(int)
    out["has_money"] = txt.str.contains(MONEY_RE, na=False).astype(int)
    out["uppercase_ratio"] = txt.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    out["exclaim_count"] = txt.str.count("!")
    out["question_count"] = txt.str.count(r"\?")
    low = txt.str.lower()
    out["urgency_hits"] = low.apply(lambda x: sum(w in x for w in URGENCY_WORDS))
    out["has_urgency"] = (out["urgency_hits"] > 0).astype(int)
    return out
