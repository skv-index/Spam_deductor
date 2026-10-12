"""Backend connectivity: load best_model.pkl, predict + explain."""
import pickle, re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.preprocess import clean_text, URGENCY_WORDS
from src.features import extract_features

_bundle = None
def _load():
    global _bundle
    if _bundle is None:
        with open("models/best_model.pkl","rb") as f:
            _bundle = pickle.load(f)
    return _bundle

def predict_message(text: str) -> dict:
    b = _load()
    model, vec, le, STRUCT = b["model"], b["vectorizer"], b["label_encoder"], b["struct_cols"]
    clean = clean_text(text)
    row = pd.DataFrame([{"message": text, "clean_text": clean}])
    # structural
    feat = extract_features(pd.DataFrame([{"message": text}]), "message")
    S = feat[STRUCT].values
    Xt = vec.transform([clean])
    Xfull = hstack([Xt, csr_matrix(S)])
    if b.get("use_text_only"):
        pred_id = model.predict(Xt)[0]
        proba = model.predict_proba(Xt)[0] if hasattr(model,"predict_proba") else None
    else:
        pred_id = model.predict(Xfull)[0]
        proba = model.predict_proba(Xfull)[0] if hasattr(model,"predict_proba") else None
    label = le.inverse_transform([pred_id])[0]
    probs = {c: float(p) for c, p in zip(le.classes_, proba)} if proba is not None else {}
    # explanation: matched urgency words + flags + top tfidf tokens hit
    low = text.lower()
    hits = [w for w in URGENCY_WORDS if w in low]
    tokens = set(clean.split())
    vocab = vec.vocabulary_
    top_global = ["otp","verify","click","prize","claim","free","urgent","bank","award","bonus","url","money"]
    matched = [w for w in top_global if w in tokens or w in low]
    reasons = []
    if feat["has_url"].iloc[0]: reasons.append("Contains URL (smishing often uses links)")
    if feat["has_shortener"].iloc[0]: reasons.append("Uses link-shortener (high-risk)")
    if feat["has_phone"].iloc[0]: reasons.append("Contains phone number")
    if feat["has_money"].iloc[0]: reasons.append("Contains money/currency cue")
    if hits: reasons.append("Urgency words: " + ", ".join(hits[:6]))
    if matched and len(reasons) < 3: reasons.append("Trigger words: " + ", ".join(matched[:6]))
    if not reasons: reasons.append("No strong scam cues — looks like normal chat.")
    action = {"legitimate":"✅ Safe — no action",
              "promotional_spam":"⚠️ Promo — soft-warning / unsubscribe",
              "scam_smishing":"🛑 SCAM — block, do NOT click/call"}[label]
    return {"label": label, "action": action, "probs": probs,
            "clean": clean, "reasons": reasons,
            "struct": feat[STRUCT].iloc[0].to_dict()}
