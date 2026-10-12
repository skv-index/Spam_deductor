"""Scam Message Detection - Streamlit UI + Backend (Phase 4)."""
import streamlit as st
import pandas as pd
from src.predict import predict_message

st.set_page_config(page_title="Scam Message Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Scam Message Detection")
st.caption("3-class: legitimate / promotional_spam / scam_smishing | XGBoost best (F1 0.92) | NLP + structural + LIME-style reasons")

examples = {
    "Legit example": "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "Promo example": "FLASH SALE! Get 50% off shoes this weekend only. Visit stores now, no OTP needed.",
    "Scam example": "BANK ALERT: Your KYC blocked! Verify OTP immediately at http://bit.ly/kyc9 to unblock. Call 9876543210",
}
cols = st.columns(3)
for i, (k, v) in enumerate(examples.items()):
    if cols[i].button(k):
        st.session_state["msg"] = v

msg = st.text_area("Enter SMS / message:", value=st.session_state.get("msg", ""), height=120,
                   placeholder="Paste suspicious SMS here...")
col1, col2 = st.columns([1,1])
predict_btn = col1.button("🔍 Classify", type="primary")
batch = col2.file_uploader("Batch CSV (column: message)", type=["csv"])

def show_result(text):
    r = predict_message(text)
    color = {"legitimate":"green","promotional_spam":"orange","scam_smishing":"red"}[r["label"]]
    st.markdown(f"### :{color}[{r['label']}]")
    st.info(r["action"])
    if r["probs"]:
        st.bar_chart(pd.Series(r["probs"]))
    with st.expander("Why this decision? (explainability)"):
        for x in r["reasons"]:
            st.write("- " + x)
        st.code(f"clean: {r['clean'][:200]}", language="text")
    with st.expander("Structural signals"):
        st.json(r["struct"])

if predict_btn and msg.strip():
    show_result(msg)
elif predict_btn:
    st.warning("Please enter a message.")

if batch is not None:
    from src.preprocess import clean_text
    df = pd.read_csv(batch, encoding="utf-8", on_bad_lines="skip")
    col = "message" if "message" in df.columns else df.columns[0]
    outs = [predict_message(str(t)) for t in df[col].fillna("").tolist()[:500]]
    res = pd.DataFrame([{"message": m, "predicted": o["label"], "action": o["action"],
                         **{f"p_{k}": v for k, v in o["probs"].items()}} for m, o in zip(df[col], outs)])
    st.dataframe(res.head(50))
    st.download_button("⬇️ Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")

st.divider()
st.caption("Backend: models/best_model.pkl (TF-IDF 5000 + 12 structural, XGBoost) | src/predict.py | Retrain: run_phase1.py → run_phase2.py")
