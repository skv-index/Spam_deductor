"""
ShieldSMS: Enterprise-Grade 3-Class Scam & Spam Message Detection
Built with Streamlit, XGBoost, TF-IDF + Heuristic Telemetry, and Explainable AI (LIME/SHAP).
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from src.predict import predict_message

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ShieldSMS | Intelligent Message Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 820px;
        margin: 0 0 16px 0;
        line-height: 1.5;
    }
    .hero-badge-group {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
    }
    .hero-pill.primary {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border-color: rgba(56, 189, 248, 0.35);
    }
    .hero-pill.success {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border-color: rgba(34, 197, 94, 0.35);
    }

    /* Result Verdict Cards */
    .verdict-card {
        border-radius: 18px;
        padding: 24px 28px;
        margin: 20px 0;
        border: 1px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .verdict-scam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.35);
        box-shadow: 0 12px 28px -10px rgba(239, 68, 68, 0.25);
    }
    .verdict-promo {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(180, 83, 9, 0.05) 100%);
        border-color: rgba(245, 158, 11, 0.35);
        box-shadow: 0 12px 28px -10px rgba(245, 158, 11, 0.2);
    }
    .verdict-legit {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(21, 128, 61, 0.05) 100%);
        border-color: rgba(34, 197, 94, 0.35);
        box-shadow: 0 12px 28px -10px rgba(34, 197, 94, 0.2);
    }
    .verdict-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .verdict-scam .verdict-title { color: #EF4444; }
    .verdict-promo .verdict-title { color: #F59E0B; }
    .verdict-legit .verdict-title { color: #22C55E; }

    .verdict-action {
        font-size: 1.05rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 12px;
    }

    /* Metric Glass Box */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
    }

    /* Reasoning Pill */
    .reason-tag {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .reason-tag.danger {
        background: #FEE2E2;
        color: #991B1B;
        border-color: #FCA5A5;
    }

    /* Sidebar Clean Styling */
    .sidebar-brand {
        padding: 12px 0 20px 0;
        border-bottom: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR TELEMETRY & ABOUT ====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2 style="margin:0; font-size:1.35rem; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px;">
            🛡️ ShieldSMS Core
        </h2>
        <span style="font-size:0.8rem; color:#64748B; font-weight:600;">CAT-2 Machine Learning Lab Project</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Production Model")
    st.info("**Selected Champion**: Extreme Gradient Boosting (XGBoost)\n- **Test Accuracy**: 95.22%\n- **Macro F1-Score**: 0.9165\n- **Scam Recall**: 95.4%")

    st.markdown("### 📊 Dual-Branch Input Spec")
    st.caption("• **Branch 1**: 5,000 TF-IDF n-grams (1, 2)\n• **Branch 2**: 12 Structural / Heuristic signals (URL counts, shorteners, urgency triggers, monetary cues)")

    st.divider()
    st.markdown("### 👥 Project Team")
    st.markdown("""
    - **Vignesh S K** `(2403717673721046)`
    - **Devi Prasath G** `(2403717673721011)`
    - **Arul Ronal X** `(2403717673721050)`
    """)

    st.markdown("### 🔗 Project Links")
    st.link_button("🌐 GitHub Repository", "https://github.com/skv-index/Spam_deductor", use_container_width=True)

# ==================== HERO HEADER ====================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🛡️ ShieldSMS Threat Intelligence Platform</span>
    </div>
    <div class="hero-subtitle">
        Enterprise 3-Class SMS & Message Classification Engine separating authentic communication, harmless promotional campaigns, and predatory smishing fraud with explainable AI reasoning.
    </div>
    <div class="hero-badge-group">
        <span class="hero-pill primary">⚡ XGBoost Ensemble</span>
        <span class="hero-pill success">🎯 95.4% Scam Recall</span>
        <span class="hero-pill">🔍 LIME / SHAP Explainability</span>
        <span class="hero-pill">⏱️ &lt; 10ms Inference Latency</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN TABS NAVIGATION ====================
tab_inspect, tab_batch, tab_benchmarks = st.tabs([
    "🔍 Real-Time Message Inspector",
    "📂 Bulk CSV Threat Screening",
    "📊 Benchmarks & XAI Intelligence"
])

# ================================================================
# TAB 1: REAL-TIME MESSAGE INSPECTOR
# ================================================================
with tab_inspect:
    st.markdown("#### 💬 Input Suspicious Message")
    st.caption("Choose one of our verified multi-class presets or type any custom SMS to inspect.")

    # Preset Example Chips
    sample_presets = {
        "🟢 Legitimate Personal": "Hey Sarah, are we still meeting for lunch tomorrow at 1:00 PM near the campus cafe?",
        "🟠 Promotional Marketing": "FLASH SALE! Flat 50% OFF on all running shoes this weekend only. Visit nearest store now, no coupon needed.",
        "🔴 High-Risk Bank KYC Scam": "BANK ALERT: Your SBI account KYC has been suspended! Verify OTP immediately at http://bit.ly/kyc9-unblock to restore access. Call 9876543210.",
        "🔴 Courier Phishing Scam": "FEDEX ALERT: Package #49281 delivery failed due to unpaid customs fee of $1.99. Confirm address at http://tinyurl.com/fedx-track today."
    }

    preset_cols = st.columns(4)
    for i, (label, text_val) in enumerate(sample_presets.items()):
        if preset_cols[i].button(label, use_container_width=True):
            st.session_state["msg_input"] = text_val

    user_text = st.text_area(
        "SMS Message Payload:",
        value=st.session_state.get("msg_input", ""),
        height=110,
        placeholder="Paste an SMS or mobile notification here to analyze threats..."
    )

    btn_col1, btn_col2 = st.columns([1, 4])
    analyze_click = btn_col1.button("🔍 Analyze Threat", type="primary", use_container_width=True)

    if analyze_click and user_text.strip():
        with st.spinner("Analyzing NLP lexical tokens & structural heuristics..."):
            res = predict_message(user_text)

        lbl = res["label"]
        probs = res["probs"]
        struct = res["struct"]
        reasons = res["reasons"]

        # Card Theme Mapping
        badge_meta = {
            "legitimate": {
                "css": "verdict-legit",
                "icon": "✅",
                "name": "LEGITIMATE COMMUNICATION",
                "desc": "This message exhibits normal conversational syntax with no deceptive heuristics.",
                "color": "#22C55E"
            },
            "promotional_spam": {
                "css": "verdict-promo",
                "icon": "⚠️",
                "name": "PROMOTIONAL MARKETING",
                "desc": "Commercial solicitation or discount broadcast. Safe to read, but not personalized.",
                "color": "#F59E0B"
            },
            "scam_smishing": {
                "css": "verdict-scam",
                "icon": "🛑",
                "name": "CRITICAL RISK: SCAM / SMISHING",
                "desc": "Malicious social engineering attack detected (credential harvesting / fraudulent link).",
                "color": "#EF4444"
            }
        }[lbl]

        # Top Verdict Hero Box
        st.markdown(f"""
        <div class="verdict-card {badge_meta['css']}">
            <div class="verdict-title">
                <span>{badge_meta['icon']}</span>
                <span>{badge_meta['name']}</span>
            </div>
            <div class="verdict-action">Directive: {res['action']}</div>
            <div style="font-size:0.95rem; color:#475569;">{badge_meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2-Column Detailed Analytics
        col_probs, col_reasons = st.columns([1, 1], gap="medium")

        with col_probs:
            st.markdown("##### 📈 Class Probability Breakdown")
            for c_name, c_color, c_tag in [
                ("legitimate", "#22C55E", "Legitimate"),
                ("promotional_spam", "#F59E0B", "Promotional"),
                ("scam_smishing", "#EF4444", "Scam / Smishing")
            ]:
                score = probs.get(c_name, 0.0)
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.9rem; font-weight:600;">
                    <span>{c_tag}</span>
                    <span style="color:{c_color};">{score*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(score))

        with col_reasons:
            st.markdown("##### 🧠 Explainable AI Decision Triggers")
            st.caption("Key heuristics and linguistic triggers identified by the dual-branch pipeline:")
            for r in reasons:
                is_danger = any(w in r.lower() for w in ["scam", "url", "shortener", "urgency", "phone", "money"])
                css_class = "danger" if is_danger else ""
                st.markdown(f"<span class='reason-tag {css_class}'>• {r}</span>", unsafe_allow_html=True)

        # Structural Telemetry Expandable
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔬 Structural Telemetry & NLP Tokenization"):
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Character Length", f"{int(struct.get('msg_length', 0))}")
            m2.metric("Word Count", f"{int(struct.get('word_count', 0))}")
            m3.metric("URLs Detected", f"{int(struct.get('url_count', 0))}")
            m4.metric("Urgency Triggers", f"{int(struct.get('urgency_hits', 0))}")
            m5.metric("Uppercase Ratio", f"{struct.get('uppercase_ratio', 0.0):.1%}")

            st.caption("Normalized Preprocessed Representation:")
            st.code(res["clean"], language="text")

    elif analyze_click:
        st.warning("Please enter or paste a message payload first.")

# ================================================================
# TAB 2: BULK CSV THREAT SCREENING
# ================================================================
with tab_batch:
    st.markdown("#### 📂 Enterprise Batch File Screening")
    st.caption("Screen incoming telecom message streams or customer service CSV dumps containing hundreds of records in seconds.")

    b_col1, b_col2 = st.columns([3, 1])
    batch_file = b_col1.file_uploader("Upload CSV (Requires a 'message' column)", type=["csv"])

    # Provide quick test sample download
    if os.path.exists("data/batch_test_sample.csv"):
        with open("data/batch_test_sample.csv", "rb") as f:
            b_col2.download_button(
                "📥 Download Test Sample CSV",
                f.read(),
                "sample_test_messages.csv",
                "text/csv",
                use_container_width=True
            )

    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file, encoding="utf-8", on_bad_lines="skip")
            msg_column = "message" if "message" in df_batch.columns else df_batch.columns[0]
            st.info(f"Loaded CSV with **{len(df_batch)} rows**. Evaluating column: `{msg_column}`")

            with st.spinner("Processing batch threat evaluations..."):
                raw_texts = df_batch[msg_column].fillna("").astype(str).tolist()
                batch_results = [predict_message(t) for t in raw_texts[:500]]

                records = []
                for orig, out in zip(raw_texts[:500], batch_results):
                    records.append({
                        "Message": orig,
                        "Verdict": out["label"],
                        "Action": out["action"],
                        "Scam Risk": f"{out['probs'].get('scam_smishing', 0.0)*100:.1f}%",
                        "Promo Prob": f"{out['probs'].get('promotional_spam', 0.0)*100:.1f}%",
                        "Legit Prob": f"{out['probs'].get('legitimate', 0.0)*100:.1f}%",
                        "Urgency Hits": out["struct"].get("urgency_hits", 0),
                        "Has URL": bool(out["struct"].get("has_url", 0))
                    })
                df_out = pd.DataFrame(records)

            # Summary Metric Cards
            s1, s2, s3, s4 = st.columns(4)
            total_n = len(df_out)
            scam_n = (df_out["Verdict"] == "scam_smishing").sum()
            promo_n = (df_out["Verdict"] == "promotional_spam").sum()
            legit_n = (df_out["Verdict"] == "legitimate").sum()

            s1.metric("Total Messages Screened", total_n)
            s2.metric("🛑 Scam / Smishing Flagged", f"{scam_n} ({scam_n/total_n:.1%})")
            s3.metric("⚠️ Promotional Spam", f"{promo_n} ({promo_n/total_n:.1%})")
            s4.metric("✅ Legitimate Verified", f"{legit_n} ({legit_n/total_n:.1%})")

            st.markdown("##### 📋 Live Results Preview")
            st.dataframe(df_out, use_container_width=True, height=350)

            csv_download = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Export Classified CSV Report",
                csv_download,
                "classified_threat_report.csv",
                "text/csv",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

# ================================================================
# TAB 3: BENCHMARKS & XAI INTELLIGENCE
# ================================================================
with tab_benchmarks:
    st.markdown("#### 📊 Comparative Model Evaluation (CAT-2 Benchmarking)")
    st.caption("Evaluation across 15,662 stratified test messages holding 6 diverse classification algorithms:")

    if os.path.exists("outputs/metrics.csv"):
        m_df = pd.read_csv("outputs/metrics.csv")
        st.dataframe(
            m_df.style.highlight_max(subset=["accuracy", "f1_macro"], color="#DCFCE7"),
            use_container_width=True
        )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("##### 🎯 Normalized Confusion Matrix (XGBoost)")
        if os.path.exists("outputs/plots/cm_normalized_best.png"):
            st.image("outputs/plots/cm_normalized_best.png", use_container_width=True)
        st.caption("Demonstrating 95.4% true positive recall on critical scam/smishing payloads.")

    with p_col2:
        st.markdown("##### 🔍 Top Global Predictive Features (Gain)")
        if os.path.exists("outputs/plots/06_top_features.png"):
            st.image("outputs/plots/06_top_features.png", use_container_width=True)
        st.caption("Confirms that structural heuristics (urgency_hits, has_url) rank among top global signals.")

    st.divider()
    st.markdown("### 🏛️ 3-Tier Enterprise Deployment Policy")
    pol1, pol2, pol3 = st.columns(3)
    with pol1:
        st.success("#### 1. Legitimate\n**Carrier Action**: Pass-through\n- Instant delivery to primary SMS inbox.\n- No disruption to bank OTPs or personal texts.")
    with pol2:
        st.warning("#### 2. Promotional Spam\n**Carrier Action**: Soft Routing\n- Route to silent 'Offers' folder.\n- Mute notification chime to eliminate spam fatigue.")
    with pol3:
        st.error("#### 3. Scam / Smishing\n**Carrier Action**: Quarantine\n- Gateway-level block or prominent warning banner.\n- Explainability pill warns user of deceptive link/OTP theft.")
