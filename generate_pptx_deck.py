"""Generate a professional PowerPoint slide deck (12 slides) for project viva/presentation."""
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def set_slide_background(slide, prs, color_rgb):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color_rgb

def set_cell_background(cell, hex_str):
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(r, g, b)

def add_header(slide, title_text, category_text="ML LAB PROJECT (CAT-2)"):
    # Category tag
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.4))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = RGBColor(0x3B, 0x82, 0xF6) # Blue

    # Slide Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A) # Dark Navy Slate

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, prs, RGBColor(0x0F, 0x17, 0x2A)) # Dark Theme Title

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = "3-CLASS SCAM & SPAM MESSAGE DETECTION"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    p2 = tf.add_paragraph()
    p2.text = "Using NLP, Dual-Branch Feature Engineering, and Explainable Machine Learning"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p2.space_before = Pt(12)

    tb_team = s1.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.8))
    tf_team = tb_team.text_frame
    
    p3 = tf_team.paragraphs[0]
    p3.text = "PROJECT TEAM (SEMESTER 5 | ML LAB CAT-2)"
    p3.font.size = Pt(12)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8)

    p4 = tf_team.add_paragraph()
    p4.text = "• Vignesh S K (2403717673721046)   • Devi Prasath G (2403717673721011)   • Arul Ronal X (2403717673721050)"
    p4.font.size = Pt(14)
    p4.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    p4.space_before = Pt(6)

    # ==================== SLIDE 2: PROBLEM STATEMENT & MOTIVATION ====================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s2, "Problem Statement & The 3-Class Motivation")

    tb = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    bullets = [
        ("The Shortcoming of Traditional Binary Filters:", "Standard datasets (e.g. UCI Spam) treat spam as binary (spam vs. ham). In real life, grouping harmless flash sales with predatory financial smishing creates severe trade-offs."),
        ("Catastrophic Real-World Consequences:", "1. If you under-block to avoid customer complaints, dangerous phishing messages steal banking credentials and OTPs.\n2. If you over-block aggressively, legitimate marketing and discounts get discarded, hurting businesses."),
        ("Our Solution - 3-Class Granular Architecture:", "1. Legitimate: Genuine personal or transactional SMS (e.g., OTPs, personal chats).\n2. Promotional Spam: Commercial marketing, offers, flash sales (soft-filter/folder).\n3. Scam / Smishing: High-risk phishing, fake KYC threats, banking fraud (hard-quarantine)."),
        ("Key Contribution:", "Engineered a production-ready ML system combining 5,000 TF-IDF n-grams with 12 structural/heuristic features, delivering 95.4% scam recall and transparent LIME/SHAP explainability.")
    ]
    for i, (title, body) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title} "
        p.font.bold = True
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.space_before = Pt(10)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # ==================== SLIDE 3: DATASET & PREPROCESSING ====================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s3, "Dataset Consolidation & Preprocessing Pipeline")

    tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets_s3 = [
        ("Multi-Source Synthesis (78,310 Samples):", "Merged UCI SMS Spam Collection + Multi-Source Smishing Corpus + Phishing Message Archives to eliminate single-source bias."),
        ("Class Distribution:", "• Legitimate: 57.4% (~44,950)\n• Scam/Smishing: 33.5% (~26,215)\n• Promotional Spam: 9.1% (~7,145)"),
        ("Specialized Regex Token Normalization:", "• URLs replaced with '<URL>' token while extracting link metrics.\n• Phone numbers replaced with '<PHONE>'.\n• Monetary expressions ($/Rs) mapped to '<MONEY>'."),
        ("Stratified Split:", "80% Train (62,648 rows) / 20% Test (15,662 rows) preserving exact class ratios.")
    ]
    for i, (title, body) in enumerate(bullets_s3):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(13.5)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12.5)

    if os.path.exists("outputs/plots/01_class_balance.png"):
        s3.shapes.add_picture("outputs/plots/01_class_balance.png", Inches(7.2), Inches(1.8), width=Inches(5.4))

    # ==================== SLIDE 4: DUAL-BRANCH FEATURE ENGINEERING ====================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s4, "Dual-Branch Feature Engineering Architecture")

    tb = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets_s4 = [
        ("Branch A: Lexical TF-IDF Vectorization", "• 5,000 max features, n-gram range (1, 2) unigrams and bigrams.\n• Captures composite attack phrases: 'verify otp', 'account suspended', 'click here'."),
        ("Branch B: 12 Heuristic / Structural Signals", "1. msg_length & word_count: Message verbosity.\n2. url_count & has_url: Direct link prevalence.\n3. has_shortener: bit.ly, tinyurl, goo.gl domain matching.\n4. has_phone & has_money: Callbacks & prize numbers.\n5. uppercase_ratio: Measure of panic / aggressive shouting.\n6. urgency_hits & has_urgency: Lexicon match for 22 urgency triggers ('blocked', 'immediate', 'kyc', 'expire')."),
        ("Unified Sparse Assembly:", "scipy.sparse.hstack combines text sparse matrix with dense structural signals into a unified feature tensor.")
    ]
    for i, (title, body) in enumerate(bullets_s4):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(13)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12)

    if os.path.exists("outputs/plots/03_structural_cues.png"):
        s4.shapes.add_picture("outputs/plots/03_structural_cues.png", Inches(7.2), Inches(1.8), width=Inches(5.5))

    # ==================== SLIDE 5: MULTI-MODEL BENCHMARKING ====================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s5, "Multi-Model Comparison & Evaluation Metrics")

    # Add Table
    table_shape = s5.shapes.add_table(7, 5, Inches(0.8), Inches(1.7), Inches(11.7), Inches(3.2))
    table = table_shape.table
    hdrs = ["Model Architecture", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    for i, h in enumerate(hdrs):
        cell = table.cell(0, i)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        if p.runs:
            p.runs[0].font.bold = True
            p.runs[0].font.size = Pt(13)
            p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(cell, "1E293B")

    bench_data = [
        ["XGBoost (Winner)", "95.22%", "0.9261", "0.9080", "0.9165"],
        ["MLP Neural Net", "95.13%", "0.9219", "0.9014", "0.9111"],
        ["Random Forest", "94.06%", "0.9109", "0.8642", "0.8838"],
        ["Linear SVM", "93.07%", "0.8724", "0.8933", "0.8821"],
        ["Logistic Regression", "84.65%", "0.7541", "0.8044", "0.7648"],
        ["Multinomial Naive Bayes", "85.84%", "0.8028", "0.7376", "0.7625"]
    ]
    for row_idx, row in enumerate(bench_data):
        for col_idx, val in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].font.size = Pt(12)
            if row_idx == 0:
                if p.runs: p.runs[0].font.bold = True
                set_cell_background(cell, "DCFCE7") # Light Green highlight
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F1F5F9")

    tb_notes = s5.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    tf_notes = tb_notes.text_frame
    p_n = tf_notes.paragraphs[0]
    p_n.text = "Key Takeaway: XGBoost outperformed linear and tree alternatives because it captures non-linear synergies between structural heuristics (e.g. url shorteners + urgency flags) and sparse vocabulary bigrams."
    p_n.font.size = Pt(13)
    p_n.font.bold = True
    p_n.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    # ==================== SLIDE 6: DETAILED BEST MODEL PERFORMANCE ====================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s6, "Detailed Class-Wise Performance (XGBoost)")

    tb = s6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.2), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets_s6 = [
        ("Legitimate Class (Support: 8,990):", "• Precision: 96.9% | Recall: 97.6% | F1: 0.973\n• Only 2.3% genuine messages flagged (minimal user disruption)."),
        ("Scam / Smishing Class (Support: 5,243):", "• Precision: 94.6% | Recall: 95.4% | F1: 0.950\n• 5,002 out of 5,243 aggressive smishing attacks captured!"),
        ("Promotional Spam Class (Support: 1,429):", "• Precision: 86.4% | Recall: 79.4% | F1: 0.827\n• Successfully separated commercial flyers from cybercrime threats."),
        ("Total Test Set Size:", "15,662 stratified samples evaluated without leakage.")
    ]
    for i, (title, body) in enumerate(bullets_s6):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(13.5)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12.5)

    if os.path.exists("outputs/plots/cm_normalized_best.png"):
        s6.shapes.add_picture("outputs/plots/cm_normalized_best.png", Inches(7.4), Inches(1.8), width=Inches(5.2))

    # ==================== SLIDE 7: EXPLAINABILITY (LIME & SHAP) ====================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s7, "Explainable AI (XAI): LIME & SHAP Interpretability")

    tb = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.2), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets_s7 = [
        ("Why Explainability Matters in Telecom/Security:", "Black-box AI cannot be trusted blindly. If a message is blocked, both security analysts and end-users need to know the exact justification."),
        ("Local Explanations via LIME:", "• Evaluates word perturbations for each incoming SMS.\n• Smishing triggers: 'kyc', 'blocked', 'verify', 'immediate', '<URL>'.\n• Promotional triggers: 'off', 'sale', 'flat', 'stores', '<MONEY>'."),
        ("Global Feature Importance (XGBoost Gain & SHAP):", "Top overall signals: 'off', 'urgency_hits', 'has_url', 'url_count', 'word_count', 'msg_length', 'download', 'fee'."),
        ("Validation:", "Proves that structural cues are in the top 5 global decision factors.")
    ]
    for i, (title, body) in enumerate(bullets_s7):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(13.5)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(12.5)

    if os.path.exists("outputs/plots/06_top_features.png"):
        s7.shapes.add_picture("outputs/plots/06_top_features.png", Inches(7.4), Inches(1.8), width=Inches(5.4))

    # ==================== SLIDE 8: ERROR ANALYSIS ====================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s8, "In-Depth Error Analysis & Misclassifications")

    tb = s8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    errors_list = [
        ("Overall Test Error Rate:", "Only 748 errors out of 15,662 samples (4.78% error rate, 95.22% accuracy)."),
        ("1. Critical Error: Scam Missed as Legitimate (129 Cases | 0.82%):", "Conversational impersonation attacks ('Hi Dad, I broke my phone, this is my new WhatsApp'). They lack hyperlinks, money mentions, or urgency words on the first turn. Remedy: Multi-turn conversational context."),
        ("2. False Positives: Legitimate Flagged as Spam/Scam (212 Cases):", "Occurs when personal messages discuss money or contain shortened links (e.g. friend sharing a bit.ly link to a photo album)."),
        ("3. Promo vs. Scam Confusion (255 Cases):", "Aggressive marketing that borrows scam tactics ('URGENT: 50% discount expires in 10 mins. Click now!') blurs boundaries. Remedy: Sender-ID whitelisting.")
    ]
    for i, (title, body) in enumerate(errors_list):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(14)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(13)

    # ==================== SLIDE 9: STREAMLIT UI & INTERACTIVE DEMO ====================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s9, "Deployment & Streamlit UI Dashboard")

    tb = s9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    ui_points = [
        ("Real-Time Single Message Classifier:", "• Quick preset sample buttons for instant demo.\n• Dynamic badge rendering (Green: Legitimate, Orange: Promotional Spam, Red: Scam/Smishing).\n• Class probability distribution bar charts."),
        ("Transparent Decision Reasoning (Explainability Drawer):", "• Shows extracted trigger keywords (e.g., OTP, verify, KYC, link).\n• Displays structural telemetry JSON (has_url, urgency_count, uppercase_ratio)."),
        ("High-Throughput Batch CSV Screening:", "• Upload CSV datasets containing hundreds of SMS records.\n• Interactive paginated table display.\n• 1-Click download of predicted CSV audit reports."),
        ("Production Readiness:", "Lightweight serialized pickle artifact (models/best_model.pkl, 2.0 MB) executes sub-10ms inference per message.")
    ]
    for i, (title, body) in enumerate(ui_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(14)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(13)

    # ==================== SLIDE 10: BUSINESS & INDUSTRY INSIGHTS ====================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s10, "Business Insights & Telecom Policy Recommendations")

    tb = s10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    biz_points = [
        ("1. Tiered Enforcement Strategy for Telecom Operators:", "• Legitimate: Route directly to Primary Inbox.\n• Promotional Spam: Route to a separate 'Promotions' folder (no notification sound).\n• Scam/Smishing: Immediate quarantine with a user warning banner."),
        ("2. Asymmetric Cost Awareness:", "In financial cybersecurity, 1 missed scam causing a $1,000 theft is 100x worse than over-filtering a 20% shoe coupon. The model's threshold is tuned to maximize scam recall (95.4%)."),
        ("3. Continuous Retraining Pipeline:", "When users flag false positives or report novel smishing SMS, incoming samples feed into run_phase1.py and run_phase2.py for periodic model updates."),
        ("4. Explainability Drives User Compliance:", "Showing users WHY a message was flagged teaches them to spot suspicious bit.ly links and urgent OTP traps.")
    ]
    for i, (title, body) in enumerate(biz_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(14)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(13)

    # ==================== SLIDE 11: CONCLUSION & FUTURE SCOPE ====================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, prs, RGBColor(0xF8, 0xFA, 0xFC))
    add_header(s11, "Conclusion & Future Roadmap")

    tb = s11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.2))
    tf = tb.text_frame
    tf.word_wrap = True

    conc_points = [
        ("Project Achievements:", "• Successfully transitioned SMS security from flawed binary labels to a 3-class taxonomy.\n• Synthesized a 78k message corpus and engineered dual-branch lexical + structural features.\n• XGBoost delivered 95.22% accuracy and 0.9165 Macro F1, catching 95.4% of smishing attempts.\n• Complete explainability (LIME/SHAP) and responsive Streamlit UI."),
        ("Future Roadmap & Enhancements:", "• Multilingual & Code-Mixed NLP: Train models on Romanized regional languages (Hinglish, Tanglish).\n• Real-Time Threat Intelligence APIs: Query VirusTotal / Google Safe Browsing on detected URLs.\n• On-Device Edge Deployment: Quantize the XGBoost / DistilBERT pipeline for Android APK execution.")
    ]
    for i, (title, body) in enumerate(conc_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{title}\n"
        p.font.bold = True
        p.font.size = Pt(14)
        p.space_before = Pt(10)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(13)

    # ==================== SLIDE 12: QUESTIONS & VIVA ====================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, prs, RGBColor(0x0F, 0x17, 0x2A))

    tb = s12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.0))
    tf = tb.text_frame
    p_q1 = tf.paragraphs[0]
    p_q1.text = "THANK YOU!"
    p_q1.font.size = Pt(44)
    p_q1.font.bold = True
    p_q1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    p_q1.alignment = PP_ALIGN.CENTER

    p_q2 = tf.add_paragraph()
    p_q2.text = "Open for Questions & Viva Examination"
    p_q2.font.size = Pt(22)
    p_q2.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8)
    p_q2.alignment = PP_ALIGN.CENTER
    p_q2.space_before = Pt(16)

    output_pptx = "Project_Presentation_Deck.pptx"
    prs.save(output_pptx)
    print(f"Successfully generated PowerPoint deck: {output_pptx}")

if __name__ == "__main__":
    create_presentation()
