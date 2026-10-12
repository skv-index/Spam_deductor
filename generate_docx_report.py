"""Generate formatted DOCX project report for Phases I, II, and III."""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_report():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # --- TITLE PAGE ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("SPAM AND SCAM MESSAGE DETECTION USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNING")
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(28)
    r_sub = p_sub.add_run("Comprehensive Project Work Report: Phases I, II & III\n(A 3-Class Explainable Machine Learning Framework)")
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

    # Academic metadata
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(36)
    r_meta = p_meta.add_run("Course: Machine Learning Laboratory (CAT-2 Project Work)\nAcademic Year: 2024–2025 | Semester V")
    r_meta.font.size = Pt(12)
    r_meta.font.bold = True

    # Authors Table
    table_auth = doc.add_table(rows=1, cols=3)
    table_auth.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table_auth.rows[0].cells
    hdr[0].text = "Vignesh S K\nReg No: 2403717673721046"
    hdr[1].text = "Devi Prasath G\nReg No: 2403717673721011"
    hdr[2].text = "Arul Ronal X\nReg No: 2403717673721050"
    for cell in hdr:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10.5)
        set_cell_background(cell, "F0F4F8")

    doc.add_page_break()

    # --- EXECUTIVE ABSTRACT ---
    h1 = doc.add_heading("Abstract", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    doc.add_paragraph(
        "Short Message Service (SMS) is an essential communication channel for multi-factor authentication, "
        "banking notifications, and personal messaging. However, its ubiquity makes it a prime vehicle for predatory "
        "cybercrime. Existing filters and academic literature predominantly cast spam detection as a binary problem "
        "('spam' vs. 'ham'). This oversimplification conflates harmless promotional offers with predatory financial "
        "phishing (smishing), leading either to catastrophic financial fraud or customer dissatisfaction due to over-blocking."
    )
    doc.add_paragraph(
        "This project introduces an end-to-end Machine Learning system based on a 3-Class Taxonomy: 'legitimate', "
        "'promotional_spam', and 'scam_smishing'. A unified multi-source dataset of 78,310 messages was curated, and a "
        "dual-branch representation was engineered combining 5,000 TF-IDF n-grams (1-2) with 12 domain-specific "
        "structural and heuristic signals (URL shorteners, urgency triggers, monetary symbols, capitalization ratios). "
        "Six machine learning classifiers were comprehensively evaluated across a 15,662 holdout test set. "
        "Our tuned XGBoost model achieved 95.22% overall accuracy and a Macro F1-score of 0.9165, with an outstanding "
        "95.4% recall on the security-critical scam_smishing class. Production explainability is integrated via LIME "
        "and TreeExplainer SHAP, and an interactive Streamlit dashboard provides real-time single and batch inference."
    )

    # --- CHAPTER 1 ---
    doc.add_heading("1. Chapter 1: Problem Identification & Literature Survey (Phase I)", level=1)
    doc.add_heading("1.1 Background & Threat Analysis", level=2)
    doc.add_paragraph(
        "Mobile users receive dozens of unsolicited messages weekly. While promotional messages advertise legitimate products, "
        "smishing messages use social engineering and urgent language to trick users into revealing OTPs, passwords, or transferring funds. "
        "Traditional telecom-level regex and blacklist filters are easily circumvented via dynamic URL shorteners (bit.ly, tinyurl) "
        "and character obfuscation."
    )
    doc.add_heading("1.2 Related Work & Research Gap", level=2)
    doc.add_paragraph(
        "Earlier works (Almeida et al., 2011; Ghourabi et al., 2020) demonstrated high accuracy on binary datasets. "
        "However, our literature survey identified three major gaps: (1) Absence of a 3-class target space separating marketing "
        "from fraud, (2) Neglect of heuristic/structural indicators in pure bag-of-words NLP, and (3) Complete lack of explainable AI "
        "giving users clear reasons for classifications."
    )

    # --- CHAPTER 2 ---
    doc.add_heading("2. Chapter 2: Data Preprocessing & Exploratory Data Analysis (Phase II)", level=1)
    doc.add_paragraph(
        "The consolidated corpus of 78,310 records was created from the UCI SMS Spam Collection, multi-source smishing repositories, "
        "and verified phishing archives. Standard regex normalizations substituted links with <URL>, phone numbers with <PHONE>, "
        "and currency symbols with <MONEY>."
    )
    
    # Add EDA Plots if present
    if os.path.exists("outputs/plots/01_class_balance.png"):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        doc.add_picture("outputs/plots/01_class_balance.png", width=Inches(5.5))
        cap = doc.add_paragraph("Figure 2.1: Target Class Distribution Across the Consolidated Corpus")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    if os.path.exists("outputs/plots/03_structural_cues.png"):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        doc.add_picture("outputs/plots/03_structural_cues.png", width=Inches(5.5))
        cap = doc.add_paragraph("Figure 2.2: Percentage of Messages Exhibiting Specific Heuristic & Structural Signals")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    # --- CHAPTER 3 ---
    doc.add_heading("3. Chapter 3: Feature Engineering & Model Building (Phase II)", level=1)
    doc.add_paragraph(
        "The feature engineering strategy couples sparse lexical n-grams with 12 dense heuristic cues:\n"
        "• Lexical Branch: TF-IDF vectorization with 5,000 maximum features and unigram + bigram context (ngram_range=(1,2)).\n"
        "• Structural Branch: 12 features including msg_length, word_count, url_count, has_url, has_shortener, has_phone, "
        "has_money, uppercase_ratio, exclaim_count, question_count, urgency_hits, and has_urgency.\n"
        "Features are combined into a unified sparse representation using scipy.sparse.hstack."
    )
    doc.add_paragraph(
        "Six models representing diverse learning paradigms were benchmarked: Multinomial Naive Bayes, Logistic Regression, "
        "Linear Support Vector Classifier, Random Forest, Multi-Layer Perceptron (MLP), and XGBoost."
    )

    # --- CHAPTER 4 ---
    doc.add_heading("4. Chapter 4: Result Analysis & Explainability (Phase II & III)", level=1)
    doc.add_paragraph("The models were evaluated on the 15,662 holdout test set. The comparative results are presented below:")

    # Table of Model Metrics
    table_m = doc.add_table(rows=7, cols=5)
    table_m.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model Architecture", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    for i, h in enumerate(headers):
        cell = table_m.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        set_cell_background(cell, "1B365D")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    data_rows = [
        ["XGBoost (Selected Best)", "95.22%", "0.9261", "0.9080", "0.9165"],
        ["MLP Baseline", "95.13%", "0.9219", "0.9014", "0.9111"],
        ["Random Forest", "94.06%", "0.9109", "0.8642", "0.8838"],
        ["Linear SVM", "93.07%", "0.8724", "0.8933", "0.8821"],
        ["Logistic Regression", "84.65%", "0.7541", "0.8044", "0.7648"],
        ["Multinomial Naive Bayes", "85.84%", "0.8028", "0.7376", "0.7625"]
    ]
    for row_idx, r in enumerate(data_rows):
        row_cells = table_m.rows[row_idx + 1].cells
        for col_idx, val in enumerate(r):
            row_cells[col_idx].text = val
            row_cells[col_idx].paragraphs[0].runs[0].font.size = Pt(9.5)
            if row_idx == 0:
                row_cells[col_idx].paragraphs[0].runs[0].font.bold = True
                set_cell_background(row_cells[col_idx], "E8F4FD")
            elif row_idx % 2 == 1:
                set_cell_background(row_cells[col_idx], "F9FAFB")

    doc.add_paragraph().paragraph_format.space_before = Pt(10)

    if os.path.exists("outputs/plots/cm_normalized_best.png"):
        doc.add_picture("outputs/plots/cm_normalized_best.png", width=Inches(4.5))
        cap = doc.add_paragraph("Figure 4.1: Normalized Confusion Matrix for Best Model (XGBoost)")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    if os.path.exists("outputs/plots/06_top_features.png"):
        doc.add_picture("outputs/plots/06_top_features.png", width=Inches(5.2))
        cap = doc.add_paragraph("Figure 4.2: Top Global Feature Importances (XGBoost Gain)")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9.5)
        cap.runs[0].font.italic = True

    doc.add_heading("4.2 Explainability Framework (LIME & SHAP)", level=2)
    doc.add_paragraph(
        "To provide transparency, LIME generates token-level attribution scores for each prediction. For example, in smishing "
        "messages, tokens such as 'kyc', 'blocked', 'verify', and 'immediate' drive the scam score. SHAP TreeExplainer "
        "confirms that domain heuristics ('urgency_hits', 'has_url', 'url_count') serve as pivotal global decision anchors."
    )

    # --- CHAPTER 5 ---
    doc.add_heading("5. Chapter 5: Deployment & User Interface Architecture (Phase III)", level=1)
    doc.add_paragraph(
        "The application is operationalized through an interactive Streamlit UI (app.py) connected to our production inference "
        "engine (src/predict.py) and serialized bundle (models/best_model.pkl). The application provides:\n"
        "1. Real-time single message classification with color-coded alerts (Green, Orange, Red).\n"
        "2. Actionable security advice for end-users.\n"
        "3. Interactive probability breakdown and explainability reasoning drawers.\n"
        "4. Batch CSV processing capable of screening hundreds of incoming messages with exportable CSV audit logs."
    )

    # --- CHAPTER 6 ---
    doc.add_heading("6. Chapter 6: Business Insights & Strategic Recommendations", level=1)
    doc.add_paragraph(
        "1. Tiered Telecom Enforcement: Route 'promotional_spam' to a secondary inbox folder rather than outright dropping, "
        "while instantly quarantining 'scam_smishing' messages.\n"
        "2. Asymmetric Cost Management: The cost of a missed financial scam far exceeds that of a delayed promotion. "
        "Our model maintains 95.4% scam recall to aggressively protect users.\n"
        "3. Explainability-Driven Trust: Presenting specific trigger reasons in the UI allows users to recognize social engineering tactics."
    )

    # --- REFERENCES ---
    doc.add_heading("References", level=1)
    refs = [
        "Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering. ACM DocEng.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD.",
        "Ghourabi, A., Mahmood, M. A., & Alzahrani, Q. M. (2020). A hybrid CNN-LSTM model for SMS spam detection. IEEE Access.",
        "Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.",
        "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). 'Why should I trust you?': Explaining the predictions of any classifier. ACM SIGKDD."
    ]
    for r in refs:
        p_ref = doc.add_paragraph(r)
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)

    output_path = "Final_Project_Report_Phases_I_II_III.docx"
    doc.save(output_path)
    print(f"Successfully generated formatted report: {output_path}")

if __name__ == "__main__":
    create_report()
