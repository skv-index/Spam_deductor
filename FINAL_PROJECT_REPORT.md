# Comprehensive Project Report (Phases I, II & III)
## 3-Class Scam and Spam Message Detection Using Natural Language Processing, Dual-Branch Feature Engineering, and Machine Learning

**Course**: Machine Learning Laboratory (CAT-2 Project Work)  
**Academic Year**: 2024–2025 / 5th Semester  
**Authors**:
* **Vignesh S K** (Register No: 2403717673721046)
* **Devi Prasath G** (Register No: 2403717673721011)
* **Arul Ronal X** (Register No: 2403717673721050)

---

## Abstract

Short Message Service (SMS) and mobile messaging remain critical communication channels for personal correspondence, banking verifications, and enterprise notifications. However, their ubiquity makes them prime targets for malicious exploitation. Traditional mobile filtering mechanisms predominantly treat message classification as a binary task (`spam` vs. `ham`). This oversimplification leads to severe real-world vulnerabilities: benign marketing offers and predatory financial phishing (smishing) are grouped together, resulting in either catastrophic financial fraud (when smishing is missed) or user dissatisfaction (when legitimate promotions are aggressively blocked).

To solve this dilemma, this project introduces an end-to-end Machine Learning pipeline operating on a **3-Class Taxonomy**: `legitimate`, `promotional_spam`, and `scam_smishing`. We synthesized a unified corpus of over 78,000 raw messages from multiple benchmarks and real-world smishing repositories. A dual-branch representation pipeline was engineered, combining **5,000 TF-IDF n-grams (unigrams and bigrams)** with **12 domain-specific structural and heuristic features** (such as URL frequency, URL shortener detection, urgency keyword triggers, capitalization ratios, phone number patterns, and monetary signals). Six machine learning architectures (Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machines, Random Forest, Multi-Layer Perceptron baseline, and Extreme Gradient Boosting / XGBoost) were evaluated using stratified 5-fold validation and an independent test holdout of 15,662 messages.

Our tuned **XGBoost** model demonstrated superior performance, achieving an overall accuracy of **95.22%**, a **Macro F1-score of 0.9165**, and notably a **95.4% recall and 0.950 F1-score on the critical `scam_smishing` class**. To ensure production transparency, the system incorporates **Local Interpretable Model-agnostic Explanations (LIME)** and **TreeExplainer SHAP** to extract decision drivers. Finally, the system is operationalized as an interactive Streamlit dashboard providing single-message real-time classification with visual explainability badges, as well as batch CSV processing capabilities.

---

## Chapter 1: Problem Identification & Literature Survey (Phase I)

### 1.1 Problem Identification & Background
The exponential growth in mobile communications has rendered SMS a vital delivery channel for time-sensitive authentications (e.g., OTPs, transactional notifications). However, this high open rate (estimated globally at over 98%) has made it the weapon of choice for cybercriminals deploying **smishing** (SMS phishing). Attackers exploit psychological urgency, brand spoofing (banks, courier deliveries, tax departments), and obfuscated shortened hyperlinks (`bit.ly`, `tinyurl.com`) to harvest credentials and perpetrate monetary theft.

Existing mobile filters installed at the telecom operator layer or on mobile operating systems suffer from two fundamental weaknesses:
1. **Binary Coarseness**: Treating all unsolicited messages as generic `spam`. A flash sale message from an e-commerce platform and a malicious message impersonating a bank's KYC verification are fundamentally different in intent, mechanism, and risk profile. Collapsing them into a single category misallocates enforcement priorities.
2. **Static Rule Brittleness**: Traditional filters rely on static keyword blacklists and sender ID whitelisting. Modern attackers evade these rules through Unicode spoofing, deliberate misspellings, and dynamic URL generation.

### 1.2 Literature Survey & Related Work
* **Almeida et al. (2011)** established the landmark UCI SMS Spam Collection, benchmarked on Support Vector Machines and Naive Bayes. While historic, the dataset is purely binary and reflects communications from over a decade ago, lacking modern multi-stage phishing vectors.
* **Jain and Gupta (2018)** analyzed feature extraction methods for SMS spam detection, demonstrating that lexical features alone struggle when attackers obfuscate vocabulary. They highlighted the necessity of structural signals such as length, punctuation, and digit ratios.
* **Ghourabi et al. (2020)** evaluated deep learning architectures (CNN and LSTM) for smishing detection. While achieving high accuracy, their models functioned as complete black boxes, lacking interpretability for end-users and requiring high computational overhead.
* **Mishra and Soni (2022)** investigated explainable AI (XAI) in cybersecurity, proving that providing end-users with contextual indicators (e.g., highlighted urgency triggers or suspicious URL flags) significantly decreases the click-through rate on fraudulent links.

### 1.3 Gap Analysis & Project Objectives
Our literature survey revealed clear operational gaps:
1. **Absence of a 3-Class Framework**: Literature rarely differentiates promotional marketing from predatory fraud.
2. **Lack of Integrated Structural Signals**: Pure NLP bag-of-words ignores metadata cues like URL shorteners and urgency density.
3. **Absence of Explainability**: End users are not told *why* a message is classified as dangerous.

**Key Objectives:**
1. Formulate a robust 3-class target space: `legitimate`, `promotional_spam`, and `scam_smishing`.
2. Construct a multi-source corpus representing current threat vectors.
3. Engineer a dual-branch feature matrix coupling TF-IDF n-grams with 12 structural/heuristic cues.
4. Benchmark six diverse ML algorithms and validate using Macro-averaged metrics under class imbalance.
5. Apply LIME and SHAP for transparent local and global explainability.
6. Build and deploy an accessible web application for real-time and batch threat intelligence.

---

## Chapter 2: Data Preprocessing & Exploratory Data Analysis (Phase II)

### 2.1 Dataset Assembly & Class Harmonization
To overcome the limitations of isolated datasets, three distinct sources were harmonized:
1. **UCI SMS Spam Collection**: Providing verified legitimate personal messages and historical spam.
2. **Multi-Source Smishing Corpus**: Providing labeled smishing and promotional marketing text.
3. **Phishing Message Archives**: Providing active credential harvesting and banking fraud scripts.

After deduplication, filtering empty payloads, and standardizing schemas, the consolidated corpus contains **78,310 messages**.
* **Legitimate**: ~57.4%
* **Scam / Smishing**: ~33.5%
* **Promotional Spam**: ~9.1%

### 2.2 Preprocessing Pipeline
Text cleaning was designed to preserve threat signals while standardizing noise:
* **Token Normalization**: Specialized regex filters replace raw URLs with `<URL>`, phone numbers with `<PHONE>`, and currency amounts with `<MONEY>`.
* **Lowercasing & Whitespace Stripping**: Elimination of non-standard spacing, tabs, and duplicate newline sequences.
* **Punctuation & Noise Reduction**: Controlled stripping of non-alphanumeric noise while tracking exclamations and question marks as dedicated numerical signals.

### 2.3 Exploratory Data Analysis (EDA) Highlights
EDA was performed using Seaborn and Matplotlib (visualizations preserved in `outputs/plots/`):
1. **Message Length Distribution (`02_length_by_class.png`)**: Legitimate messages exhibit a bimodal, relatively short distribution (median length ~45 characters). Promotional and scam messages are significantly longer (median ~135 characters), packed with promotional terms, terms of service disclaimers, or detailed threat scripts.
2. **Structural Cue Discrepancies (`03_structural_cues.png`)**:
   * **URL Presence**: Present in over 68% of smishing messages, compared to < 2% of legitimate messages.
   * **URL Shorteners**: Over 38% of scam messages utilize shorteners (`bit.ly`, `tinyurl`, `goo.gl`) to mask malicious domains.
   * **Urgency Keyword Density**: Scam messages exhibit an average of 1.4 urgency tokens per message (`urgent`, `immediately`, `blocked`, `suspended`, `expire`, `verify`), compared to 0.08 in legitimate communications.
3. **Vocabulary Patterns (`05a`, `05b`, `05c`)**:
   * *Legitimate*: Common conversational words (`come`, `good`, `time`, `home`, `love`, `work`).
   * *Promotional*: Marketing words (`sale`, `free`, `off`, `discount`, `shop`, `offer`, `stores`).
   * *Scam/Smishing*: Coercive and financial terms (`account`, `kyc`, `otp`, `bank`, `verify`, `click`, `winner`, `claim`).

---

## Chapter 3: Feature Engineering & Model Building (Phase II)

### 3.1 Dual-Branch Feature Engineering
To capture both semantic syntax and structural behavioral patterns, a dual-branch feature pipeline was constructed:

#### Branch A: Lexical TF-IDF Vectorization
* **Vocabulary Limit**: 5,000 top n-grams.
* **N-gram Range**: (1, 2) (Unigrams and Bigrams to capture phrases such as `verify otp`, `bank alert`, `click here`).
* **Frequency Threshold**: `min_df=5` to eliminate rare noise and typographical idiosyncrasies.

#### Branch B: 12 Heuristic & Structural Signals
1. `msg_length`: Total character length of original message.
2. `word_count`: Number of whitespace-separated tokens.
3. `url_count`: Total hyperlinks detected via standard URL regex.
4. `has_url`: Binary flag (1 if URLs present, 0 otherwise).
5. `has_shortener`: Detection of known link compression domains.
6. `has_phone`: Regex detection of 10–12 digit phone numbers.
7. `has_money`: Detection of currency symbols (`$`, `£`, `€`, `Rs`, `INR`) and prize denominations.
8. `uppercase_ratio`: Ratio of uppercase characters to total alpha characters (measuring shouting / panic).
9. `exclaim_count`: Count of exclamation points (`!`).
10. `question_count`: Count of question marks (`?`).
11. `urgency_hits`: Number of keyword matches against a curated urgency lexicon (22 high-risk tokens).
12. `has_urgency`: Binary flag for presence of urgency cues.

The sparse TF-IDF matrix and CSR matrix of structural signals were joined via `scipy.sparse.hstack`.

### 3.2 Model Selection & Training Setup
The dataset was split using an **80/20 stratified train-test split** (Test size: 15,662 samples). Six diverse machine learning classifiers were systematically trained and evaluated:
1. **Multinomial Naive Bayes (MNB)**: Probabilistic text-only baseline.
2. **Logistic Regression (LogReg)**: Linear classifier with L2 regularization and balanced class weights.
3. **Linear Support Vector Classifier (Linear SVC)**: Maximum-margin separation with balanced weighting.
4. **Random Forest Classifier (RF)**: Ensemble of 120 decision trees with balanced subsample weighting.
5. **Multi-Layer Perceptron (MLP Baseline)**: Single hidden layer (128 units) feedforward neural network.
6. **Extreme Gradient Boosting (XGBoost)**: Gradient boosted tree ensemble (`n_estimators=200`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.9`, `eval_metric='mlogloss'`).

---

## Chapter 4: Result Analysis & Explainability (Phase II & III)

### 4.1 Comparative Model Performance
The performance metrics across the 15,662 test set messages are summarized below:

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9522** | **0.9261** | **0.9080** | **0.9165** |
| **MLP Baseline** | 0.9513 | 0.9219 | 0.9014 | 0.9111 |
| **Random Forest** | 0.9406 | 0.9109 | 0.8642 | 0.8838 |
| **Linear SVM** | 0.9307 | 0.8724 | 0.8933 | 0.8821 |
| **Logistic Regression** | 0.8465 | 0.7541 | 0.8044 | 0.7648 |
| **Naive Bayes** | 0.8584 | 0.8028 | 0.7376 | 0.7625 |

**Analysis**:
* XGBoost established the highest Macro F1-score (**0.9165**) and Accuracy (**95.22%**). Its tree-based nature allows non-linear interactions between structural signals (e.g., `has_url` combined with `urgency_hits`) and lexical n-grams without feature scaling distortions.
* Linear models (Logistic Regression and Naive Bayes) suffered lower macro recall, particularly on the minority promotional class.

### 4.2 Class-Specific Performance Breakdown (XGBoost)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Legitimate** | 0.969 | 0.976 | 0.973 | 8,990 |
| **Promotional Spam** | 0.864 | 0.794 | 0.827 | 1,429 |
| **Scam / Smishing** | 0.946 | 0.954 | 0.950 | 5,243 |

* **Fraud Protection**: For the security-critical `scam_smishing` class, the model achieves a recall of **95.4%**, catching 5,002 out of 5,243 phishing attempts.
* **Low False Positive Burden**: Only 212 legitimate messages out of 8,990 (2.3%) were flagged as non-legitimate.

### 4.3 Detailed Error Analysis
An analysis of the 748 misclassified samples (`outputs/misclassified.csv`) revealed:
* **Scam Missed as Legit (129 cases, 0.82% of test set)**: Attackers utilized highly disguised conversational lures without immediate URLs (e.g., "Hi Grandma, I lost my phone, message me here"). These require multi-turn conversational analysis.
* **Promo Over-Blocked as Scam (143 cases)**: Legitimate e-commerce promotions utilizing aggressive urgency wording ("Hurry! Offer expires in 2 hours. Claim now") triggered smishing heuristics.
* **Scam Classified as Promo (112 cases)**: Fraudulent messages masquerading as lottery or shopping vouchers without explicit banking keywords.

### 4.4 Model Explainability (LIME & SHAP)
1. **LIME (Local Explanations)**:
   * Validated on individual samples. For smishing messages, tokens such as `kyc`, `blocked`, `verify`, `otp`, and `<URL>` provided positive weights pushing probabilities toward `scam_smishing`.
   * For promotional messages, tokens like `sale`, `off`, `discount`, `stores`, and `<MONEY>` dominated.
2. **Global Feature Importance (XGBoost Gain & SHAP)**:
   * The top 10 most influential features across the entire dataset were: `off`, `urgency_hits`, `has_url`, `url_count`, `word_count`, `msg_length`, `download`, `stores`, `rs`, and `fee`.
   * This confirms that heuristic/structural engineering provides indispensable predictive power alongside vocabulary tokens.

---

## Chapter 5: Deployment & User Interface Architecture (Phase III)

### 5.1 System Integration & Interface Design
The operational application is delivered via **Streamlit** (`app.py`), directly wired to the backend prediction module (`src/predict.py`) and model artifact (`models/best_model.pkl`).

**Core Features of the Web Application:**
1. **Interactive Single Message Inference**: Users can paste any incoming SMS or click preset examples.
2. **Visual Severity Indicator**: Highlighting predictions in green (`legitimate`), orange (`promotional_spam`), and red (`scam_smishing`).
3. **Prescriptive Guidance**: Direct actionable advice accompanying the classification (e.g., "Do not click links, share OTPs, or call back").
4. **Transparent Explainability Accordion**: Displays the exact reasons why the model reached its decision (trigger words and clean token view).
5. **Structural Telemetry**: Real-time display of extracted structural metrics (URL counts, urgency hits, uppercase ratio).
6. **Batch CSV Intelligence**: Upload arbitrary CSV files containing up to 500+ records, inspect processed results in a table, and download the annotated predictions.

### 5.2 Deployment Architecture
* **Packaging**: Configured with explicit `requirements.txt` and `.gitignore`.
* **Hosting**: Ready for deployment to **Streamlit Community Cloud** or **Hugging Face Spaces** via GitHub continuous integration.

---

## Chapter 6: Business Insights & Strategic Recommendations

From an enterprise and telecom deployment perspective, the model outputs provide concrete operational strategies:
1. **Tiered Policy Enforcement**:
   * *Scam/Smishing*: Apply automated carrier-level blocking or high-visibility red quarantine flags.
   * *Promotional Spam*: Route to a dedicated "Promotions" inbox folder (similar to email categorization) rather than blocking, preserving marketing utility while cleaning the primary inbox.
2. **Asymmetric Cost Management**:
   * The business cost of a missed financial scam ($1,000s in stolen consumer funds) vastly exceeds the cost of over-filtering an e-commerce promotion. Threshold tuning should prioritize scam recall (maintaining $\ge 95\%$).
3. **Explainability Drives User Trust**:
   * Presenting users with the highlighted rationale (e.g., "Flagged because it combines an unverified shortener with an urgent OTP request") educates consumers and reduces accidental credential surrender.
4. **Continuous Learning Loop**:
   * Unlabeled or flagged messages reported by users can be periodically ingested, retrained via `run_phase1.py` $\rightarrow$ `run_phase2.py`, and redeployed.

---

## Chapter 7: Conclusion & Future Scope

### 7.1 Conclusion
This project successfully designed, implemented, and validated an end-to-end 3-class scam and spam message detection system. By integrating TF-IDF n-grams with 12 structural domain signals, the XGBoost classifier achieved a state-of-the-art Macro F1-score of **0.9165** and an exceptional **95.4% scam recall**. The end-to-end pipeline bridges the gap between academic machine learning and real-world cybersecurity through local explainability (LIME), global impact analysis (SHAP), and an interactive Streamlit dashboard.

### 7.2 Future Work
* **Multilingual & Hinglish Support**: Expanding tokenization to handle transliterated regional messaging.
* **Deep Sequential Transformer Integration**: Fine-tuning compact transformer models (e.g., MobileBERT or DistilRoBERTa) for edge deployment on mobile operating systems.
* **Real-Time URL Reputation Lookup**: Integrating live Google Safe Browsing or VirusTotal APIs during inference for zero-day URL verification.

---

## References

1. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 11th ACM Symposium on Document Engineering*, 259–262.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
3. Ghourabi, A., Mahmood, M. A., & Alzahrani, Q. M. (2020). A hybrid CNN-LSTM model for SMS spam detection in mobile communication. *IEEE Access*, 8, 131673–131685.
4. Jain, G., & Gupta, B. B. (2018). Rule-based approach to detect SMS spam on Android devices. *International Journal of Information and Computer Security*, 10(2-3), 237–258.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
6. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference*, 1135–1144.
