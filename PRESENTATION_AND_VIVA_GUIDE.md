# 🎤 Project Presentation & Examiner Viva Q&A Guide
## 3-Class Scam & Spam Message Detection Using NLP & Machine Learning

**Course**: Machine Learning Laboratory (CAT-2 Project Work)  
**Authors**: Vignesh S K (2403717673721046) | Devi Prasath G (2403717673721011) | Arul Ronal X (2403717673721050)

---

## Part 1: Slide-by-Slide Speaker Delivery Notes (10-Minute Presentation)

| Slide | Title | Suggested Speaker | Recommended Duration | Key Points to Emphasize |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Title & Team | Vignesh | 45 sec | Introduce the project, mention that we move beyond obsolete binary spam filters to a **3-class real-world taxonomy**. |
| **2** | Problem Statement & Motivation | Vignesh | 1 min 15 sec | Explain the asymmetric cost: missing a phishing message causes financial loss, while over-blocking promo SMS annoys users. |
| **3** | Dataset & Preprocessing | Devi Prasath | 1 min 15 sec | Emphasize the 78k multi-source dataset, deduplication, and regex tokenization (`<URL>`, `<PHONE>`, `<MONEY>`). |
| **4** | Dual-Branch Feature Engineering | Devi Prasath | 1 min 30 sec | Explain why TF-IDF alone fails against obfuscation; explain the 12 structural heuristics (shorteners, urgency triggers, caps ratio). |
| **5** | Multi-Model Benchmarking | Arul Ronal | 1 min 30 sec | Walk through the 6 models; highlight why Macro-F1 was the deciding metric rather than raw accuracy; explain why XGBoost won. |
| **6** | Detailed Best Model Performance | Arul Ronal | 1 min 15 sec | Focus on the **95.4% scam recall**; show the normalized confusion matrix. |
| **7** | Explainability (LIME & SHAP) | Vignesh | 1 min 15 sec | Show LIME word triggers and SHAP global importance; prove that structural features rank in the top 5 global signals. |
| **8** | In-Depth Error Analysis | Devi Prasath | 1 min 00 sec | Discuss conversational impersonation attacks ("Hi Dad, I lost my phone") and aggressive promotional language. |
| **9** | Deployment & Streamlit UI | Arul Ronal | 1 min 00 sec | Highlight real-time inference (<10ms), single-message reasoning drawer, and batch CSV processing with export. |
| **10** | Business & Telecom Insights | Vignesh | 45 sec | Present the 3-tiered carrier routing policy and continuous retraining loop. |
| **11** | Conclusion & Future Roadmap | Vignesh | 30 sec | Summarize accomplishments and mention multilingual & edge on-device deployment. |
| **12** | Questions & Viva | All Team Members | Open | Confidently invite questions. |

---

## Part 2: Top 20 Examiner Viva Questions & Answers

### Section A: Problem Formulation & Dataset Integrity

#### Q1: Why did you formulate this as a 3-class problem instead of the standard binary (Spam vs. Ham)?
> **Answer**:  
> Traditional binary filters lump commercial marketing (e-commerce sales, store coupons) and malicious cybercrime (fake KYC threats, banking credential harvesting) into the same "spam" bucket. In real-world telecommunications, treating them identically creates a severe operational trade-off:
> 1. If an operator applies aggressive blocking to catch fraud, legitimate e-commerce promotions get dropped, harming legitimate business communication.
> 2. If the filter is softened to prevent complaints, aggressive smishing slips into the inbox, causing severe financial loss.  
> Formulating this as 3 classes allows a **tiered policy**: route legitimate SMS to the primary inbox, route promotional spam to a quiet "Offers" folder, and immediately quarantine predatory smishing with user warnings.

#### Q2: How did you handle the class imbalance in your dataset (57.4% Legit, 33.5% Scam, 9.1% Promo)?
> **Answer**:  
> We applied a multi-pronged approach:
> 1. **Stratified Splitting**: Both train and test partitions preserve the exact distribution across all 3 classes without target leakage.
> 2. **Algorithmic Weighting**: Models like Logistic Regression, Linear SVC, and Random Forest were configured with `class_weight='balanced'` or `balanced_subsample`, penalizing misclassifications inversely proportional to class frequencies.
> 3. **Metric Selection**: We selected **Macro-Averaged F1-score** rather than Accuracy as our primary evaluation metric. A naive classifier predicting only the majority class would show high accuracy (~57%) but disastrous Macro-F1 (~0.24). Macro-F1 gives equal weight to all three classes.

#### Q3: Where did your data come from, and how did you prevent data leakage?
> **Answer**:  
> We synthesized data from three established repositories: the UCI SMS Spam Collection, multi-source academic smishing corpora, and active phishing message archives. To ensure complete integrity:
> * We stripped exact duplicates across the corpus.
> * Train and test splits (80/20) were performed strictly **before** fitting the TF-IDF vectorizer and calculating normalization statistics. The vectorizer and encoders were fit solely on the training partition.

---

### Section B: Feature Engineering & NLP Representation

#### Q4: Why did you combine TF-IDF with 12 handcrafted structural features instead of using TF-IDF alone?
> **Answer**:  
> Lexical models like TF-IDF rely on vocabulary. However, modern attackers deliberately evade lexical matching through character substitutions, zero-width spaces, or novel obfuscated text.  
> Handcrafted structural features—such as the presence of URL shorteners (`bit.ly`, `tinyurl`), phone number callbacks, excessive uppercase shouting ratios, and urgency keyword counts—capture the **behavioral signature** of a scam regardless of lexical variation. Our global feature importance analysis proved that structural cues (such as `urgency_hits` and `has_url`) rank in the top 5 most important decision drivers globally.

#### Q5: Why did you use n-gram range (1, 2) instead of only unigrams in TF-IDF?
> **Answer**:  
> Unigrams capture standalone words like `verify` or `alert`. However, scam and legitimate messages often share isolated tokens. The key discriminative power lies in **bigram phrase structures**:
> * `verify otp` vs. `verify email`
> * `bank alert` vs. `traffic alert`
> * `click here` vs. `meet here`  
> Bigrams allow the linear and tree models to capture local syntactic context without requiring complex recurrent architectures.

#### Q6: How do you normalize entities like URLs, phone numbers, and money amounts?
> **Answer**:  
> Prior to tokenization, regex patterns replace raw URLs with `<URL>`, telephone numbers with `<PHONE>`, and currency amounts with `<MONEY>`. This prevents the TF-IDF vocabulary from becoming saturated with unique domain strings, random phone digits, or fluctuating dollar/rupee amounts, while retaining the semantic knowledge that an external link, phone contact, or financial sum was mentioned.

---

### Section C: Model Selection & Machine Learning Theory

#### Q7: Why did XGBoost outperform the other 5 models, including the MLP Neural Network?
> **Answer**:  
> Our dataset consists of a heterogeneous feature space: 5,000 sparse TF-IDF text features combined with 12 dense continuous/binary heuristic features.  
> Gradient boosted decision trees (XGBoost) excel at heterogeneous tabular/sparse data because:
> 1. Decision trees are scale-invariant to continuous structural features (length, uppercase ratio).
> 2. They inherently learn **non-linear feature interactions** (e.g., *IF has_url == 1 AND urgency_hits >= 2 THEN Scam*).
> 3. Linear models (Logistic Regression, Naive Bayes) assume feature independence or strictly linear hyperplanes, failing on complex multi-condition fraud patterns.
> 4. While the MLP achieved 95.13% accuracy, XGBoost achieved a higher Macro-F1 (0.9165 vs 0.9111) with faster training convergence and full compatibility with TreeExplainer SHAP.

#### Q8: Why did Multinomial Naive Bayes perform worse (Macro F1 = 0.7625)?
> **Answer**:  
> Multinomial Naive Bayes assumes conditional feature independence given the class label. In reality, words in SMS spam are strongly correlated (e.g., `account` + `blocked` + `otp`). Furthermore, MNB requires non-negative count/frequency data, restricting it primarily to lexical tokens and preventing it from fully utilizing normalized negative or scaled continuous structural signals.

#### Q9: What hyperparameters were tuned in your XGBoost model?
> **Answer**:  
> We tuned:
> * `n_estimators=200`: Number of boosting iterations, balancing expressive capacity without overfitting.
> * `max_depth=7`: Tree depth chosen to capture high-order interactions between structural cues and text n-grams without creating leaf overfitting.
> * `learning_rate=0.1`: Step size shrinkage to prevent overshooting during gradient updates.
> * `subsample=0.9` & `colsample_bytree=0.9`: Stochastic subsampling of rows and features to introduce randomness and prevent tree correlation.
> * `eval_metric='mlogloss'`: Multiclass logarithmic loss objective.

#### Q10: What is the significance of the Confusion Matrix in evaluating this system?
> **Answer**:  
> In a 3-class problem, overall accuracy obscures class-specific failure modes. The confusion matrix reveals:
> * **False Negatives on Scam (Critical)**: Missed scams classified as legitimate (129 cases, 0.82%). This is the costliest error because it allows fraudulent messages into the user's inbox.
> * **False Positives on Legit**: Legitimate messages mislabeled as spam or scam (212 cases, 2.3%).
> * **Boundary Errors**: Distinguishing promotional marketing from scam (255 cases).  
> The normalized confusion matrix shows our scam recall is **95.4%**, demonstrating high security efficacy.

---

### Section D: Explainability (XAI) & Interpretability

#### Q11: What is LIME, and how does it work in your application?
> **Answer**:  
> LIME stands for **Local Interpretable Model-agnostic Explanations**. It explains the prediction of a black-box model on a specific instance by perturbing the input text (removing random words), passing the perturbed samples through the model to observe changes in prediction probabilities, and fitting a localized linear surrogate model weighted by proximity.  
> In our Streamlit UI, LIME extracts the top positive and negative word contributions (e.g., highlighting that `kyc` and `blocked` were the primary drivers pushing the scam probability to 98%).

#### Q12: What is the difference between LIME and SHAP in your project?
> **Answer**:  
> * **LIME (Local)**: Focuses on explaining an individual message at inference time. It tells the end-user why their specific SMS was flagged.
> * **SHAP (Global)**: Based on cooperative game theory (Shapley values). We use **TreeExplainer SHAP** to quantify the overall marginal contribution of each feature across the entire dataset. It reveals which features (like `urgency_hits` and `has_url`) are the most influential across the whole model.

---

### Section E: Deployment, Latency & Edge Cases

#### Q13: How is your system deployed, and what is the inference latency?
> **Answer**:  
> The system is packaged as an interactive **Streamlit web application** (`app.py`). The backend inference pipeline (`src/predict.py`) loads the pre-trained bundle (`models/best_model.pkl`, 2.0 MB) into memory on startup.  
> Inference latency per message is **under 10 milliseconds**: regex preprocessing and feature extraction take ~1–2 ms, sparse vector transformation takes ~2 ms, and XGBoost tree traversal takes ~3 ms. This makes it suitable for real-time telecom message gateways.

#### Q14: How does your system perform on Batch Processing?
> **Answer**:  
> The UI includes a batch processing pipeline that accepts arbitrary CSV files. It vectorizes all messages in batch, executes matrix prediction, outputs an interactive results table with per-class probabilities, and allows security auditors to download an annotated CSV report in one click.

#### Q15: What are the main failure modes of your model based on your error analysis?
> **Answer**:  
> Based on our analysis of `outputs/misclassified.csv`:
> 1. **Conversational Scams**: Impersonation lures ("Hi Mom, my phone broke, text me on this new number") that do not contain links or urgency keywords in the opening message.
> 2. **Aggressive Marketing**: Promotional flyers using panic language ("Hurry! 50% off expires in 30 minutes! Claim now!") that trigger smishing urgency heuristics.

#### Q16: How would you prevent attackers from adversarial evasion (e.g. replacing 'otp' with '0-t-p')?
> **Answer**:  
> 1. Character-level n-gram or subword tokenization (e.g. BPE) to catch obfuscated spellings.
> 2. Our structural heuristics (URL shortener detection, capitalization ratios, phone patterns) remain active even if lexical words are modified.
> 3. Continual retraining using active honeypot feeds.

#### Q17: Can this model run directly on an Android smartphone?
> **Answer**:  
> Yes. The XGBoost model artifact is only 2 MB. It can be converted to ONNX format or Treelite C-code and executed natively inside an Android background service or SMS BroadcastReceiver with near-zero battery overhead.

#### Q18: Why did you not use a large Transformer model like BERT or GPT?
> **Answer**:  
> 1. **Latency & Throughput**: Mobile messaging gateways process millions of SMS per hour. XGBoost evaluates messages in under 10 ms with negligible memory overhead, whereas BERT requires GPU inference and tens of milliseconds per text.
> 2. **Structural Signal Integration**: Transformers focus on sequential text syntax but do not natively ingest tabular metadata features like URL counts and shortener flags without custom multi-modal heads.
> 3. **Environmental & Deployment Footprint**: A 2 MB XGBoost bundle deploys seamlessly on low-cost edge servers and mobile apps without requiring costly GPU instances.

#### Q19: What telecom policy would you recommend based on your business insights?
> **Answer**:  
> We recommend a **3-tier routing architecture**:
> * **Tier 1 (Legitimate)**: Delivered directly to the primary inbox with full alerts.
> * **Tier 2 (Promotional Spam)**: Diverted to a dedicated "Promotions" folder without ringing the phone.
> * **Tier 3 (Scam/Smishing)**: Quarantined at the carrier gateway or flagged with a prominent red security warning displaying the LIME explainability reasons.

#### Q20: If an examiner asks: "What was your personal individual contribution to this project?"
> * **Vignesh S K**: Architected the 3-class problem formulation, literature survey, LIME/SHAP explainability pipeline, and end-to-end integration.
> * **Devi Prasath G**: Managed multi-source dataset curation, regex tokenization, preprocessing pipelines, and exploratory data analysis plots.
> * **Arul Ronal X**: Implemented the dual-branch feature engineering pipeline, benchmarked the 6 ML models, and developed the Streamlit user interface and batch processing engine.
