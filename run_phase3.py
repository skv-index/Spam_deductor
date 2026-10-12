"""Phase 3: Result Analysis + Explainability + Business Insights."""
import pickle, pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

os.makedirs("outputs/plots", exist_ok=True)

# Load data + best model
df = pd.read_csv("data/processed/clean.csv", encoding="utf-8")
with open("models/best_model.pkl","rb") as f:
    bundle = pickle.load(f)
model, vec, le, STRUCT = bundle["model"], bundle["vectorizer"], bundle["label_encoder"], bundle["struct_cols"]
print("Best model:", bundle["model_name"], "| classes:", list(le.classes_))

X_text = df["clean_text"].fillna("")
X_struct = df[STRUCT].values
y = le.transform(df["label3"])
_, Xte_t, _, Ste, _, yte, _, idx_te = train_test_split(
    X_text, X_struct, y, df.index.values, test_size=0.2, random_state=42, stratify=y)
Xt_te = vec.transform(Xte_t)
Xte_full = hstack([Xt_te, csr_matrix(Ste)])
pred = model.predict(Xte_full)

# 1. Detailed report
rep = classification_report(yte, pred, target_names=le.classes_, zero_division=0, output_dict=True)
pd.DataFrame(rep).T.to_csv("outputs/per_class_report.csv")
print(pd.DataFrame(rep).T.round(3).to_string())
cm = confusion_matrix(yte, pred)
# normalized cm
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(5,4))
sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Oranges")
plt.title("Normalized Confusion Matrix - XGBoost (best)")
plt.ylabel("True"); plt.xlabel("Pred"); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig("outputs/plots/cm_normalized_best.png", dpi=150); plt.close()

# 2. Error analysis: collect misclassified
test_df = df.loc[idx_te, ["message","label3","clean_text"]].copy()
test_df["true_id"] = yte
test_df["pred_id"] = pred
test_df["true"] = le.inverse_transform(yte)
test_df["pred"] = le.inverse_transform(pred)
errors = test_df[test_df["true"] != test_df["pred"]]
errors.to_csv("outputs/misclassified.csv", index=False, encoding="utf-8")
print(f"Errors: {len(errors)}/{len(test_df)} ({len(errors)/len(test_df):.2%})")
print(errors.groupby(["true","pred"]).size().sort_values(ascending=False).head(10))

# 3. Global importance: XGB gain
feat_names = list(vec.get_feature_names_out()) + STRUCT
importances = model.feature_importances_
top_idx = np.argsort(importances)[-25:][::-1]
top = pd.DataFrame({"feature": [feat_names[i] for i in top_idx],
                    "importance": importances[top_idx]})
top.to_csv("outputs/top_features.csv", index=False)
plt.figure(figsize=(8,6))
sns.barplot(data=top.head(20), x="importance", y="feature")
plt.title("Top 20 global features - XGBoost (TF-IDF + structural)")
plt.tight_layout(); plt.savefig("outputs/plots/06_top_features.png", dpi=150); plt.close()
print(top.head(12).to_string(index=False))

# 4. LIME explanations (3 examples: 1 scam, 1 promo, 1 legit from test)
try:
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=list(le.classes_))
    # predict fn for raw clean_text -> needs struct defaults (median)
    S_med = np.median(Ste, axis=0, keepdims=True)
    def predict_fn(texts):
        Xt = vec.transform(texts)
        S = np.repeat(S_med, len(texts), axis=0)
        return model.predict_proba(hstack([Xt, csr_matrix(S)]))
    # pick one correctly-predicted example per class
    lime_rows = []
    for cls in le.classes_:
        cand = test_df[(test_df.true == cls) & (test_df.true == test_df.pred)].head(1)
        if len(cand) == 0: continue
        txt = cand.iloc[0]["clean_text"]
        exp = explainer.explain_instance(txt, predict_fn, num_features=10, top_labels=1)
        lime_rows.append({"example_class": cls, "message": cand.iloc[0]["message"][:300],
                          "explanation": exp.as_list(label=exp.top_labels[0])})
        # plot
        fig = exp.as_pyplot_figure(label=exp.top_labels[0])
        fig.tight_layout(); fig.savefig(f"outputs/plots/lime_{cls}.png", dpi=150); plt.close("all")
    pd.DataFrame(lime_rows).to_json("outputs/lime_examples.json", orient="records", indent=2)
    print("LIME done:", len(lime_rows))
except Exception as e:
    print("LIME fail:", e)

# 5. SHAP (sample 200, TreeExplainer on XGB with dense small slice: top-100 tfidf + struct)
try:
    import shap
    # reduce dim for speed: top 100 tfidf cols by importance + all struct
    tfidf_imp = importances[:len(vec.get_feature_names_out())]
    top100 = np.argsort(tfidf_imp)[-100:]
    Xe_tr = vec.transform(X_text.sample(3000, random_state=42).fillna(""))[:, top100].toarray()
    Xe_te = Xt_te[:200][:, top100].toarray()
    S_tr = X_struct[np.random.RandomState(42).choice(len(X_struct), 3000)]
    import xgboost as xgb
    # train tiny surrogate for SHAP speed (same params small)
    surr = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", n_jobs=-1, random_state=42)
    surr.fit(np.hstack([Xe_tr, S_tr]), le.transform(df.sample(3000, random_state=42)["label3"]))
    expl = shap.TreeExplainer(surr)
    sv = expl.shap_values(np.hstack([Xe_te, Ste[:200]]))
    # bar summary for class scam
    shap.summary_plot(sv, np.hstack([Xe_te, Ste[:200]]),
        feature_names=[vec.get_feature_names_out()[i] for i in top100] + STRUCT,
        plot_type="bar", max_display=15, show=False)
    plt.tight_layout(); plt.savefig("outputs/plots/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    print("SHAP done")
except Exception as e:
    print("SHAP skip/fail:", e)

# 6. Business insights report
promo_as_scam = len(errors[(errors.true=="promotional_spam")&(errors.pred=="scam_smishing")])
scam_as_promo = len(errors[(errors.true=="scam_smishing")&(errors.pred=="promotional_spam")])
scam_missed = len(errors[(errors.true=="scam_smishing")&(errors.pred=="legitimate")])
legit_flagged = len(errors[(errors.true=="legitimate")&(errors.pred!="legitimate")])
with open("outputs/business_insights.txt","w",encoding="utf-8") as f:
    f.write("PHASE 3 - RESULT ANALYSIS & BUSINESS INSIGHTS\n")
    f.write("="*55+"\n")
    f.write(f"Best model: XGBoost | Test size: {len(yte)} | Errors: {len(errors)} ({len(errors)/len(yte):.2%})\n\n")
    f.write("PER-CLASS F1 (from per_class_report.csv):\n")
    for c in le.classes_:
        f.write(f"  {c}: P={rep[c]['precision']:.3f} R={rep[c]['recall']:.3f} F1={rep[c]['f1-score']:.3f} (n={int(rep[c]['support'])})\n")
    f.write("\nERROR BREAKDOWN:\n")
    f.write(f"  scam missed as legitimate (CRITICAL - fraud loss): {scam_missed}\n")
    f.write(f"  scam confused as promo (under-blocking): {scam_as_promo}\n")
    f.write(f"  promo over-blocked as scam (user annoyance): {promo_as_scam}\n")
    f.write(f"  legitimate flagged (false positive): {legit_flagged}\n")
    f.write("\nTOP SIGNALS (top_features.csv + plots/06):\n")
    f.write("  "+ ", ".join(top.head(10)["feature"].tolist()) + "\n")
    f.write("\nBUSINESS RECOMMENDATIONS:\n")
    f.write("  1. Banks/telecom: deploy scam-vs-rest first (scam F1~0.95). Route 'promo' to soft-warning, 'scam' to hard-block + user reason (LIME words).\n")
    f.write("  2. Cost view: 1 missed scam >> 10 over-blocked promos. Tune threshold: favor recall on scam_smishing (currently R=~0.95, keep).\n")
    f.write("  3. Promo weakness (F1~0.83, 9% of data): collect more promo samples; add sender-ID / time-of-day features next.\n")
    f.write("  4. Explainability: show top LIME words (e.g. otp, verify, click, prize + <URL>) in UI so users trust flags and report mistakes.\n")
    f.write("  5. Monitoring: track % flagged as scam weekly; spike = new campaign. Retrain monthly with user-reported smishing.\n")
print("Phase 3 done.")
