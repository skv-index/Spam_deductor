"""Phase 2: Model Building - TF-IDF + structural, 5 classical + MLP baseline."""
import pandas as pd, numpy as np, os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack, csr_matrix

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/clean.csv", encoding="utf-8")
STRUCT = ["msg_length","word_count","url_count","has_url","has_shortener","has_phone",
          "has_money","uppercase_ratio","exclaim_count","question_count","urgency_hits","has_urgency"]

le = LabelEncoder()
y = le.fit_transform(df["label3"])  # 0 legit, 1 promo, 2 scam (alphabetical)
print("Labels:", list(le.classes_))

X_train_t, X_test_t, S_train, S_test, y_train, y_test = train_test_split(
    df["clean_text"].fillna(""), df[STRUCT].values, y,
    test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)
Xt_train = vec.fit_transform(X_train_t)
Xt_test = vec.transform(X_test_t)
Xtr_full = hstack([Xt_train, csr_matrix(S_train)])
Xte_full = hstack([Xt_test, csr_matrix(S_test)])
print(f"TF-IDF shape: {Xt_train.shape}, full: {Xtr_full.shape}")

models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "SVM_Linear": LinearSVC(class_weight="balanced", C=1.0),
    "RandomForest": RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    "XGBoost": None,  # filled below
    "MLP_baseline": MLPClassifier(hidden_layer_sizes=(128,), max_iter=15, random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", n_jobs=-1, random_state=42)
except Exception as e:
    print("XGB import fail, using HGB:", e)
    models["XGBoost"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42)

results = []
best_f1, best_name, best_model = -1, "", None

for name, clf in models.items():
    print(f"\n--- Training {name} ---")
    if name == "NaiveBayes":
        clf.fit(Xt_train, y_train)  # text only (NB needs non-negative)
        pred = clf.predict(Xt_test)
    else:
        clf.fit(Xtr_full, y_train)
        pred = clf.predict(Xte_full)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="macro", zero_division=0)
    rec = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    results.append([name, round(acc,4), round(prec,4), round(rec,4), round(f1,4)])
    print(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} F1macro={f1:.4f}")
    print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title(f"Confusion Matrix - {name} (F1={f1:.3f})")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(f"outputs/plots/cm_{name}.png", dpi=150); plt.close()
    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, clf

res = pd.DataFrame(results, columns=["model","accuracy","precision_macro","recall_macro","f1_macro"]).sort_values("f1_macro", ascending=False)
res.to_csv("outputs/metrics.csv", index=False)
print("\n" + res.to_string(index=False))

# Save best: vectorizer + label encoder + model + struct cols
with open("models/best_model.pkl","wb") as f:
    pickle.dump({"model_name": best_name, "model": best_model, "vectorizer": vec,
                 "label_encoder": le, "struct_cols": STRUCT, "use_text_only": (best_name=="NaiveBayes")}, f)
with open("outputs/best_model.txt","w") as f:
    f.write(f"Best: {best_name} F1macro={best_f1:.4f}\n" + res.to_string(index=False))
print(f"\nBest model: {best_name} saved to models/best_model.pkl")
