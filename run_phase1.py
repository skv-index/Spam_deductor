"""Phase 1: combine web datasets -> 3-class -> preprocess -> EDA plots."""
import pandas as pd, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import clean_text
from src.features import extract_features

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# 1. Load UCI (web download)
uci = pd.read_csv("data/raw/uci/SMSSpamCollection", sep="\t", header=None,
                  names=["label","message"], encoding="latin-1")
uci["label3"] = uci["label"].map({"ham":"legitimate","spam":"promotional_spam"})
uci3 = uci[["message","label3"]]

# 2. Load Combined-Labeled (web download, 5 sources merged)
cmb = pd.read_csv("data/raw/Combined-Labeled-Dataset.csv", encoding="utf-8",
                  on_bad_lines="skip", dtype=str)
cmb = cmb.dropna(subset=["message"])
def map_cmb(r):
    if str(r["smishing label"]) == "1":
        return "scam_smishing"
    if str(r["spam label"]) == "1":
        return "promotional_spam"
    return "legitimate"
cmb["label3"] = cmb.apply(map_cmb, axis=1)
cmb3 = cmb[["message","label3"]]

# 3. Load pure phishing (web download)
try:
    ph = pd.read_csv("data/raw/phishing_messages.csv", encoding="utf-8", on_bad_lines="skip")
    msg_col = "message" if "message" in ph.columns else ph.columns[2]
    ph3 = pd.DataFrame({"message": ph[msg_col].dropna().astype(str), "label3": "scam_smishing"})
except Exception as e:
    print("phishing skip:", e)
    ph3 = pd.DataFrame(columns=["message","label3"])

# 4. Combine + dedupe
full = pd.concat([uci3, cmb3, ph3], ignore_index=True)
full["message"] = full["message"].astype(str).str.strip()
full = full[full["message"].str.len() > 1]
full = full.drop_duplicates(subset=["message"])
print(f"Combined rows: {len(full)}")
print(full["label3"].value_counts())
print(full["label3"].value_counts(normalize=True).round(3))

# 5. Preprocess + features
full["clean_text"] = full["message"].apply(clean_text)
full = extract_features(full, "message")
full.to_csv("data/processed/clean.csv", index=False, encoding="utf-8")
print("Saved data/processed/clean.csv")

# ---- EDA plots ----
sns.set_style("whitegrid")

# P1 class balance
plt.figure(figsize=(7,4))
ax = sns.countplot(data=full, x="label3", order=["legitimate","promotional_spam","scam_smishing"])
plt.title("Class distribution (web-combined 3-class)")
plt.xticks(rotation=10)
for c in ax.containers: ax.bar_label(c)
plt.tight_layout(); plt.savefig("outputs/plots/01_class_balance.png", dpi=150); plt.close()

# P2 length by class
plt.figure(figsize=(8,4))
sns.boxplot(data=full[full["msg_length"]<500], x="label3", y="msg_length",
            order=["legitimate","promotional_spam","scam_smishing"])
plt.title("Message length by class")
plt.tight_layout(); plt.savefig("outputs/plots/02_length_by_class.png", dpi=150); plt.close()

# P3 structural cues %
struct = full.groupby("label3")[["has_url","has_phone","has_money","has_urgency","has_shortener"]].mean().reindex(
    ["legitimate","promotional_spam","scam_smishing"]) * 100
struct.plot(kind="bar", figsize=(9,4))
plt.title("% messages with URL / phone / money / urgency / shortener")
plt.ylabel("%"); plt.xticks(rotation=10); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("outputs/plots/03_structural_cues.png", dpi=150); plt.close()

# P4 urgency hits hist
plt.figure(figsize=(7,4))
for lab in ["legitimate","promotional_spam","scam_smishing"]:
    sns.kdeplot(full[full.label3==lab]["urgency_hits"], label=lab, fill=False)
plt.title("Urgency keyword hits distribution")
plt.legend(); plt.tight_layout(); plt.savefig("outputs/plots/04_urgency.png", dpi=150); plt.close()

# P5 top words per class (quick)
from collections import Counter
stop = set("""the to and a i you u of in is that it for on was are as with his they be at one have this from or had by not but what all were we when your can said there use an each which she do how their if will up other about out many then them these so some her would make like him into has two more very after words just where most know""".split())
for lab, fn in [("legitimate","05a"),("promotional_spam","05b"),("scam_smishing","05c")]:
    words = " ".join(full[full.label3==lab]["clean_text"]).split()
    words = [w for w in words if w not in stop and w not in ("url","phone","money","email") and len(w)>2]
    top = Counter(words).most_common(20)
    if top:
        w, c = zip(*top)
        plt.figure(figsize=(8,4)); sns.barplot(x=list(c), y=list(w))
        plt.title(f"Top 20 words - {lab}"); plt.tight_layout()
        plt.savefig(f"outputs/plots/{fn}_topwords_{lab}.png", dpi=150); plt.close()

# Summary txt
with open("outputs/eda_summary.txt","w",encoding="utf-8") as f:
    f.write("PHASE 1 EDA SUMMARY (data from web)\n")
    f.write(f"Total rows: {len(full)}\n")
    f.write(str(full["label3"].value_counts())+"\n")
    f.write(str(full["label3"].value_counts(normalize=True).round(3))+"\n\n")
    f.write("Mean length by class:\n"+str(full.groupby("label3")["msg_length"].mean().round(1))+"\n\n")
    f.write("Structural % by class:\n"+str(struct.round(1))+"\n")
print("EDA done. Plots in outputs/plots/")
