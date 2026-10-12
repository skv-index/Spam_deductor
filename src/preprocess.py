import re

URL_RE = re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\S+|smsg\.io/\S+|tx\.vc/\S+|\S+\.(com|io|ly|vc|me|co|net|org)/\S*', re.I)
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{7,}\d)')
MONEY_RE = re.compile(r'(₹|\$|£|€|rs\.?|inr|usd)\s*\d[\d,\.]*|\d[\d,\.]*\s*(rupees|rs|usd|dollars|pounds)', re.I)
EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
SHORTENER_RE = re.compile(r'bit\.ly|tinyurl|cutt\.ly|goo\.gl|t\.co|smsg\.io|tx\.vc|is\.gd|ow\.ly', re.I)

URGENCY_WORDS = ["urgent","verify","immediately","blocked","suspended","alert","otp","pin",
  "kyc","bank","prize","winner","lottery","claim","free","congratulations","click","link",
  "activate","expire","locked","security","confirm","update","award","bonus"]

KEEP_WORDS = set(["free","urgent","win","prize","otp","bank","verify","click","claim"])

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = EMAIL_RE.sub(" <EMAIL> ", t)
    t = URL_RE.sub(" <URL> ", t)
    t = PHONE_RE.sub(" <PHONE> ", t)
    t = MONEY_RE.sub(" <MONEY> ", t)
    # keep <URL> style tokens, remove other punctuation
    t = re.sub(r"[^a-z0-9<>\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
