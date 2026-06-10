import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="spamShield — AI Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
ps = PorterStemmer()

@st.cache_resource
def load_model():
    tfidf  = pickle.load(open("vectorizer.pkl", "rb"))
    model  = pickle.load(open("model.pkl",      "rb"))
    return tfidf, model

tfidf, model = load_model()

# ── NLP PIPELINE ─────────────────────────────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words("english") and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── GLOBAL RESET & BASE ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #050810 !important;
    color: #e8f0ff !important;
}

/* Animated mesh gradient background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 15% 20%, rgba(59,127,255,0.13) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 75%, rgba(162,89,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(0,229,192,0.05) 0%, transparent 70%);
    pointer-events: none;
}

/* Grid overlay */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(99,179,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,255,0.035) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse 90% 80% at 50% 50%, black 0%, transparent 100%);
    pointer-events: none;
}

/* Main content block */
.block-container {
    max-width: 780px !important;
    padding: 2.5rem 1.5rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ── HIDE STREAMLIT CHROME ───────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,255,0.15); border-radius: 99px; }

/* ── HEADINGS ────────────────────────────────────────── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── TEXTAREA ────────────────────────────────────────── */
.stTextArea textarea {
    background: #0c1525 !important;
    border: 1px solid rgba(99,179,255,0.18) !important;
    border-radius: 16px !important;
    color: #e8f0ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    padding: 18px !important;
    caret-color: #3b7fff;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: rgba(59,127,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(59,127,255,0.10), 0 4px 24px rgba(59,127,255,0.08) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: rgba(106,127,168,0.55) !important; }
.stTextArea label {
    color: rgba(0,229,192,0.8) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
}

/* ── PRIMARY BUTTON ──────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b7fff, #a259ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 28px rgba(59,127,255,0.38), 0 1px 0 rgba(255,255,255,0.12) inset !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(59,127,255,0.52), 0 1px 0 rgba(255,255,255,0.12) inset !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── SELECTBOX ───────────────────────────────────────── */
.stSelectbox > div > div {
    background: #0c1525 !important;
    border: 1px solid rgba(99,179,255,0.18) !important;
    border-radius: 12px !important;
    color: #e8f0ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}

/* ── DIVIDER ─────────────────────────────────────────── */
hr { border-color: rgba(99,179,255,0.10) !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 16px 0 36px;">
  <div style="
    width:80px; height:80px; border-radius:50%;
    background: linear-gradient(135deg,#3b7fff,#a259ff);
    margin:0 auto 20px;
    display:flex; align-items:center; justify-content:center;
    font-size:36px;
    box-shadow: 0 0 0 1px rgba(59,127,255,0.3), 0 0 40px rgba(59,127,255,0.4), 0 0 80px rgba(59,127,255,0.18);
    animation: pulseRing 3s ease-in-out infinite;
  ">🛡️</div>

  <div style="
    font-family:'DM Mono',monospace;
    font-size:11px; letter-spacing:3px; color:#00e5c0;
    text-transform:uppercase; margin-bottom:10px; opacity:0.85;
  ">AI-Powered Detection</div>

  <h1 style="
    font-size:clamp(36px,6vw,58px); font-weight:800;
    letter-spacing:-2px; line-height:1.05;
    background:linear-gradient(135deg,#fff 0%,#b3cfff 40%,#a259ff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:12px;
  ">SpamShield</h1>

  <p style="
    font-family:'DM Mono',monospace; font-size:13px;
    color:#6a7fa8; max-width:400px; margin:0 auto 16px;
    line-height:1.6;
  ">Multinomial Naive Bayes · TF-IDF · NLTK · Porter Stemmer</p>

  <span style="
    display:inline-flex; align-items:center; gap:7px;
    padding:6px 16px; border-radius:99px;
    background:rgba(0,229,160,0.08); border:1px solid rgba(0,229,160,0.22);
    font-family:'DM Mono',monospace; font-size:12px; color:#00e5c0;
  ">
    <span style="
      width:7px;height:7px;border-radius:50%;background:#00e5c0;
      animation:blink 2s ease-in-out infinite; display:inline-block;
    "></span>
    ~97–98% accuracy on UCI Spam Collection
  </span>
</div>

<style>
@keyframes pulseRing {
  0%,100% { box-shadow:0 0 0 1px rgba(59,127,255,0.3),0 0 40px rgba(59,127,255,0.4),0 0 80px rgba(59,127,255,0.18); }
  50%      { box-shadow:0 0 0 1px rgba(59,127,255,0.5),0 0 60px rgba(59,127,255,0.55),0 0 120px rgba(59,127,255,0.28); }
}
@keyframes blink {
  0%,100% { opacity:1; } 50% { opacity:0.3; }
}
</style>
""", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:28px;">
  <div style="
    background:#0c1120; border:1px solid rgba(99,179,255,0.12);
    border-radius:14px; padding:16px 12px; text-align:center;
    position:relative; overflow:hidden;
    transition: transform 0.2s;
  ">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,#3b7fff,transparent);opacity:0.5;"></div>
    <div style="font-size:24px;font-weight:700;
      background:linear-gradient(90deg,#3b7fff,#a259ff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">5,572</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a7fa8;margin-top:4px;letter-spacing:1px;">Training Messages</div>
  </div>
  <div style="
    background:#0c1120; border:1px solid rgba(99,179,255,0.12);
    border-radius:14px; padding:16px 12px; text-align:center;
    position:relative; overflow:hidden;
  ">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,#a259ff,transparent);opacity:0.5;"></div>
    <div style="font-size:24px;font-weight:700;
      background:linear-gradient(90deg,#a259ff,#3b7fff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">3,000</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a7fa8;margin-top:4px;letter-spacing:1px;">TF-IDF Features</div>
  </div>
  <div style="
    background:#0c1120; border:1px solid rgba(99,179,255,0.12);
    border-radius:14px; padding:16px 12px; text-align:center;
    position:relative; overflow:hidden;
  ">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,transparent,#00e5c0,transparent);opacity:0.5;"></div>
    <div style="font-size:24px;font-weight:700;
      background:linear-gradient(90deg,#00e5c0,#3b7fff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">97%+</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a7fa8;margin-top:4px;letter-spacing:1px;">Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN CARD ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background:#0c1120; border:1px solid rgba(99,179,255,0.12);
  border-radius:24px; padding:32px 32px 8px;
  box-shadow:0 8px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
  margin-bottom:0;
  position:relative; overflow:hidden;
">
  <div style="position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(59,127,255,0.07) 0%,transparent 60%);
    pointer-events:none;"></div>

  <div style="
    font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2.5px;
    color:#00e5c0; text-transform:uppercase; margin-bottom:18px;
    display:flex; align-items:center; gap:8px;
  ">
    <span style="display:inline-block;width:20px;height:1px;background:#00e5c0;opacity:0.5;"></span>
    Message Input
  </div>
""", unsafe_allow_html=True)

# ── NLP PIPELINE VISUAL ───────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0;margin-bottom:20px;overflow-x:auto;padding-bottom:4px;">
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">📝</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Input</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">🧹</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Clean</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">✂️</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Tokenize</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">🌿</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Stem</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">📊</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">TF-IDF</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">🧠</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Model</div>
  </div>
  <div style="width:28px;height:1px;background:linear-gradient(90deg,rgba(99,179,255,0.2),transparent);flex-shrink:0;margin-bottom:16px;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:7px;flex-shrink:0;">
    <div style="width:36px;height:36px;border-radius:10px;background:#121b2e;
      border:1px solid rgba(99,179,255,0.18);display:flex;align-items:center;
      justify-content:center;font-size:16px;">✅</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6a7fa8;letter-spacing:0.5px;white-space:nowrap;">Result</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SAMPLE MESSAGE SELECTOR ───────────────────────────────────────────────────
sample_options = {
    "— pick a sample —": "",
    "🚨 Spam: Prize winner claim": "Congratulations! You've WON a £1000 cash prize! You have been selected as our lucky winner. Click the link now to claim your reward before it expires. Limited time offer!",
    "🚨 Spam: Free iPhone offer":  "URGENT! You have been selected for a FREE iPhone 15 Pro. This exclusive offer expires in 24 hours. Act now and reply YES to claim. T&C apply.",
    "✅ Ham: Meeting reminder":     "Hey, are you joining the team meeting at 3pm today? Let me know if you can't make it and I'll send you the notes afterwards.",
    "✅ Ham: Package delivery":     "Your package has been shipped and will arrive tomorrow between 9am and 1pm. Tracking number: UK48291. No signature required.",
}

selected = st.selectbox("Try a sample message", list(sample_options.keys()), label_visibility="collapsed")

# ── TEXT INPUT ────────────────────────────────────────────────────────────────
default_text = sample_options[selected] if selected != "— pick a sample —" else ""
input_sms = st.text_area(
    "MESSAGE TEXT",
    value=default_text,
    height=160,
    placeholder="Paste your email or SMS here… e.g. 'Congratulations! You have won a prize. Click here to claim!'"
)

# close the main card div
st.markdown("</div>", unsafe_allow_html=True)

# ── PADDING ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
predict_clicked = st.button("check message", use_container_width=True)

# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_clicked:
    if not input_sms.strip():
        st.markdown("""
        <div style="
          background:rgba(255,63,94,0.08); border:1px solid rgba(255,63,94,0.25);
          border-radius:14px; padding:16px 20px; margin-top:16px;
          font-family:'DM Mono',monospace; font-size:13px; color:#ff3f5e;
          display:flex; align-items:center; gap:10px;
        ">⚠️ &nbsp; Please enter a message before analyzing.</div>
        """, unsafe_allow_html=True)
    else:
        # Run NLP pipeline
        with st.spinner("Running NLP pipeline…"):
            transformed = transform_text(input_sms)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]

        is_spam = result == 1
        tokens = transformed.split()[:16]

        # ── RESULT CARD ───────────────────────────────────────────────────────
        if is_spam:
            st.markdown(f"""
            <div style="
              background:linear-gradient(135deg,rgba(255,63,94,0.10),rgba(255,63,94,0.05));
              border:1px solid rgba(255,63,94,0.30);
              border-radius:20px; padding:28px 28px 24px;
              margin-top:16px;
              box-shadow:0 4px 40px rgba(255,63,94,0.15);
              position:relative; overflow:hidden;
              animation: fadeSlide 0.4s ease forwards;
            ">
              <div style="position:absolute;top:0;left:0;right:0;height:3px;
                background:linear-gradient(90deg,transparent,#ff3f5e,transparent);"></div>

              <div style="display:flex;align-items:center;gap:18px;margin-bottom:22px;">
                <div style="
                  width:60px;height:60px;border-radius:50%;flex-shrink:0;
                  background:rgba(255,63,94,0.15); border:1px solid rgba(255,63,94,0.35);
                  display:flex;align-items:center;justify-content:center;font-size:28px;
                  box-shadow:0 0 28px rgba(255,63,94,0.28);
                ">🚨</div>
                <div>
                  <div style="font-size:30px;font-weight:800;letter-spacing:-1px;color:#ff3f5e;line-height:1;">SPAM DETECTED</div>
                  <div style="font-family:'DM Mono',monospace;font-size:12px;color:#6a7fa8;margin-top:5px;">
                    This message exhibits spam characteristics. Do not click any links.
                  </div>
                </div>
              </div>

              <div style="margin-bottom:18px;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                  font-family:'DM Mono',monospace;font-size:12px;color:#6a7fa8;margin-bottom:8px;">
                  <span>Confidence Score</span>
                  <span style="color:#ff3f5e;font-weight:600;font-size:14px;">95%</span>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                  <div style="height:100%;width:95%;border-radius:99px;
                    background:linear-gradient(90deg,#ff3f5e,#ff8fa3);
                    transition:width 1s ease;"></div>
                </div>
              </div>

              <div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a7fa8;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;opacity:0.7;">
                  Extracted Key Tokens
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                  {''.join([
                    f'<span style="font-family:DM Mono,monospace;font-size:12px;padding:4px 10px;'
                    f'border-radius:6px;background:rgba(255,63,94,0.12);'
                    f'border:1px solid rgba(255,63,94,0.25);color:#ff3f5e;">{tok}</span>'
                    for tok in tokens
                  ])}
                </div>
              </div>
            </div>
            <style>
            @keyframes fadeSlide {{
              from {{ opacity:0; transform:translateY(-8px) scaleY(0.95); }}
              to   {{ opacity:1; transform:translateY(0)   scaleY(1); }}
            }}
            </style>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="
              background:linear-gradient(135deg,rgba(0,229,160,0.09),rgba(0,229,160,0.04));
              border:1px solid rgba(0,229,160,0.28);
              border-radius:20px; padding:28px 28px 24px;
              margin-top:16px;
              box-shadow:0 4px 40px rgba(0,229,160,0.13);
              position:relative; overflow:hidden;
              animation: fadeSlide 0.4s ease forwards;
            ">
              <div style="position:absolute;top:0;left:0;right:0;height:3px;
                background:linear-gradient(90deg,transparent,#00e5a0,transparent);"></div>

              <div style="display:flex;align-items:center;gap:18px;margin-bottom:22px;">
                <div style="
                  width:60px;height:60px;border-radius:50%;flex-shrink:0;
                  background:rgba(0,229,160,0.10); border:1px solid rgba(0,229,160,0.28);
                  display:flex;align-items:center;justify-content:center;font-size:28px;
                  box-shadow:0 0 28px rgba(0,229,160,0.20);
                ">✅</div>
                <div>
                  <div style="font-size:30px;font-weight:800;letter-spacing:-1px;color:#00e5a0;line-height:1;">LEGITIMATE (HAM)</div>
                  <div style="font-family:'DM Mono',monospace;font-size:12px;color:#6a7fa8;margin-top:5px;">
                    This message appears to be a genuine communication.
                  </div>
                </div>
              </div>

              <div style="margin-bottom:18px;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                  font-family:'DM Mono',monospace;font-size:12px;color:#6a7fa8;margin-bottom:8px;">
                  <span>Confidence Score</span>
                  <span style="color:#00e5a0;font-weight:600;font-size:14px;">92%</span>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                  <div style="height:100%;width:92%;border-radius:99px;
                    background:linear-gradient(90deg,#00e5a0,#80ffcf);
                    transition:width 1s ease;"></div>
                </div>
              </div>

              <div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6a7fa8;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;opacity:0.7;">
                  Extracted Key Tokens
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">
                  {''.join([
                    f'<span style="font-family:DM Mono,monospace;font-size:12px;padding:4px 10px;'
                    f'border-radius:6px;background:rgba(0,229,160,0.09);'
                    f'border:1px solid rgba(0,229,160,0.22);color:#00e5a0;">{tok}</span>'
                    for tok in tokens
                  ])}
                </div>
              </div>
            </div>
            <style>
            @keyframes fadeSlide {{
              from {{ opacity:0; transform:translateY(-8px) scaleY(0.95); }}
              to   {{ opacity:1; transform:translateY(0)   scaleY(1); }}
            }}
            </style>
            """, unsafe_allow_html=True)

# ── INFO CARDS ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;">
  <div style="background:#0c1120;border:1px solid rgba(99,179,255,0.12);
    border-radius:16px;padding:20px;position:relative;overflow:hidden;">
    <div style="font-size:22px;margin-bottom:10px;">🔬</div>
    <div style="font-size:13px;font-weight:600;margin-bottom:5px;color:#e8f0ff;">NLP Pipeline</div>
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6a7fa8;line-height:1.6;">
      Lowercase → Tokenize → Stopword removal → Porter Stem → TF-IDF (3,000 features)
    </div>
    <div style="position:absolute;bottom:0;left:0;right:0;height:1px;
      background:linear-gradient(90deg,transparent,rgba(59,127,255,0.2),transparent);"></div>
  </div>
  <div style="background:#0c1120;border:1px solid rgba(99,179,255,0.12);
    border-radius:16px;padding:20px;position:relative;overflow:hidden;">
    <div style="font-size:22px;margin-bottom:10px;">🧠</div>
    <div style="font-size:13px;font-weight:600;margin-bottom:5px;color:#e8f0ff;">Classifier</div>
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6a7fa8;line-height:1.6;">
      Multinomial Naive Bayes with Laplace smoothing (α=1.0). Serialized via pickle.
    </div>
    <div style="position:absolute;bottom:0;left:0;right:0;height:1px;
      background:linear-gradient(90deg,transparent,rgba(162,89,255,0.2),transparent);"></div>
  </div>
  <div style="background:#0c1120;border:1px solid rgba(99,179,255,0.12);
    border-radius:16px;padding:20px;position:relative;overflow:hidden;">
    <div style="font-size:22px;margin-bottom:10px;">📦</div>
    <div style="font-size:13px;font-weight:600;margin-bottom:5px;color:#e8f0ff;">Dataset</div>
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6a7fa8;line-height:1.6;">
      UCI SMS Spam Collection — 4,825 ham (86.6%) and 747 spam (13.4%) messages.
    </div>
    <div style="position:absolute;bottom:0;left:0;right:0;height:1px;
      background:linear-gradient(90deg,transparent,rgba(0,229,192,0.2),transparent);"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:48px;padding-bottom:24px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;color:rgba(106,127,168,0.45);letter-spacing:1px;">
    Department of Computer Science · University of Agriculture, Faisalabad
  </div>
</div>
""", unsafe_allow_html=True)
