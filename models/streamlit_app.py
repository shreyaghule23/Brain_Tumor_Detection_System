import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0a0e1a;color:#e8eaf6;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2rem!important;max-width:860px!important;}
.hero{text-align:center;padding:2.5rem 1rem 1.5rem;}
.badge{display:inline-block;background:rgba(79,156,249,0.12);border:1px solid rgba(79,156,249,0.35);color:#4f9cf9;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;padding:.3rem 1rem;border-radius:100px;margin-bottom:1rem;}
.title{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#fff,#a5c8ff,#4f9cf9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .5rem;}
.sub{color:#8892a4;font-size:.95rem;}
.card{background:linear-gradient(145deg,#111827,#0d1525);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:1.5rem;margin-bottom:1rem;}
.clabel{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#4f9cf9;margin-bottom:.4rem;}
.cval{font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;color:#fff;}
.pill{display:inline-block;background:rgba(125,211,176,0.12);border:1px solid rgba(125,211,176,0.3);color:#7dd3b0;font-size:.78rem;padding:.25rem .8rem;border-radius:100px;margin-top:.5rem;}
.stitle{font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;color:#555f70;margin:1.2rem 0 .5rem;}
hr{border:none!important;border-top:1px solid rgba(255,255,255,0.07)!important;margin:1.4rem 0!important;}
[data-testid="stImage"] img{border-radius:14px!important;box-shadow:0 8px 40px rgba(0,0,0,.5)!important;}
[data-testid="stFileUploadDropzone"]{background:rgba(79,156,249,0.04)!important;border:1.5px dashed rgba(79,156,249,0.25)!important;border-radius:14px!important;color:#8892a4!important;}
.footer{text-align:center;color:#3d4758;font-size:.78rem;padding:1rem 0;}
.fname{color:#4f9cf9;font-weight:500;}
</style>""", unsafe_allow_html=True)

CLASSES     = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
COLORS      = ["#ef5757", "#f4a261", "#7dd3b0", "#4f9cf9"]
ICONS       = ["⚠️", "🔶", "✅", "🔵"]
FOOTER_HTML = '<div class="footer">Created by <span class="fname">Shreya Ghule</span></div>'

@st.cache_resource
def load(): return load_model("models/vgg16_best.h5")
model = load()

st.markdown("""<div class="hero">
  <div class="badge">VGG16 · TensorFlow · 4-Class Detection</div>
  <div class="title">🧠 Brain Tumor MRI Classifier</div>
  <div class="sub">Upload an MRI scan for instant AI-powered analysis</div>
</div><hr>""", unsafe_allow_html=True)

uploaded = st.file_uploader("📂 Drop your MRI scan here (JPG / PNG)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    a = np.array(img)
    if np.mean(np.abs(a[:,:,0].astype(int)-a[:,:,1].astype(int))) > 10:
        st.error("⚠️ Please upload a real MRI scan, not a regular photo."); st.stop()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    inp  = np.expand_dims(np.array(img.resize((224,224)))/255.0, 0)
    pred = model.predict(inp, verbose=0)[0]
    idx  = int(np.argmax(pred))

    with col2:
        st.markdown(f"""<div class="stitle">Diagnosis Result</div>
        <div class="card">
          <div class="clabel">Detected Class</div>
          <div class="cval">{ICONS[idx]} {CLASSES[idx]}</div>
          <div class="pill">🎯 {pred[idx]*100:.1f}% confidence</div>
        </div>
        <div class="stitle">All Probabilities</div>""", unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=[p*100 for p in pred], y=CLASSES, orientation="h",
            marker_color=COLORS, text=[f"{p*100:.1f}%" for p in pred],
            textposition="outside", textfont=dict(color="#8892a4", size=11)
        ))
        fig.update_layout(
            margin=dict(l=0,r=40,t=10,b=0), height=185,
            xaxis=dict(range=[0,110], gridcolor="rgba(255,255,255,0.04)",
                       tickfont=dict(color="#4a5568",size=9), showline=False),
            yaxis=dict(tickfont=dict(color="#c0c8d8",size=12), showline=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr><div class="footer">⚠️ For research use only — not a medical diagnosis tool.<br>'
                'Created by <span class="fname">Shreya Ghule</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center;padding:2.5rem;color:#3d4758">'
                '<div style="font-size:3rem">🩻</div>Upload an MRI scan above to begin</div>'
                + FOOTER_HTML, unsafe_allow_html=True)
