import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, time

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp { background: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1423 !important;
    border-right: 1px solid #1e2740;
}
[data-testid="stSidebar"] * { color: #c8d0e7 !important; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 15px;
    padding: 0.6rem 2rem;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.35);
}

/* Number inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #111827 !important;
    border: 1px solid #1e2740 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { padding: 0; }

/* Text colors */
h1, h2, h3, h4, p, label, span { color: #e2e8f0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e2740;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    border-radius: 7px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: white !important;
}

/* Alert boxes */
.fraud-alert {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.safe-alert {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 1px solid #22c55e;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.alert-icon { font-size: 3rem; }
.alert-title { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; font-family: 'Syne', sans-serif; }
.alert-sub { font-size: 0.9rem; opacity: 0.8; font-family: 'DM Mono', monospace; }

/* Score bar container */
.score-bar-wrap {
    background: #111827;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}

/* Section labels */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace;
}

/* Divider */
hr { border-color: #1e2740; }

/* Expander */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* Info box */
.info-box {
    background: #0c1a3a;
    border: 1px solid #1e3a6e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 13px;
    color: #7dd3fc;
    font-family: 'DM Mono', monospace;
    line-height: 1.7;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #1e2740;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path   = r"D:\Training\Mindset\Graduation project\fraud_model_xgb.json"
    scaler_a_path = r"D:\Training\Mindset\Graduation project\scaler_amount.pkl"
    scaler_t_path = r"D:\Training\Mindset\Graduation project\scaler_time.pkl"

    missing = [p for p in [model_path, scaler_a_path, scaler_t_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    scaler_amount = joblib.load(scaler_a_path)
    scaler_time   = joblib.load(scaler_t_path)
    return model, scaler_amount, scaler_time, []

model, scaler_amount, scaler_time, missing_files = load_model()


# ── HELPERS ───────────────────────────────────────────────────────────────────
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']

def predict_transaction(features_dict):
    row = pd.DataFrame([features_dict])
    row['Amount_scaled'] = scaler_amount.transform([[row['Amount'][0]]])[0][0]
    row['Time_scaled']   = scaler_time.transform([[row['Time'][0]]])[0][0]
    row = row.drop(['Amount', 'Time'], axis=1)
    row = row[FEATURE_COLS]
    proba = model.predict_proba(row)[0][1]
    pred  = int(proba >= 0.5)
    return proba, pred

def risk_label(p):
    if p < 0.2:  return "Very Low",  "#22c55e"
    if p < 0.4:  return "Low",       "#84cc16"
    if p < 0.6:  return "Moderate",  "#f59e0b"
    if p < 0.8:  return "High",      "#f97316"
    return "Critical",               "#ef4444"

def gauge_html(proba):
    pct   = proba * 100
    label, color = risk_label(proba)
    angle = -90 + (proba * 180)
    return f"""
    <div style="text-align:center; padding: 0.5rem 0;">
      <svg viewBox="0 0 200 110" width="240" style="overflow:visible;">
        <defs>
          <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#22c55e"/>
            <stop offset="40%"  stop-color="#f59e0b"/>
            <stop offset="75%"  stop-color="#f97316"/>
            <stop offset="100%" stop-color="#ef4444"/>
          </linearGradient>
        </defs>
        <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#1e2740" stroke-width="16" stroke-linecap="round"/>
        <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="url(#gauge-grad)" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="{283*(1-proba)}"/>
        <g transform="rotate({angle}, 100, 100)">
          <line x1="100" y1="100" x2="100" y2="22" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="100" cy="100" r="5" fill="white"/>
        </g>
        <text x="100" y="86" text-anchor="middle" fill="{color}" font-size="22" font-weight="800" font-family="Syne,sans-serif">{pct:.1f}%</text>
        <text x="100" y="102" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="DM Mono,monospace">{label} Risk</text>
      </svg>
    </div>"""


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Single Transaction", "Batch Analysis", "Model Info"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-label">Model Status</div>', unsafe_allow_html=True)
    if model is not None:
        st.success("✅ Model loaded")
        st.markdown('<div class="info-box">XGBoost · Cost-Sensitive<br>Time-Based Split<br>AUPRC optimized</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ Model not found")
        st.markdown(f'<div class="info-box">Missing files:<br>{"<br>".join(missing_files)}</div>', unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
  <h1 style="font-size: 2.4rem; font-weight: 800; color: #f1f5f9; margin:0; letter-spacing:-0.02em;">
    Credit Card Fraud Detection
  </h1>
  <p style="color:#475569; font-family:'DM Mono',monospace; font-size:13px; margin-top:0.4rem;">
    Powered by XGBoost · Cost-Sensitive · Time-Based Evaluation
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"Model files not found. Please copy `fraud_model_xgb.json`, `scaler_amount.pkl`, and `scaler_time.pkl` into the same directory as `app.py`.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Single Transaction":

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Transaction Details</div>', unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount ($)", min_value=0.01, max_value=50000.0, value=120.50, step=0.01, format="%.2f")
            time_s = c2.number_input("Time (seconds)", min_value=0, max_value=172800, value=43200)

        st.markdown('<div class="section-label" style="margin-top:1rem;">PCA Features (V1 – V28)</div>', unsafe_allow_html=True)
        st.caption("These are the anonymized PCA-transformed features from the original dataset.")

        v_values = {}
        cols_per_row = 4
        v_names = [f'V{i}' for i in range(1, 29)]
        rows = [v_names[i:i+cols_per_row] for i in range(0, len(v_names), cols_per_row)]
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, vname in zip(cols, row):
                v_values[vname] = col.number_input(vname, value=0.0, format="%.4f",
                                                    label_visibility="visible",
                                                    key=f"input_{vname}")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Analyze Transaction", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

        if predict_btn:
            features = {'Amount': amount, 'Time': time_s, **v_values}

            with st.spinner("Analyzing..."):
                time.sleep(0.4)
                proba, pred = predict_transaction(features)

            label, color = risk_label(proba)

            if pred == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                  <div class="alert-icon">🚨</div>
                  <div class="alert-title" style="color:#fca5a5;">FRAUD DETECTED</div>
                  <div class="alert-sub" style="color:#fca5a5;">This transaction has been flagged for review</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                  <div class="alert-icon">✅</div>
                  <div class="alert-title" style="color:#86efac;">TRANSACTION SAFE</div>
                  <div class="alert-sub" style="color:#86efac;">No fraudulent patterns detected</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(gauge_html(proba), unsafe_allow_html=True)

            st.markdown('<div class="score-bar-wrap">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Score",    f"{proba*100:.1f}%")
            c2.metric("Risk Level",     label)
            c3.metric("Decision",       "🚨 Flag" if pred else "✅ Pass")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Feature Contribution (Top 10)"):
                fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
                top10 = fi.nlargest(10).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#111827')
                bars = ax.barh(top10.index, top10.values,
                               color=['#ef4444' if f.startswith('V') else '#3b82f6' for f in top10.index],
                               height=0.6)
                ax.tick_params(colors='#94a3b8', labelsize=10)
                ax.spines[:].set_visible(False)
                ax.set_xlabel("Importance", color='#475569', fontsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1e2740')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        else:
            st.markdown("""
            <div style="height:400px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        background:#111827; border:1px dashed #1e2740;
                        border-radius:14px; color:#475569; font-family:'DM Mono',monospace;">
              <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
              <div>Fill in the form and click</div>
              <div style="font-weight:600; color:#3b82f6; margin-top:4px;">Analyze Transaction</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":

    st.markdown('<div class="section-label">Batch Transaction Scoring</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with columns: `Time`, `V1`–`V28`, `Amount` (optionally `Class` for evaluation).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_batch):,} transactions")

        required = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = [c for c in required if c not in df_batch.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            df_pred = df_batch.copy()
            df_pred['Amount_scaled'] = scaler_amount.transform(df_pred[['Amount']])
            df_pred['Time_scaled']   = scaler_time.transform(df_pred[['Time']])
            X_batch = df_pred[FEATURE_COLS]

            with st.spinner("Scoring transactions..."):
                probas = model.predict_proba(X_batch)[:, 1]
                preds  = (probas >= 0.5).astype(int)

            df_batch['fraud_score'] = probas
            df_batch['prediction']  = preds
            df_batch['risk_level']  = [risk_label(p)[0] for p in probas]

            # Summary metrics
            n_fraud = preds.sum()
            n_total = len(preds)
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", f"{n_total:,}")
            c2.metric("Flagged as Fraud",   f"{n_fraud:,}")
            c3.metric("Fraud Rate",         f"{n_fraud/n_total*100:.2f}%")
            c4.metric("Avg Fraud Score",    f"{probas.mean()*100:.1f}%")

            tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📋 Results Table", "📈 Score Analysis"])

            with tab1:
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                fig.patch.set_facecolor('#0a0e1a')
                for ax in axes:
                    ax.set_facecolor('#111827')
                    ax.tick_params(colors='#94a3b8')
                    for s in ax.spines.values():
                        s.set_edgecolor('#1e2740')

                # Score distribution
                axes[0].hist(probas[preds==0], bins=40, alpha=0.8, color='#22c55e', label='Safe', density=True)
                axes[0].hist(probas[preds==1], bins=40, alpha=0.8, color='#ef4444', label='Fraud', density=True)
                axes[0].set_title('Fraud Score Distribution', color='#e2e8f0', fontsize=11)
                axes[0].set_xlabel('Fraud Score', color='#475569', fontsize=9)
                axes[0].legend(facecolor='#1e2740', edgecolor='#1e2740', labelcolor='#e2e8f0')

                # Risk level breakdown
                risk_counts = df_batch['risk_level'].value_counts()
                risk_order  = ['Very Low','Low','Moderate','High','Critical']
                risk_colors = ['#22c55e','#84cc16','#f59e0b','#f97316','#ef4444']
                vals   = [risk_counts.get(r, 0) for r in risk_order]
                colors = [c for r, c in zip(risk_order, risk_colors) if risk_counts.get(r, 0) > 0]
                lbls   = [r for r in risk_order if risk_counts.get(r, 0) > 0]
                vvals  = [v for v in vals if v > 0]
                axes[1].pie(vvals, labels=lbls, colors=colors, autopct='%1.1f%%',
                            textprops={'color':'#e2e8f0', 'fontsize': 9},
                            wedgeprops={'linewidth': 1, 'edgecolor': '#0a0e1a'})
                axes[1].set_title('Risk Level Breakdown', color='#e2e8f0', fontsize=11)

                # Amount distribution by prediction
                if 'Amount' in df_batch.columns:
                    axes[2].hist(df_batch[df_batch['prediction']==0]['Amount'], bins=40,
                                 alpha=0.8, color='#22c55e', label='Safe', density=True)
                    axes[2].hist(df_batch[df_batch['prediction']==1]['Amount'], bins=20,
                                 alpha=0.9, color='#ef4444', label='Fraud', density=True)
                    axes[2].set_title('Amount by Prediction', color='#e2e8f0', fontsize=11)
                    axes[2].set_xlabel('Amount ($)', color='#475569', fontsize=9)
                    axes[2].legend(facecolor='#1e2740', edgecolor='#1e2740', labelcolor='#e2e8f0')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # If true labels present
                if 'Class' in df_batch.columns:
                    from sklearn.metrics import classification_report, confusion_matrix
                    y_true = df_batch['Class'].values
                    st.markdown("---")
                    st.markdown('<div class="section-label">Evaluation vs True Labels</div>', unsafe_allow_html=True)
                    report = classification_report(y_true, preds, target_names=['Legitimate','Fraud'], output_dict=True)
                    rep_df = pd.DataFrame(report).T.round(3)
                    st.dataframe(rep_df, use_container_width=True)

            with tab2:
                display_cols = ['Amount', 'fraud_score', 'risk_level', 'prediction']
                if 'Class' in df_batch.columns:
                    display_cols.append('Class')
                st.dataframe(
                    df_batch[display_cols].style
                        .background_gradient(subset=['fraud_score'], cmap='RdYlGn_r')
                        .format({'fraud_score': '{:.3f}', 'Amount': '${:.2f}'}),
                    use_container_width=True, height=400
                )

                csv = df_batch.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", csv,
                                   "fraud_predictions.csv", "text/csv",
                                   use_container_width=True)

            with tab3:
                fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
                fig2.patch.set_facecolor('#0a0e1a')
                for ax in axes2:
                    ax.set_facecolor('#111827')
                    ax.tick_params(colors='#94a3b8')
                    for s in ax.spines.values():
                        s.set_edgecolor('#1e2740')

                # Cumulative score
                sorted_scores = np.sort(probas)[::-1]
                axes2[0].plot(np.arange(len(sorted_scores)), sorted_scores,
                              color='#3b82f6', lw=2)
                axes2[0].axhline(0.5, color='#ef4444', linestyle='--', lw=1, label='Decision threshold (0.5)')
                axes2[0].fill_between(np.arange(len(sorted_scores)), sorted_scores, 0.5,
                                      where=sorted_scores > 0.5, alpha=0.2, color='#ef4444')
                axes2[0].set_title('Score Rank Chart', color='#e2e8f0', fontsize=11)
                axes2[0].set_xlabel('Transaction Rank', color='#475569', fontsize=9)
                axes2[0].set_ylabel('Fraud Score', color='#475569', fontsize=9)
                axes2[0].legend(facecolor='#1e2740', edgecolor='#1e2740', labelcolor='#e2e8f0', fontsize=9)

                # Score histogram with threshold
                axes2[1].hist(probas, bins=60, color='#3b82f6', alpha=0.8, edgecolor='#0a0e1a')
                axes2[1].axvline(0.5, color='#ef4444', lw=2, linestyle='--', label='Threshold = 0.5')
                axes2[1].set_title('Score Histogram', color='#e2e8f0', fontsize=11)
                axes2[1].set_xlabel('Fraud Score', color='#475569', fontsize=9)
                axes2[1].set_ylabel('Count', color='#475569', fontsize=9)
                axes2[1].legend(facecolor='#1e2740', edgecolor='#1e2740', labelcolor='#e2e8f0', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

    else:
        st.markdown("""
        <div style="height:250px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    background:#111827; border:1px dashed #1e2740;
                    border-radius:14px; color:#475569; font-family:'DM Mono',monospace;">
          <div style="font-size:2.5rem; margin-bottom:1rem;">📂</div>
          <div>Upload a CSV to begin batch analysis</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Info":

    st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1e2740; border-radius:12px; padding:1.5rem;">
          <h3 style="color:#60a5fa; margin-top:0;">XGBoost Classifier</h3>
          <table style="width:100%; font-family:'DM Mono',monospace; font-size:13px; color:#94a3b8;">
            <tr><td style="padding:5px 0; color:#475569;">Algorithm</td><td style="color:#e2e8f0;">Gradient Boosted Trees</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Estimators</td><td style="color:#e2e8f0;">200</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Learning Rate</td><td style="color:#e2e8f0;">0.05</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Max Depth</td><td style="color:#e2e8f0;">6</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Imbalance Handling</td><td style="color:#22c55e;">Cost-Sensitive (scale_pos_weight)</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Split Strategy</td><td style="color:#e2e8f0;">Time-Based (80/20)</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Eval Metric</td><td style="color:#e2e8f0;">AUPRC</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1e2740; border-radius:12px; padding:1.5rem;">
          <h3 style="color:#60a5fa; margin-top:0;">Dataset & Features</h3>
          <table style="width:100%; font-family:'DM Mono',monospace; font-size:13px; color:#94a3b8;">
            <tr><td style="padding:5px 0; color:#475569;">Source</td><td style="color:#e2e8f0;">ULB Credit Card Dataset</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Transactions</td><td style="color:#e2e8f0;">284,807</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Fraud Rate</td><td style="color:#e2e8f0;">0.172%</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Features</td><td style="color:#e2e8f0;">V1–V28 (PCA) + Amount + Time</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Amount</td><td style="color:#e2e8f0;">StandardScaled</td></tr>
            <tr><td style="padding:5px 0; color:#475569;">Time</td><td style="color:#e2e8f0;">StandardScaled</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)

    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#111827')

    colors = ['#ef4444' if v > fi.quantile(0.8) else '#3b82f6' if v > fi.quantile(0.5) else '#1e3a6e'
              for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors, height=0.7)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.set_xlabel('Feature Importance Score', color='#475569', fontsize=10)
    ax.set_title('All Feature Importances', color='#e2e8f0', fontsize=13, pad=12)
    for s in ax.spines.values():
        s.set_visible(False)

    patches = [
        mpatches.Patch(color='#ef4444', label='Top 20%'),
        mpatches.Patch(color='#3b82f6', label='Top 50%'),
        mpatches.Patch(color='#1e3a6e', label='Lower 50%'),
    ]
    ax.legend(handles=patches, facecolor='#111827', edgecolor='#1e2740',
              labelcolor='#e2e8f0', fontsize=10, loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <b style="color:#60a5fa;">Single Transaction</b> — Enter V1–V28 PCA features, Amount, and Time to score one transaction in real time.<br><br>
      <b style="color:#60a5fa;">Batch Analysis</b> — Upload a CSV with columns Time, V1–V28, Amount.
      Optionally include a <code>Class</code> column (0/1) for automatic evaluation metrics.<br><br>
      <b style="color:#60a5fa;">Fraud Score</b> — The model outputs a probability [0–1]. Scores ≥ 0.5 are flagged as fraud.
      Adjust the threshold in production based on your cost tolerance (FP vs FN tradeoff).
    </div>""", unsafe_allow_html=True)