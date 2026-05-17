import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# AI utilities
try:
    from ai_utils import (
        generate_spending_insights,
        predict_monthly_spending,
        detect_anomalies_ai,
        generate_budget_recommendations,
        analyze_subscriptions_ai,
        generate_financial_health_assessment,
        generate_smart_bill_reminders
    )
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False

st.set_page_config(
    page_title="Financial Subscription Manager AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg-base:       #0f1117;
  --bg-surface:    #161b27;
  --bg-card:       #1c2336;
  --bg-card-hover: #202840;
  --bg-input:      #141929;
  --border:        rgba(255,255,255,0.07);
  --border-focus:  rgba(99,179,237,0.45);
  --text-primary:  #eef0f6;
  --text-secondary:#8b92a9;
  --text-muted:    #555f7a;
  --accent-blue:   #4a90d9;
  --accent-teal:   #38b2ac;
  --accent-gold:   #c9a84c;
  --accent-red:    #e05c6a;
  --accent-green:  #48bb78;
  --radius-card:   14px;
  --radius-sm:     8px;
  --shadow-card:   0 2px 12px rgba(0,0,0,0.35);
  --font-sans:     'DM Sans', sans-serif;
  --font-mono:     'DM Mono', monospace;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
footer { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Base ── */
html, body, [class*="css"] { font-family: var(--font-sans) !important; color: var(--text-primary); }
.stApp { background-color: var(--bg-base) !important; }
.block-container { padding: 28px 36px 48px !important; max-width: 1300px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebarContent"] { padding-top: 0 !important; }

/* Sidebar brand */
.sidebar-brand {
  padding: 36px 18px 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 4px;
}
.sidebar-brand .brand-icon {
  width: 54px; height: 54px;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 16px;
  font-size: 26px;
  font-weight: 800;
  color: #fff;
  box-shadow: 0 6px 18px rgba(74,144,217,0.4);
  letter-spacing: -1px;
}
.sidebar-brand h1 {
  font-size: 24px !important;
  font-weight: 800 !important;
  letter-spacing: -0.6px;
  color: var(--text-primary) !important;
  margin: 0 !important;
  line-height: 1.2 !important;
}
.sidebar-brand h1 span.highlight {
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sidebar-brand p {
  font-size: 11.5px !important;
  color: var(--text-muted) !important;
  margin: 8px 0 0 !important;
  letter-spacing: 0.9px;
  text-transform: uppercase;
}

/* Section labels */
.sidebar-section-label {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--text-muted) !important;
  padding: 16px 22px 6px;
  display: block;
}

/* Dividers */
.sidebar-divider { height: 1px; background: var(--border); margin: 10px 16px; }

/* Status badge */
.sidebar-status {
  margin: 6px 14px 2px;
  padding: 8px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}
.sidebar-status .status-dot {
  display: inline-block; width: 6px; height: 6px;
  border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.sidebar-status .status-dot.green { background: var(--accent-green); box-shadow: 0 0 5px var(--accent-green); }
.sidebar-status .status-dot.blue  { background: var(--accent-blue); }

/* ── Radio — hide native widget label, show only text ── */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div { margin-top: 0 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  display: flex !important; flex-direction: column !important; gap: 1px !important;
  padding: 0 8px !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
  padding: 9px 14px !important;
  border-radius: var(--radius-sm) !important;
  cursor: pointer !important;
  transition: background 0.15s ease !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  display: flex !important;
  align-items: center !important;
  background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
  background: rgba(255,255,255,0.05) !important;
  color: var(--text-primary) !important;
}
/* hide the radio circle dot */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
  display: none !important;
}
/* active state — selected item */
[data-testid="stSidebar"] .stRadio input:checked + div {
  color: var(--accent-blue) !important;
}

/* ── KPI Cards ── */
.kpi-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-card); padding: 20px 20px 16px;
  box-shadow: var(--shadow-card); position: relative; overflow: hidden;
  margin-bottom: 4px; transition: border-color 0.2s, transform 0.15s;
}
.kpi-card:hover { border-color: rgba(255,255,255,0.14); transform: translateY(-1px); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-card.blue::before  { background: linear-gradient(90deg, var(--accent-blue), transparent); }
.kpi-card.teal::before  { background: linear-gradient(90deg, var(--accent-teal), transparent); }
.kpi-card.gold::before  { background: linear-gradient(90deg, var(--accent-gold), transparent); }
.kpi-card.red::before   { background: linear-gradient(90deg, var(--accent-red), transparent); }
.kpi-card.green::before { background: linear-gradient(90deg, var(--accent-green), transparent); }
.kpi-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.9px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }
.kpi-value { font-size: 24px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.5px; line-height: 1.1; font-family: var(--font-mono) !important; }
.kpi-delta { font-size: 11.5px; color: var(--text-muted); margin-top: 6px; }

/* ── Section cards ── */
.section-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-card); padding: 22px 24px;
  box-shadow: var(--shadow-card); margin-bottom: 20px;
}
.section-card-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--border);
}
.section-card-title { font-size: 14px; font-weight: 600; color: var(--text-primary); letter-spacing: -0.1px; }
.section-card-subtitle { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; }

/* ── Status boxes ── */
.status-box {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 13px 16px; border-radius: var(--radius-sm);
  margin: 10px 0; font-size: 13px; line-height: 1.6;
}
.status-box.danger  { background: rgba(224,92,106,0.08); border: 1px solid rgba(224,92,106,0.28); color: #f0858f; }
.status-box.warning { background: rgba(201,168,76,0.08); border: 1px solid rgba(201,168,76,0.28); color: #d4a843; }
.status-box.success { background: rgba(72,187,120,0.08); border: 1px solid rgba(72,187,120,0.28); color: #5cc98c; }
.status-box.info    { background: rgba(74,144,217,0.08); border: 1px solid rgba(74,144,217,0.28); color: #76aee8; }
.status-box .icon   { font-size: 15px; flex-shrink: 0; margin-top: 2px; font-weight: 700; }

/* ── Page titles ── */
.page-title { font-size: 21px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.4px; margin-bottom: 4px; }
.page-subtitle { font-size: 12.5px; color: var(--text-muted); margin-bottom: 22px; }

/* ── Streamlit metric ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-card) !important; padding: 18px 20px !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stMetricLabel"] { font-size: 10.5px !important; font-weight: 600 !important; letter-spacing: 0.9px !important; text-transform: uppercase !important; color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 700 !important; font-family: var(--font-mono) !important; color: var(--text-primary) !important; letter-spacing: -0.4px !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important; border-radius: var(--radius-sm) !important;
  font-family: var(--font-sans) !important; font-size: 13px !important;
  font-weight: 500 !important; padding: 8px 18px !important; transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: var(--bg-card-hover) !important; border-color: rgba(255,255,255,0.15) !important;
  color: var(--text-primary) !important; transform: translateY(-1px) !important;
}

/* ── Form controls ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
  background: var(--bg-input) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important; color: var(--text-primary) !important; font-size: 13px !important;
}
[data-testid="stFileUploader"] section {
  background: var(--bg-input) !important; border: 1px dashed var(--border-focus) !important;
  border-radius: var(--radius-sm) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-secondary) !important; font-size: 13px !important; }

/* ── Progress ── */
.stProgress > div > div > div { background: var(--accent-blue) !important; border-radius: 4px !important; }
.stProgress > div > div { background: var(--bg-input) !important; border-radius: 4px !important; }

/* ── Misc ── */
hr { border-color: var(--border) !important; margin: 18px 0 !important; }
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; font-size: 13px !important; }
h1, h2, h3 { font-family: var(--font-sans) !important; color: var(--text-primary) !important; }
.stCaption, [data-testid="stCaption"] { color: var(--text-muted) !important; font-size: 11.5px !important; }
[data-testid="stDownloadButton"] > button {
  background: rgba(74,144,217,0.1) !important; border: 1px solid rgba(74,144,217,0.35) !important;
  color: var(--accent-blue) !important; border-radius: var(--radius-sm) !important;
  font-size: 13px !important; font-weight: 500 !important;
}
[data-testid="stDownloadButton"] > button:hover { background: rgba(74,144,217,0.18) !important; }
[data-testid="stPlotlyChart"] { border-radius: var(--radius-sm) !important; overflow: hidden !important; }
[data-testid="stCheckbox"] { color: var(--text-secondary) !important; font-size: 13px !important; }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ── Plotly base layout ──
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#8b92a9', size=12),
    margin=dict(l=16, r=16, t=16, b=16),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickcolor='rgba(255,255,255,0)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickcolor='rgba(255,255,255,0)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.08)'),
    hoverlabel=dict(bgcolor='#1c2336', font_size=12, font_family='DM Sans, sans-serif', bordercolor='rgba(255,255,255,0.15)'),
)

COLOR_PALETTE = ['#4a90d9','#38b2ac','#c9a84c','#e05c6a','#48bb78','#805ad5','#ed8936','#63b3ed','#68d391','#f6ad55']

# ============================================================
#  DATA LOADING & PROCESSING
# ============================================================

def load_sample_data():
    try:
        df = pd.read_csv('sample_transactions.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        if 'transaction_type' in df.columns:
            txn_type = df['transaction_type'].astype(str).str.strip().str.lower()
            income_mask = txn_type == 'income'
        else:
            income_mask = df['amount'] < 0
        df['amount'] = df['amount'].abs()
        df['transaction_type'] = np.where(income_mask, 'income', 'expense')
        df = df[df['amount'] > 0].sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("sample_transactions.csv not found. Please run create_excel.py first.")
        return None

def load_user_data(uploaded_file):
    """Robust file loader — handles varied column naming conventions."""
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith('.xlsx') or uploaded_file.name.lower().endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
            return None

        if df.empty:
            st.error("The uploaded file is empty.")
            return None

        # Normalise column names
        df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]', '_', regex=True)

        # ── DATE column ──
        date_candidates = [c for c in df.columns if any(k in c for k in ['date','time','when','day'])]
        date_col = date_candidates[0] if date_candidates else df.columns[0]
        df['date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        if df['date'].isna().all():
            # try every column
            for c in df.columns:
                attempt = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                if attempt.notna().sum() > len(df) * 0.5:
                    df['date'] = attempt
                    break
        df = df.dropna(subset=['date'])
        if df.empty:
            st.error("Could not parse any date values. Ensure your file has a column with dates.")
            return None

        # ── AMOUNT column ──
        amount_candidates = [c for c in df.columns if any(k in c for k in ['amount','sum','total','debit','credit','value','price','cost'])]
        if not amount_candidates:
            # pick first numeric column that is not the date col
            for c in df.columns:
                if c == 'date': continue
                try:
                    numeric_vals = pd.to_numeric(df[c], errors='coerce')
                    if numeric_vals.notna().sum() > len(df) * 0.5:
                        amount_candidates = [c]
                        break
                except Exception:
                    pass
        amount_col = amount_candidates[0] if amount_candidates else None
        if amount_col is None:
            st.error("Could not find an amount/value column in the file.")
            return None
        df['amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
        df = df[df['amount'] != 0]
        if df.empty:
            st.error("No valid transaction amounts found.")
            return None

        # ── MERCHANT / DESCRIPTION column ──
        merchant_candidates = [c for c in df.columns if any(k in c for k in ['merchant','description','desc','name','payee','vendor','note','memo','narration'])]
        merchant_col = merchant_candidates[0] if merchant_candidates else None
        if merchant_col:
            df['merchant_name'] = df[merchant_col].astype(str).str.strip()
        else:
            df['merchant_name'] = 'Unknown'

        # ── CATEGORY column ──
        cat_candidates = [c for c in df.columns if 'categ' in c or 'type' in c or 'tag' in c]
        if cat_candidates and cat_candidates[0] != 'transaction_type':
            df['category'] = df[cat_candidates[0]].astype(str).str.strip()
        else:
            df['category'] = df['merchant_name'].apply(auto_categorize)

        df['description'] = df['merchant_name']

        # ── TRANSACTION TYPE ──
        txn_candidates = [c for c in df.columns if 'transaction_type' in c or c in ['type','txn_type','kind']]
        if txn_candidates:
            txn_type = df[txn_candidates[0]].astype(str).str.strip().str.lower()
            income_mask = txn_type.isin(['income','credit','deposit','salary','revenue'])
        else:
            # negative = income convention
            income_mask = df['amount'] < 0

        df['amount'] = df['amount'].abs()
        df['transaction_type'] = np.where(income_mask, 'income', 'expense')

        df = df[['date','merchant_name','category','amount','description','transaction_type']].sort_values('date').reset_index(drop=True)
        df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
        return df

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def auto_categorize(merchant_name):
    merchant = str(merchant_name).lower()
    keywords = {
        'Groceries':    ['grocery','whole foods','kroger','safeway','trader','costco','sprouts'],
        'Food & Dining':['restaurant','starbucks','chipotle','pizza','sushi','mcdonald','cafe','burger','subway'],
        'Utilities':    ['electric','water','gas','internet','phone','utility'],
        'Rent':         ['rent','landlord','apartment','lease','management'],
        'Transportation':['uber','lyft','parking','transit','bus','shell','chevron'],
        'Entertainment':['netflix','spotify','hulu','cinema','movie','steam','disney'],
        'Shopping':     ['amazon','target','walmart','h&m','best buy','store'],
        'Healthcare':   ['pharmacy','cvs','walgreens','doctor','hospital','dental'],
        'Subscriptions':['subscription','gym','membership','adobe','microsoft'],
    }
    for category, kw_list in keywords.items():
        if any(kw in merchant for kw in kw_list):
            return category
    return 'Shopping'

def detect_fraud(df):
    if len(df) < 10:
        df = df.copy(); df['is_fraud'] = False; return df
    X = df[['amount']].copy()
    X['day_of_month'] = df['date'].dt.day
    X['day_of_week']  = df['date'].dt.dayofweek
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled)
    df_result = df.copy()
    df_result['is_fraud'] = predictions == -1
    return df_result

def detect_recurring_payments(df):
    if len(df) == 0: return pd.DataFrame()
    recurring = []
    for merchant in df['merchant_name'].unique():
        merchant_txns = df[df['merchant_name'] == merchant].sort_values('date')
        if len(merchant_txns) >= 2:
            amounts = merchant_txns['amount'].values
            amount_mean = amounts.mean()
            if amount_mean == 0: continue
            amount_cv = amounts.std() / amount_mean
            interval_cv = 0
            if len(merchant_txns) >= 3:
                dates = pd.to_datetime(merchant_txns['date']).values
                intervals = np.diff(dates).astype('timedelta64[D]').astype(int)
                if len(intervals) > 0 and intervals.mean() > 0:
                    interval_cv = intervals.std() / intervals.mean()
            if amount_cv <= 0.15 and interval_cv <= 0.3:
                recurring.append({
                    'merchant_name': merchant,
                    'category':      merchant_txns['category'].iloc[0],
                    'count':         len(merchant_txns),
                    'avg_amount':    round(amount_mean, 2),
                    'total_annual':  round(amount_mean * 12, 2),
                })
    return pd.DataFrame(recurring).sort_values('total_annual', ascending=False) if recurring else pd.DataFrame()

# ============================================================
#  SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="brand-icon">$</div>
  <h1><span class="highlight">Subscription</span><br>&amp; Financial AI</h1>
  <p>Analytics &amp; Insights</p>
</div>
""", unsafe_allow_html=True)

# ── Data Source ──
st.sidebar.markdown('<span class="sidebar-section-label">Data Source</span>', unsafe_allow_html=True)
data_source = st.sidebar.radio(
    "data_source_radio",
    ["Use Sample Data", "Upload Your Data"],
    label_visibility="collapsed",
    key="data_source"
)

df = None

if data_source == "Upload Your Data":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Needs columns for: date, amount, merchant/description. Category and transaction_type are optional.",
        key="file_upload"
    )
    if uploaded_file is not None:
        df = load_user_data(uploaded_file)
        if df is not None:
            st.sidebar.markdown(
                f'<div class="sidebar-status"><span class="status-dot green"></span>{len(df):,} transactions loaded successfully</div>',
                unsafe_allow_html=True
            )
    else:
        st.sidebar.markdown(
            '<div class="sidebar-status" style="color:var(--text-muted)">No file uploaded yet</div>',
            unsafe_allow_html=True
        )
else:
    df = load_sample_data()
    if df is not None:
        days = (df['date'].max() - df['date'].min()).days
        st.sidebar.markdown(
            f'<div class="sidebar-status"><span class="status-dot blue"></span>Sample data — {len(df):,} transactions, {days} days</div>',
            unsafe_allow_html=True
        )

if df is None or len(df) == 0:
    if data_source == "Upload Your Data":
        st.info("Upload a CSV or Excel file in the sidebar to get started.")
    else:
        st.error("Could not load sample data. Make sure sample_transactions.csv is in the same folder as app.py.")
    st.stop()

df_fraud     = detect_fraud(df)
recurring_df = detect_recurring_payments(df)

# ── Navigation ──
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<span class="sidebar-section-label">Navigation</span>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "navigation_radio",
    ["Dashboard", "Analytics", "Fraud Detection", "Recurring Payments",
     "Budget Planning", "Monthly Report", "Bill Reminders", "Reports & Export"],
    label_visibility="collapsed",
    key="navigation"
)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Footer stats
fraud_count_sidebar = int(df_fraud['is_fraud'].sum())
st.sidebar.markdown(f"""
<div style="padding:8px 16px 4px;">
  <div style="font-size:11px;color:var(--text-muted);line-height:2;">
    <div>Total transactions: <span style="color:var(--text-secondary);font-weight:600">{len(df):,}</span></div>
    <div>Flagged anomalies: <span style="color:{'#e05c6a' if fraud_count_sidebar > 0 else 'var(--text-secondary)'};font-weight:600">{fraud_count_sidebar}</span></div>
    <div>Subscriptions: <span style="color:var(--text-secondary);font-weight:600">{len(recurring_df)}</span></div>
  </div>
</div>
<div style="
  padding: 14px 18px 20px;
  margin-top: 6px;
  border-top: 1px solid rgba(255,255,255,0.06);
  text-align: center;
">
  <div style="
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.7;
    letter-spacing: 0.2px;
  ">
    Made with <span style="color:#e05c6a;font-size:13px;">&#9829;</span> by<br>
    <span style="
      color: var(--text-secondary);
      font-weight: 600;
      font-size: 12px;
      background: linear-gradient(90deg, #4a90d9, #38b2ac);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    ">Hunain Zain &amp; Shaheer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def card_open(title, subtitle=""):
    sub = f'<div class="section-card-subtitle">{subtitle}</div>' if subtitle else ''
    st.markdown(
        f'<div class="section-card"><div class="section-card-header"><div>' +
        f'<div class="section-card-title">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True
    )

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def status(type_, msg):
    icons = {'danger':'!','warning':'!','success':'+','info':'i'}
    st.markdown(f'<div class="status-box {type_}"><span class="icon">{icons[type_]}</span><div>{msg}</div></div>', unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def spacer(h=12):
    st.markdown(f'<div style="height:{h}px"></div>', unsafe_allow_html=True)

def plotly_fig(fig, height=300, **kwargs):
    layout = dict(**PLOTLY_LAYOUT)
    layout['height'] = height
    layout.update(kwargs)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
#  DASHBOARD
# ============================================================

if page == "Dashboard":
    page_header("Financial Dashboard", "Overview of your financial activity and key metrics")

    total_income   = df[df['transaction_type'] == 'income']['amount'].sum()
    total_spending = df[df['transaction_type'] == 'expense']['amount'].sum()
    balance        = total_income - total_spending
    fraud_alerts   = df_fraud['is_fraud'].sum()
    savings_rate   = (balance / total_income * 100) if total_income > 0 else 0

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, value, delta, color in [
        (col1, "Total Income",   f"${total_income:,.0f}",   f"{len(df[df['transaction_type']=='income'])} transactions", "teal"),
        (col2, "Total Expenses", f"${total_spending:,.0f}", f"{len(df[df['transaction_type']=='expense'])} transactions", "blue"),
        (col3, "Net Balance",    f"${balance:,.0f}",        "Income minus expenses", "gold" if balance >= 0 else "red"),
        (col4, "Savings Rate",   f"{savings_rate:.1f}%",    "Of total income saved", "green"),
        (col5, "Fraud Alerts",   str(int(fraud_alerts)),    f"{(fraud_alerts/len(df)*100):.1f}% flagged", "red" if fraud_alerts > 5 else "blue"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}"><div class="kpi-label">{label}</div>' +
                f'<div class="kpi-value">{value}</div><div class="kpi-delta">{delta}</div></div>',
                unsafe_allow_html=True
            )

    spacer(20)

    # Charts row
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        card_open("Monthly Spending Trend", "Expense totals per calendar month")
        expenses_only = df[df['transaction_type'] == 'expense'].copy()
        if len(expenses_only) > 0:
            # Build monthly series with proper datetime x-axis
            expenses_only['month'] = expenses_only['date'].dt.to_period('M').dt.to_timestamp()
            monthly = expenses_only.groupby('month')['amount'].sum().reset_index()
            monthly = monthly.sort_values('month')
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['month'], y=monthly['amount'],
                mode='lines+markers', fill='tozeroy',
                fillcolor='rgba(74,144,217,0.09)',
                line=dict(color='#4a90d9', width=2.5),
                marker=dict(size=7, color='#4a90d9', line=dict(color='#161b27', width=2)),
                hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.2f}<extra></extra>"
            ))
            plotly_fig(fig, height=290, showlegend=False,
                       xaxis_title="", yaxis_title="USD ($)",
                       xaxis=dict(gridcolor='rgba(255,255,255,0.05)', type='date'),
                       hovermode='x unified')
        card_close()

    with col2:
        card_open("Spending by Category", "Proportional expense breakdown")
        category_data = (df[df['transaction_type'] == 'expense']
                         .groupby('category')['amount'].sum()
                         .sort_values(ascending=False))
        if len(category_data) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=category_data.index, values=category_data.values, hole=0.52,
                marker=dict(colors=COLOR_PALETTE, line=dict(color='#1c2336', width=2)),
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f} — %{percent}<extra></extra>"
            )])
            plotly_fig(fig, height=290, showlegend=True,
                       legend=dict(orientation='v', x=1.0, y=0.5, font=dict(size=10)))
        card_close()

    # Recent transactions
    card_open("Recent Transactions", "Last 20 recorded entries")
    recent = df.tail(20)[['date','merchant_name','category','amount','transaction_type']].copy()
    recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
    recent.columns = ['Date','Merchant','Category','Amount ($)','Type']
    st.dataframe(recent, use_container_width=True, hide_index=True)
    card_close()

    # AI Insights
    card_open("AI-Powered Insights", "Intelligent analysis of your financial data")
    if AI_ENABLED:
        st.write("**Financial Health Score**")
        with st.spinner("Calculating..."):
            health = generate_financial_health_assessment(df, total_income, total_spending)
            score_text = health.get('_health_score', '0/100')
            try:   score_value = int(str(score_text).split('/')[0])
            except: score_value = 0
            score_value = max(0, min(100, score_value))
            score_label = "Strong" if score_value >= 75 else ("Moderate" if score_value >= 50 else "Needs Improvement")
            st.metric("Financial Health Score", f"{score_value}/100", delta=score_label)
            st.progress(score_value / 100)
            if '_assessment' in health:     st.caption(health['_assessment'])
            if '_recommendations' in health: status('info', health['_recommendations'])
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Spending Analysis**")
            with st.spinner("Analyzing..."): st.write(generate_spending_insights(df))
        with c2:
            st.write("**Spending Forecast**")
            with st.spinner("Forecasting..."):
                forecast = predict_monthly_spending(df, months_ahead=3)
                if 'error' not in forecast:
                    for month, amount in forecast.get('forecasts', {}).items():
                        if not str(month).startswith('_') and isinstance(amount, (int, float, np.integer, np.floating)):
                            st.write(f"- {month}: ${amount:,.0f}")
                    if 'confidence_interval' in forecast: st.caption(f"Confidence: {forecast['confidence_interval']}")
                    if 'trend_direction' in forecast:     st.caption(f"Trend: {forecast['trend_direction']}")
                    if '_interpretation' in forecast:     status('info', forecast['_interpretation'])
                else:
                    st.info("Insufficient data for forecast")
    else:
        status('info', 'AI features require an Anthropic API key. Set the <code>ANTHROPIC_API_KEY</code> environment variable to enable.')
    card_close()

# ============================================================
#  ANALYTICS
# ============================================================

elif page == "Analytics":
    page_header("Detailed Analytics", "Granular breakdowns of spending patterns and merchant activity")
    expenses_df = df[df['transaction_type'] == 'expense'].copy()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        card_open("Daily Spending Pattern", "Expense amount per day")
        daily = expenses_df.groupby(expenses_df['date'].dt.date)['amount'].sum().reset_index()
        daily.columns = ['date', 'amount']
        if len(daily) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily['date'], y=daily['amount'],
                marker_color='#4a90d9', marker_line_color='rgba(0,0,0,0)', opacity=0.85,
                hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"
            ))
            plotly_fig(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="USD ($)", hovermode='x')
        card_close()

    with col2:
        card_open("Top 10 Merchants", "Highest cumulative spend by vendor")
        top_merchants = expenses_df.groupby('merchant_name')['amount'].sum().nlargest(10).sort_values()
        if len(top_merchants) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_merchants.index, x=top_merchants.values, orientation='h',
                marker=dict(color=top_merchants.values, colorscale=[[0,'#243050'],[1,'#4a90d9']], line=dict(color='rgba(0,0,0,0)')),
                hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>"
            ))
            plotly_fig(fig, height=300, showlegend=False, xaxis_title="USD ($)",
                       yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='rgba(255,255,255,0.08)', tickcolor='rgba(255,255,255,0)'))
        card_close()

    card_open("Category Breakdown", "Expense statistics grouped by category")
    cat_stats = df[df['transaction_type'] == 'expense'].groupby('category').agg({'amount':['sum','count','mean']}).round(2)
    cat_stats.columns = ['Total ($)','Transactions','Avg ($)']
    cat_stats = cat_stats.sort_values('Total ($)', ascending=False)
    st.dataframe(cat_stats, use_container_width=True)
    card_close()

# ============================================================
#  FRAUD DETECTION
# ============================================================

elif page == "Fraud Detection":
    page_header("Fraud Detection & Security", "Isolation Forest anomaly detection on your transaction history")
    fraud_count = df_fraud['is_fraud'].sum()
    fraud_pct   = (fraud_count / len(df_fraud) * 100) if len(df_fraud) > 0 else 0

    if AI_ENABLED and st.checkbox("Enable AI Anomaly Analysis"):
        with st.spinner("Running AI anomaly detection..."):
            ai_anomalies = detect_anomalies_ai(df_fraud)
            if ai_anomalies:
                st.markdown('<div class="section-card-title" style="margin:16px 0 10px">AI-Detected Anomalies</div>', unsafe_allow_html=True)
                for anomaly in ai_anomalies[:5]:
                    c1, c2 = st.columns([3,2])
                    with c1:
                        st.write(f"**{anomaly['merchant']}** ({anomaly['category']})")
                        st.write(f"Date: {anomaly['date'].strftime('%Y-%m-%d')}")
                        if 'ai_assessment' in anomaly: status('warning', anomaly['ai_assessment'])
                    with c2:
                        st.metric("Amount", f"${anomaly['amount']:.2f}", delta="Anomaly")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Suspicious Transactions", int(fraud_count))
    with col2: st.metric("Fraud Rate", f"{fraud_pct:.1f}%")
    with col3: st.metric("Safe Transactions", int(len(df_fraud) - fraud_count))
    spacer(8)

    if fraud_pct > 10:
        status('danger',  'High fraud rate detected. Review your transactions immediately and contact your financial institution.')
    elif fraud_pct > 5:
        status('warning', 'Moderate fraud rate detected. Monitor your account activity closely.')
    else:
        status('success', 'Fraud rate is within normal range. Your account appears secure.')

    spacer(8)
    card_open("Flagged Transactions", "Transactions identified as anomalous by the model")
    fraud_txns = df_fraud[df_fraud['is_fraud']][['date','merchant_name','category','amount']].copy()
    if len(fraud_txns) > 0:
        fraud_txns['date'] = fraud_txns['date'].dt.strftime('%Y-%m-%d %H:%M')
        fraud_txns.columns = ['Date','Merchant','Category','Amount ($)']
        st.dataframe(fraud_txns, use_container_width=True, hide_index=True)
        status('warning', 'Review these transactions carefully. Contact your bank if any are unrecognized.')
    else:
        status('success', 'No suspicious transactions detected.')
    card_close()

# ============================================================
#  RECURRING PAYMENTS
# ============================================================

elif page == "Recurring Payments":
    page_header("Recurring Payments & Subscriptions", "Automatically detected regular charges and subscriptions")

    if len(recurring_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Active Subscriptions", len(recurring_df))
        with col2: st.metric("Monthly Cost", f"${recurring_df['avg_amount'].sum():,.2f}")
        with col3: st.metric("Annual Cost",  f"${recurring_df['total_annual'].sum():,.2f}")
        spacer()

        card_open("Your Subscriptions", "Sorted by annual cost")
        rd = recurring_df[['merchant_name','category','count','avg_amount','total_annual']].copy()
        rd.columns = ['Merchant','Category','Occurrences','Avg Amount ($)','Annual Cost ($)']
        st.dataframe(rd, use_container_width=True, hide_index=True)
        status('warning', f'Total yearly subscription cost: <strong>${recurring_df["total_annual"].sum():,.2f}</strong>')
        card_close()

        card_open("AI Subscription Analysis", "Optimization recommendations powered by AI")
        if AI_ENABLED:
            if st.button("Analyze Subscriptions with AI"):
                with st.spinner("Analyzing..."):
                    ai_analysis = analyze_subscriptions_ai(recurring_df)
                    if '_unused_services' in ai_analysis:      status('warning', f'<strong>Potentially Unused:</strong> {ai_analysis["_unused_services"]}')
                    if '_savings_opportunities' in ai_analysis: status('success', f'<strong>Savings Opportunities:</strong> {ai_analysis["_savings_opportunities"]}')
                    if '_alternatives' in ai_analysis:         status('info',    f'<strong>Alternatives:</strong> {ai_analysis["_alternatives"]}')
        else:
            status('info', 'AI analysis requires <code>ANTHROPIC_API_KEY</code> environment variable.')
        card_close()

        card_open("Subscription Audit Tips")
        st.write("- Which subscriptions do you actively use?")
        st.write("- Are there free alternatives available?")
        st.write("- Can you negotiate a lower rate or switch to an annual plan?")
        if len(recurring_df) > 3:
            potential = recurring_df.nlargest(3, 'total_annual')['total_annual'].sum() * 0.3
            status('info', f'Potential savings: ~<strong>${potential:,.2f}/year</strong> if you optimize your top 3 subscriptions.')
        card_close()
    else:
        status('info', 'No recurring payments detected. Upload data with recurring transactions to see this section.')

# ============================================================
#  BUDGET PLANNING
# ============================================================

elif page == "Budget Planning":
    page_header("Budget Planning", "Compare your actual spending against industry-standard budget allocations")
    expenses_df = df[df['transaction_type'] == 'expense'].copy()
    income_df   = df[df['transaction_type'] == 'income'].copy()
    total_income   = income_df['amount'].sum()
    total_expenses = expenses_df['amount'].sum()

    if total_income > 0:
        period_months       = max(1, (df['date'].max() - df['date'].min()).days / 30)
        avg_monthly_income  = total_income  / period_months
        avg_monthly_expenses = total_expenses / period_months
        savings_rate        = ((total_income - total_expenses) / total_income * 100)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Avg Monthly Income",   f"${avg_monthly_income:,.2f}")
        with col2: st.metric("Avg Monthly Spending", f"${avg_monthly_expenses:,.2f}")
        with col3: st.metric("Savings Rate",         f"{savings_rate:.1f}%")
        spacer()

        standard_budget = {
            'Housing (Rent/Mortgage)': 0.25, 'Utilities & Internet': 0.08,
            'Groceries': 0.12, 'Transportation': 0.12, 'Food & Dining': 0.10,
            'Healthcare': 0.06, 'Entertainment & Shopping': 0.15,
            'Subscriptions': 0.04, 'Savings & Emergency': 0.08,
        }

        card_open("Industry-Standard Budget Allocations", "Recommended percentage of monthly income")
        budget_data = [{'Category': cat, 'Recommended %': f"{pct*100:.0f}%", 'Monthly Budget ($)': f"${avg_monthly_income*pct:,.0f}"} for cat, pct in standard_budget.items()]
        st.dataframe(pd.DataFrame(budget_data), use_container_width=True, hide_index=True)
        card_close()

        card_open("Your Actual Spending by Category", "Compared to total income")
        actual = expenses_df.groupby('category')['amount'].agg(['sum','count','mean']).round(2)
        actual = actual.sort_values('sum', ascending=False)
        actual.columns = ['Total ($)','Transactions','Avg ($)']
        actual['% of Income'] = (actual['Total ($)'] / total_income * 100).round(1)
        st.dataframe(actual, use_container_width=True)
        card_close()

        card_open("Budget Recommendations", "Categories deviating from standard guidelines")
        recommendation_map = {
            'Rent': 'Housing (Rent/Mortgage)', 'Utilities': 'Utilities & Internet',
            'Groceries': 'Groceries', 'Transportation': 'Transportation',
            'Food & Dining': 'Food & Dining', 'Healthcare': 'Healthcare',
            'Entertainment': 'Entertainment & Shopping', 'Shopping': 'Entertainment & Shopping',
            'Subscriptions': 'Subscriptions',
        }
        found_any = False
        for actual_cat, recommended_cat in recommendation_map.items():
            if actual_cat not in actual.index: continue
            recommended_monthly = avg_monthly_income * standard_budget[recommended_cat]
            actual_monthly      = actual.loc[actual_cat, 'Total ($)'] / period_months
            if actual_monthly > recommended_monthly * 1.2:
                status('warning', f'<strong>{actual_cat}:</strong> ${actual_monthly:,.0f}/month — recommended ${recommended_monthly:,.0f}/month')
                found_any = True
            elif actual_monthly < recommended_monthly * 0.5:
                status('info',    f'<strong>{actual_cat}:</strong> ${actual_monthly:,.0f}/month — below recommended ${recommended_monthly:,.0f}/month')
                found_any = True
        if not found_any:
            status('success', 'Your spending aligns well with recommended budget allocations.')
        card_close()

        card_open("AI-Powered Budget Optimization", "Personalized recommendations based on your data")
        if AI_ENABLED:
            if st.button("Generate AI Budget Recommendations"):
                with st.spinner("Analyzing..."):
                    ai_recs = generate_budget_recommendations(df, total_income)
                    if '_ai_tips' in ai_recs: status('success', ai_recs['_ai_tips'])
        else:
            status('info', 'AI recommendations require <code>ANTHROPIC_API_KEY</code> environment variable.')
        card_close()
    else:
        status('warning', 'No income transactions detected. Upload data with a <code>transaction_type</code> column or use negative amounts for income.')

# ============================================================
#  MONTHLY REPORT
# ============================================================

elif page == "Monthly Report":
    page_header("Monthly Financial Report", "Detailed breakdown of income and expenses for a selected month")
    months = df['date'].dt.to_period('M').unique()
    selected_month = st.selectbox("Select Month", sorted(months, reverse=True))
    month_df = df[df['date'].dt.to_period('M') == selected_month]

    if len(month_df) > 0:
        month_income   = month_df[month_df['transaction_type'] == 'income']['amount'].sum()
        month_expenses = month_df[month_df['transaction_type'] == 'expense']['amount'].sum()
        month_balance  = month_income - month_expenses

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Income",       f"${month_income:,.2f}")
        with col2: st.metric("Expenses",     f"${month_expenses:,.2f}")
        with col3: st.metric("Balance",      f"${month_balance:,.2f}")
        with col4: st.metric("Transactions", len(month_df))
        spacer()

        card_open("Top Expenses This Month")
        top_exp = month_df[month_df['transaction_type'] == 'expense'].nlargest(15, 'amount')[['date','merchant_name','category','amount']].copy()
        top_exp['date'] = top_exp['date'].dt.strftime('%Y-%m-%d')
        top_exp.columns = ['Date','Merchant','Category','Amount ($)']
        st.dataframe(top_exp, use_container_width=True, hide_index=True)
        card_close()

        card_open("Spending by Category")
        cat_sum = month_df[month_df['transaction_type'] == 'expense'].groupby('category')['amount'].agg(['sum','count']).sort_values('sum', ascending=False)
        cat_sum.columns = ['Total ($)','Count']
        st.dataframe(cat_sum, use_container_width=True)
        card_close()

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            card_open("Expenses by Category")
            cat_data = month_df[month_df['transaction_type'] == 'expense'].groupby('category')['amount'].sum()
            if len(cat_data) > 0:
                fig = px.pie(values=cat_data.values, names=cat_data.index, color_discrete_sequence=COLOR_PALETTE, hole=0.45)
                plotly_fig(fig, height=300, showlegend=True)
            card_close()

        with col2:
            card_open("Daily Spending")
            daily_data = month_df[month_df['transaction_type'] == 'expense'].groupby(month_df['date'].dt.date)['amount'].sum()
            if len(daily_data) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily_data.index, y=daily_data.values,
                    marker_color='#4a90d9', marker_line_color='rgba(0,0,0,0)',
                    hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"
                ))
                plotly_fig(fig, height=300, showlegend=False, xaxis_title="", yaxis_title="USD ($)")
            card_close()

        card_open("AI Monthly Insights")
        if AI_ENABLED:
            if st.button("Generate AI Analysis for This Month"):
                with st.spinner("Analyzing..."):
                    month_analysis = generate_spending_insights(month_df)
                    status('success', str(month_analysis))
        else:
            status('info', 'AI insights require <code>ANTHROPIC_API_KEY</code>.')
        card_close()

# ============================================================
#  BILL REMINDERS
# ============================================================

elif page == "Bill Reminders":
    page_header("Bill Reminders & Payment Tracking", "Upcoming bills derived from your recurring payment patterns")
    bills = recurring_df.copy().sort_values('avg_amount', ascending=False)

    if len(bills) > 0:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Monthly Bills", f"${bills['avg_amount'].sum():,.0f}", delta=f"{len(bills)} subscriptions")
        with col2: st.metric("Annual Bill Cost", f"${bills['total_annual'].sum():,.0f}")
        with col3: st.metric("Next Bill Date", (datetime.now() + timedelta(days=1)).strftime('%b %d'))
        spacer()

        card_open("Payment Calendar — Next 30 Days")
        upcoming = []
        for _, row in bills.iterrows():
            due_date   = datetime.now() + timedelta(days=1)
            days_until = (due_date - datetime.now()).days
            upcoming.append({
                'Bill': row['merchant_name'], 'Category': row['category'],
                'Amount': f"${row['avg_amount']:,.0f}", 'Due Date': due_date.strftime('%b %d, %Y'),
                'Days Until': days_until,
                'Status': 'Due Now' if days_until <= 0 else ('Due Soon' if days_until <= 3 else 'Upcoming')
            })
        st.dataframe(pd.DataFrame(upcoming), use_container_width=True, hide_index=True)
        card_close()

        card_open("AI Bill Optimization")
        if AI_ENABLED:
            if st.button("Get AI Bill Recommendations"):
                with st.spinner("Analyzing..."):
                    bill_recs = generate_smart_bill_reminders(recurring_df, df)
                    if '_recommendations' in bill_recs: status('success', bill_recs['_recommendations'])
                    if '_urgency' in bill_recs:         status('warning', bill_recs['_urgency'])
        else:
            status('info', 'AI recommendations require <code>ANTHROPIC_API_KEY</code>.')
        card_close()

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            card_open("Reduce Bill Amount")
            st.write("- Review contract terms (internet, insurance)")
            st.write("- Negotiate rates with providers")
            st.write("- Bundle services for discounts")
            card_close()
        with col2:
            card_open("Annual Savings Opportunity")
            potential = bills[bills['category'] == 'Subscriptions']['total_annual'].sum() * 0.3
            status('info', f'Reducing 30% of discretionary bills could save <strong>${potential:,.0f}/year</strong>.')
            card_close()

        card_open("Bill Payment History")
        ph = df[df['category'].isin(bills['category'].unique())].copy().sort_values('date', ascending=False).head(20)
        ph['date'] = ph['date'].dt.strftime('%Y-%m-%d')
        ph = ph[['date','merchant_name','category','amount']]
        ph.columns = ['Date','Merchant','Category','Amount ($)']
        st.dataframe(ph, use_container_width=True, hide_index=True)
        card_close()
    else:
        status('info', 'No recurring bills detected. Upload data with recurring transactions to see this section.')

# ============================================================
#  REPORTS & EXPORT
# ============================================================

elif page == "Reports & Export":
    page_header("Reports & Export", "Generate and download financial reports in your preferred format")

    card_open("Report Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Report Type**")
        report_type = st.selectbox("Choose report", ["Monthly Summary","Annual Summary","Category Analysis","Subscription Audit"], label_visibility="collapsed")
    with col2:
        st.write("**Export Format**")
        export_format = st.selectbox("Choose format", ["Excel (XLSX)","CSV"], label_visibility="collapsed")
    card_close()

    card_open(report_type)

    if report_type == "Monthly Summary":
        months = sorted(df['date'].dt.to_period('M').unique(), reverse=True)
        selected_months = st.multiselect("Select months", months, default=[months[0]] if months else [])
        if selected_months:
            report_data = []
            for month in selected_months:
                mdf = df[df['date'].dt.to_period('M') == month]
                inc = mdf[mdf['transaction_type'] == 'income']['amount'].sum()
                exp = mdf[mdf['transaction_type'] == 'expense']['amount'].sum()
                report_data.append({'Month': str(month), 'Income': f"${inc:,.2f}", 'Expenses': f"${exp:,.2f}", 'Balance': f"${inc-exp:,.2f}", 'Transactions': len(mdf)})
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            if export_format == "Excel (XLSX)":
                buffer = pd.ExcelWriter('monthly_summary.xlsx', engine='openpyxl')
                report_df.to_excel(buffer, sheet_name='Monthly Summary', index=False)
                buffer.close()
                with open('monthly_summary.xlsx', 'rb') as f:
                    st.download_button("Download Monthly Summary (Excel)", f.read(), f"monthly_summary_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.download_button("Download Monthly Summary (CSV)", report_df.to_csv(index=False), f"monthly_summary_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    elif report_type == "Annual Summary":
        ann_income   = df[df['transaction_type'] == 'income']['amount'].sum()
        ann_expenses = df[df['transaction_type'] == 'expense']['amount'].sum()
        savings      = ann_income - ann_expenses
        srate        = (savings / ann_income * 100) if ann_income > 0 else 0
        period_days  = (df['date'].max() - df['date'].min()).days
        period_mo    = max(1, period_days / 30)
        summary_df   = pd.DataFrame({
            'Metric': ['Total Income','Total Expenses','Net Savings','Savings Rate (%)','Avg Monthly Income','Avg Monthly Expenses','Days Tracked'],
            'Amount': [f"${ann_income:,.2f}",f"${ann_expenses:,.2f}",f"${savings:,.2f}",f"{srate:.1f}%",
                       f"${ann_income/period_mo:,.2f}",f"${ann_expenses/period_mo:,.2f}",str(period_days)]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown('<div class="section-card-title" style="margin-bottom:12px">AI Financial Health Assessment</div>', unsafe_allow_html=True)
        if AI_ENABLED:
            if st.button("Generate AI Financial Assessment"):
                with st.spinner("Assessing..."):
                    ha = generate_financial_health_assessment(df, ann_income, ann_expenses)
                    if '_health_score' in ha:     st.success(f"Financial Health Score: {ha['_health_score']}")
                    if '_assessment' in ha:        st.write(ha['_assessment'])
                    if '_recommendations' in ha:   status('info', ha['_recommendations'])
        else:
            status('info', 'AI assessment requires <code>ANTHROPIC_API_KEY</code>.')
        if export_format == "Excel (XLSX)":
            buffer = pd.ExcelWriter('annual_summary.xlsx', engine='openpyxl')
            summary_df.to_excel(buffer, sheet_name='Annual Summary', index=False)
            cat_s = df[df['transaction_type']=='expense'].groupby('category')['amount'].agg(['sum','count','mean']).reset_index()
            cat_s.columns = ['Category','Total','Count','Average']
            cat_s.to_excel(buffer, sheet_name='Category Breakdown', index=False)
            buffer.close()
            with open('annual_summary.xlsx', 'rb') as f:
                st.download_button("Download Annual Summary (Excel)", f.read(), f"annual_summary_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.download_button("Download Annual Summary (CSV)", summary_df.to_csv(index=False), f"annual_summary_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    elif report_type == "Category Analysis":
        cat_analysis = df[df['transaction_type']=='expense'].groupby('category').agg({'amount':['sum','count','mean','min','max']}).round(2)
        cat_analysis.columns = ['Total ($)','Count','Avg ($)','Min ($)','Max ($)']
        cat_analysis = cat_analysis.sort_values('Total ($)', ascending=False)
        st.dataframe(cat_analysis, use_container_width=True)
        if export_format == "Excel (XLSX)":
            buffer = pd.ExcelWriter('category_analysis.xlsx', engine='openpyxl')
            cat_analysis.to_excel(buffer, sheet_name='Category Analysis')
            buffer.close()
            with open('category_analysis.xlsx', 'rb') as f:
                st.download_button("Download Category Analysis (Excel)", f.read(), f"category_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.download_button("Download Category Analysis (CSV)", cat_analysis.to_csv(), f"category_analysis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    elif report_type == "Subscription Audit":
        if len(recurring_df) > 0:
            audit = recurring_df.copy().sort_values('total_annual', ascending=False)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Active Subscriptions", len(audit))
            with col2: st.metric("Annual Cost",  f"${audit['total_annual'].sum():,.2f}")
            with col3: st.metric("Monthly Cost", f"${audit['total_annual'].sum()/12:,.2f}")
            st.divider()
            audit_display = audit[['merchant_name','category','avg_amount','count','total_annual']].copy()
            audit_display.columns = ['Subscription','Category','Monthly ($)','Occurrences','Annual ($)']
            st.dataframe(audit_display, use_container_width=True, hide_index=True)
            st.divider()
            st.write("**Top 3 Most Expensive Subscriptions:**")
            for _, row in audit.nlargest(3, 'total_annual').iterrows():
                st.write(f"- **{row['merchant_name']}** — ${row['total_annual']:,.0f}/year — could save ${row['total_annual']*0.3:,.0f} with alternatives")
            st.divider()
            audit['usage_score'] = (audit['count'] / audit['count'].max() * 100).round(0)
            underused = audit[audit['usage_score'] < 50]
            if len(underused) > 0:
                status('warning', f'<strong>{len(underused)}</strong> subscription(s) have low usage frequency. Consider canceling them.')
                for _, row in underused.iterrows():
                    st.write(f"- {row['merchant_name']} — only {int(row['count'])} transactions/year")
            else:
                status('success', 'All subscriptions appear to be actively used.')
            if export_format == "Excel (XLSX)":
                buffer = pd.ExcelWriter('subscription_audit.xlsx', engine='openpyxl')
                audit_display.to_excel(buffer, sheet_name='Subscriptions', index=False)
                buffer.close()
                with open('subscription_audit.xlsx', 'rb') as f:
                    st.download_button("Download Subscription Audit (Excel)", f.read(), f"subscription_audit_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.download_button("Download Subscription Audit (CSV)", audit_display.to_csv(index=False), f"subscription_audit_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            status('info', 'No subscriptions detected. Upload data with recurring transactions to see the subscription audit.')

    card_close()
