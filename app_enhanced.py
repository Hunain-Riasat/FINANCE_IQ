"""
AI-Powered Financial Intelligence Platform v2.1
FIXES in this version:
  1. Chat quick-prompt buttons work instantly (pending_prompt pattern, no race condition)
  2. OCR works without Tesseract — uses Claude Vision API as primary, Tesseract as fallback
  3. Data source redesigned: Upload (NayaPay/real), Sample Data (NayaPay format), AI-Generate
  4. PDF parsing directly loads into the app — no re-upload step needed
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings, io, base64, os, re
warnings.filterwarnings('ignore')

# ── Original AI utilities ─────────────────────────────────────────
try:
    from ai_utils import (
        generate_spending_insights, predict_monthly_spending,
        detect_anomalies_ai, generate_budget_recommendations,
        analyze_subscriptions_ai, generate_financial_health_assessment,
        generate_smart_bill_reminders,
    )
    ORIGINAL_AI_ENABLED = True
except ImportError:
    ORIGINAL_AI_ENABLED = False

# ── Enhanced AI utilities ─────────────────────────────────────────
try:
    from ai_utils_enhanced import (
        chat_with_claude, parse_pdf_statement, parse_image_receipt,
        extract_transaction_from_ocr, generate_realistic_dataset,
        detect_recurring_payments_enhanced, compute_financial_health_score,
        generate_smart_alerts, CHATBOT_ENABLED,
        parse_nayapay_csv, parse_nayapay_pdf_text,   # new specific parsers
        HF_API_KEY,
    )
    ENHANCED_AI = True
except ImportError:
    ENHANCED_AI = False
    CHATBOT_ENABLED = False
    HF_API_KEY = ""

AI_ENABLED = ORIGINAL_AI_ENABLED

st.set_page_config(page_title="AI Financial Intelligence Platform", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500&display=swap');
:root{--bg-base:#0f1117;--bg-surface:#161b27;--bg-card:#1c2336;--bg-card-hover:#202840;--bg-input:#141929;--border:rgba(255,255,255,0.07);--text-primary:#eef0f6;--text-secondary:#8b92a9;--text-muted:#555f7a;--accent-blue:#4a90d9;--accent-teal:#38b2ac;--accent-gold:#c9a84c;--accent-red:#e05c6a;--accent-green:#48bb78;--radius-card:14px;--radius-sm:8px;--shadow-card:0 2px 12px rgba(0,0,0,0.35);--font-sans:'DM Sans',sans-serif;}
#MainMenu,footer,.stDeployButton{visibility:hidden!important;display:none!important;}
header[data-testid="stHeader"]{background:transparent!important;border-bottom:none!important;box-shadow:none!important;}
[data-testid="stSidebarCollapsedControl"]{display:flex!important;visibility:visible!important;opacity:1!important;background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important;z-index:9999!important;}
html,body,[class*="css"]{font-family:var(--font-sans)!important;color:var(--text-primary);}
.stApp{background-color:var(--bg-base)!important;}
.block-container{padding:8px 36px 48px!important;max-width:1300px!important;}
[data-testid="stSidebar"]{background-color:var(--bg-surface)!important;border-right:1px solid var(--border)!important;padding-top:0!important;}
.kpi-card{padding:18px 20px;border-radius:var(--radius-card);border:1px solid var(--border);background:var(--bg-card);margin-bottom:4px;}
.kpi-label{font-size:11px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;}
.kpi-value{font-size:24px;font-weight:700;letter-spacing:-.6px;color:var(--text-primary);}
.kpi-delta{font-size:11px;color:var(--text-secondary);margin-top:5px;}
.kpi-card.teal{border-top:2px solid var(--accent-teal);}
.kpi-card.blue{border-top:2px solid var(--accent-blue);}
.kpi-card.gold{border-top:2px solid var(--accent-gold);}
.kpi-card.green{border-top:2px solid var(--accent-green);}
.kpi-card.red{border-top:2px solid var(--accent-red);}
.section-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-card);padding:20px 22px;margin-bottom:18px;box-shadow:var(--shadow-card);}
.section-card-title{font-size:14px;font-weight:600;color:var(--text-primary);}
.section-card-subtitle{font-size:12px;color:var(--text-muted);margin-top:2px;}
.status-banner{padding:10px 14px;border-radius:var(--radius-sm);font-size:13px;margin:8px 0;border:1px solid;}
.status-success{background:rgba(72,187,120,.09);border-color:rgba(72,187,120,.25);color:#68d391;}
.status-warning{background:rgba(201,168,76,.09);border-color:rgba(201,168,76,.25);color:#e9c46a;}
.status-danger{background:rgba(224,92,106,.09);border-color:rgba(224,92,106,.25);color:#fc8181;}
.status-info{background:rgba(74,144,217,.09);border-color:rgba(74,144,217,.25);color:#7ec8e3;}
.page-title{font-size:21px;font-weight:700;color:var(--text-primary);letter-spacing:-.4px;margin-bottom:4px;}
.page-subtitle{font-size:12.5px;color:var(--text-muted);margin-bottom:22px;}
.sidebar-section-label{font-size:10px!important;font-weight:600!important;letter-spacing:1.4px;text-transform:uppercase;color:var(--text-muted)!important;padding:16px 22px 6px;display:block;}
.sidebar-divider{height:1px;background:var(--border);margin:10px 16px;}
.chat-user{background:rgba(74,144,217,.13);border:1px solid rgba(74,144,217,.2);border-radius:12px 12px 4px 12px;padding:10px 14px;margin:6px 0 6px 60px;font-size:13.5px;color:var(--text-primary);}
.chat-ai{background:var(--bg-card);border:1px solid var(--border);border-radius:12px 12px 12px 4px;padding:10px 14px;margin:6px 60px 6px 0;font-size:13.5px;color:var(--text-primary);}
.chat-label-user{font-size:10px;font-weight:700;color:var(--accent-blue);letter-spacing:.8px;text-align:right;margin:0 2px 2px;}
.chat-label-ai{font-size:10px;font-weight:700;color:var(--accent-teal);letter-spacing:.8px;margin:0 2px 2px;}
.dsrc-note{font-size:11px;color:#8b92a9;padding:6px 14px 10px;line-height:1.6;}
</style>""", unsafe_allow_html=True)

COLOR_PALETTE = ['#4a90d9','#38b2ac','#c9a84c','#e05c6a','#48bb78','#9f7aea','#ed8936','#667eea','#f687b3','#81e6d9']
PLOTLY_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans',color='#8b92a9',size=12),margin=dict(l=12,r=12,t=12,b=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)',zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)',zerolinecolor='rgba(255,255,255,0.08)'))

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def card_open(title,subtitle=""):
    sub = f'<div class="section-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-card"><div class="section-card-title">{title}</div>{sub}</div>',unsafe_allow_html=True)
def card_close(): pass
def status(t,m): st.markdown(f'<div class="status-banner status-{t}">{m}</div>',unsafe_allow_html=True)
def page_header(t,s=""):
    st.markdown(f'<div class="page-title">{t}</div>',unsafe_allow_html=True)
    if s: st.markdown(f'<div class="page-subtitle">{s}</div>',unsafe_allow_html=True)
def spacer(h=12): st.markdown(f'<div style="height:{h}px"></div>',unsafe_allow_html=True)
def plotly_fig(fig,height=300,**kw):
    fig.update_layout(**{**PLOTLY_LAYOUT,**kw})
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

# ══════════════════════════════════════════════════════════════════
#  CORE ML
# ══════════════════════════════════════════════════════════════════
def auto_categorize(m):
    m = str(m).lower()
    kw = {'Groceries':['grocery','imtiaz','carrefour','chase up','metro','alfatah','hyperstar','naheed'],
          'Food & Dining':['restaurant','cafe','burger','pizza','mcdonald','kfc','hardee','foodpanda','cheetay','careem food','biryani'],
          'Utilities':['lesco','gepco','ssgc','sngpl','ptcl','jazz','telenor','zong','ufone','sui','electric','water','utility','internet'],
          'Rent':['rent','landlord','apartment','lease'],
          'Transportation':['careem','indrive','bykea','uber','lyft','pso','shell','caltex','petrol','fuel','parking'],
          'Entertainment':['netflix','spotify','youtube','cinema','movie','steam','disney','nueplex','cinepax','hulu'],
          'Shopping':['daraz','telemart','shophive','amazon','target','walmart','h&m'],
          'Healthcare':['oladoc','shifa','agha khan','hospital','pharmacy','doctor','dental','lab'],
          'Subscriptions':['subscription','adobe','microsoft','icloud','canva','zoom','dropbox','linkedin','gym','membership'],
          'Financial':['nayapay','jazzcash','easypaisa','hbl','meezan','alfalah','bank','transfer']}
    for cat,words in kw.items():
        if any(w in m for w in words): return cat
    return 'Shopping'

def detect_fraud(df):
    if len(df)<10:
        d=df.copy(); d['is_fraud']=False; d['fraud_score']=0.0; return d
    X=df[['amount']].copy(); X['dom']=df['date'].dt.day; X['dow']=df['date'].dt.dayofweek
    Xs=StandardScaler().fit_transform(X)
    iso=IsolationForest(contamination=0.05,random_state=42); preds=iso.fit_predict(Xs)
    d=df.copy(); d['is_fraud']=preds==-1; d['fraud_score']=iso.decision_function(Xs); return d

def detect_recurring_payments(df):
    if len(df)==0: return pd.DataFrame()
    rec=[]
    for merchant in df['merchant_name'].unique():
        t=df[df['merchant_name']==merchant].sort_values('date')
        if len(t)<2: continue
        a=t['amount'].values; ma=a.mean()
        if ma==0: continue
        cva=a.std()/ma; cvi=0
        if len(t)>=3:
            diffs=np.diff(t['date'].values).astype('timedelta64[D]').astype(int)
            if len(diffs)>0 and diffs.mean()>0: cvi=diffs.std()/diffs.mean()
        if cva<=0.15 and cvi<=0.3:
            rec.append({'merchant_name':merchant,'category':t['category'].iloc[0],'count':len(t),'avg_amount':round(ma,2),'total_annual':round(ma*12,2)})
    return pd.DataFrame(rec).sort_values('total_annual',ascending=False) if rec else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════
def _normalize(df):
    df.columns=df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]','_',regex=True)
    dc=[c for c in df.columns if any(k in c for k in ['date','time','when','day'])]
    dc=dc[0] if dc else df.columns[0]
    df['date']=pd.to_datetime(df[dc],errors='coerce')
    if df['date'].isna().all():
        for c in df.columns:
            a=pd.to_datetime(df[c],errors='coerce')
            if a.notna().sum()>len(df)*.5: df['date']=a; break
    df=df.dropna(subset=['date'])
    if df.empty: return None
    ac=[c for c in df.columns if any(k in c for k in ['amount','sum','total','debit','credit','value','price','cost'])]
    if not ac:
        for c in df.columns:
            if c=='date': continue
            v=pd.to_numeric(df[c],errors='coerce')
            if v.notna().sum()>len(df)*.5: ac=[c]; break
    if not ac: return None
    df['amount']=pd.to_numeric(df[ac[0]],errors='coerce').fillna(0)
    df=df[df['amount']!=0]; 
    if df.empty: return None
    mc=[c for c in df.columns if any(k in c for k in ['merchant','description','desc','name','payee','vendor','note','memo','narration','title'])]
    df['merchant_name']=df[mc[0]].astype(str).str.strip() if mc else 'Unknown'
    cc=[c for c in df.columns if 'categ' in c or c in ['type','tag']]
    df['category']=df[cc[0]].astype(str).str.strip() if (cc and cc[0]!='transaction_type') else df['merchant_name'].apply(auto_categorize)
    df['description']=df['merchant_name']
    tc=[c for c in df.columns if 'transaction_type' in c or c in ['type','txn_type','kind']]
    if tc:
        im=df[tc[0]].astype(str).str.strip().str.lower().isin(['income','credit','deposit','salary','revenue'])
    else:
        im=df['amount']<0
    df['amount']=df['amount'].abs()
    df['transaction_type']=np.where(im,'income','expense')
    df=df[['date','merchant_name','category','amount','description','transaction_type']].sort_values('date').reset_index(drop=True)
    df['transaction_id']=[f'TXN{i:06d}' for i in range(len(df))]
    return df

def load_sample_data():
    try:
        df=pd.read_csv('sample_transactions.csv')
        df['date']=pd.to_datetime(df['date']); df['amount']=pd.to_numeric(df['amount'],errors='coerce')
        df=df.dropna(subset=['amount'])
        im=(df['transaction_type'].astype(str).str.strip().str.lower()=='income' if 'transaction_type' in df.columns else df['amount']<0)
        df['amount']=df['amount'].abs(); df['transaction_type']=np.where(im,'income','expense')
        if 'merchant_name' not in df.columns: df['merchant_name']=df.get('description','Unknown')
        if 'category' not in df.columns: df['category']=df['merchant_name'].apply(auto_categorize)
        if 'transaction_id' not in df.columns: df['transaction_id']=[f'TXN{i:06d}' for i in range(len(df))]
        return df[df['amount']>0].sort_values('date').reset_index(drop=True)
    except Exception: return None

def load_user_data(uploaded_file):
    """Smart loader: NayaPay CSV, standard CSV, Excel."""
    try:
        n = uploaded_file.name.lower()
        raw_bytes = uploaded_file.read()
        if n.endswith('.csv'):
            raw_text = raw_bytes.decode('utf-8', errors='replace')
            # NayaPay CSV has metadata rows — skip to TIMESTAMP header
            if any(k in raw_text for k in ['TIMESTAMP','NayaPay','Peer to Peer','IBFT']):
                if ENHANCED_AI:
                    try:
                        df = parse_nayapay_csv(raw_bytes)
                        if df is not None and len(df) > 0:
                            return df
                    except Exception: pass
                # Manual skip-header fallback
                import io as _io
                lines = raw_text.split('\n')
                start = 0
                for i, l in enumerate(lines):
                    if ('TIMESTAMP' in l or 'TIME' in l) and 'AMOUNT' in l:
                        start = i; break
                df = pd.read_csv(_io.StringIO('\n'.join(lines[start:])))
                # Clean NayaPay amount column
                if 'AMOUNT' in df.columns:
                    df['AMOUNT'] = (df['AMOUNT'].astype(str)
                                   .str.replace(r'Rs\.?\s*','',regex=True)
                                   .str.replace(',','').str.replace('+','').str.strip())
                    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
                if 'DESCRIPTION' in df.columns:
                    df['DESCRIPTION'] = df['DESCRIPTION'].astype(str).str.split('|').str[0].str.split('\n').str[0].str.strip()
                r = _normalize(df)
                if r is not None: return r
            # Standard CSV
            import io as _io
            df = pd.read_csv(_io.BytesIO(raw_bytes))
            if df.empty: st.error("File is empty"); return None
            r = _normalize(df)
            if r is None: st.error("Could not parse date/amount columns."); return None
            return r
        elif n.endswith(('.xlsx', '.xls')):
            import io as _io
            df = pd.read_excel(_io.BytesIO(raw_bytes))
            if df.empty: st.error("File is empty"); return None
            r = _normalize(df)
            if r is None: st.error("Could not parse date/amount columns."); return None
            return r
        else:
            st.error("Please upload CSV, Excel (.xlsx/.xls), or PDF"); return None
    except Exception as e:
        st.error(f"Error reading file: {e}"); return None


def load_pdf_data(file_bytes):
    """Parse NayaPay PDF and generic bank PDF into DataFrame."""
    try:
        import pdfplumber, io as _io
        full_text = ''; table_records = []
        with pdfplumber.open(_io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                pt = page.extract_text() or ''
                full_text += pt + '\n'
                for tbl in page.extract_tables():
                    if not tbl or len(tbl)<2: continue
                    hdrs=[str(h).lower().strip().replace(' ','_') if h else f'c{i}' for i,h in enumerate(tbl[0])]
                    for row in tbl[1:]:
                        if row and any(c for c in row if c):
                            table_records.append(dict(zip(hdrs,[str(c).strip() if c else '' for c in row])))

        # NayaPay PDF: parse text line by line
        if any(k in full_text for k in ['NayaPay','Peer to Peer','IBFT In','Raast','hunainriasat']):
            TYPES = ['Peer to Peer','IBFT In','IBFT Out','Raast In','Raast Out',
                     'Online','Cash In','Cash Out','Request In','Request Out','Bill Payment']
            INCOME_T = {'IBFT In','Raast In','Cash In','Request In'}
            line_pat = re.compile(
                r'^(\d{2} \w{3} \d{4})\s+(.+?)\s+([+-]?Rs\.?\s*[\d,]+\.?\d*)\s+(Rs\.?\s*[\d,]+\.?\d*)\s*$'
            )
            records = []
            for line in full_text.split('\n'):
                m = line_pat.match(line.strip())
                if not m: continue
                date_str, middle, amt_str, _ = m.groups()
                txn_type='Unknown'; desc=middle.strip()
                for t in TYPES:
                    if middle.startswith(t): txn_type=t; desc=middle[len(t):].strip(); break
                # Parse amount properly — keep sign
                sign = -1 if '-' in amt_str else 1
                clean = re.sub(r'[Rs\.\s,]','', amt_str).replace('+','').replace('-','').strip()
                try: amount = float(clean) * sign
                except: continue
                is_inc = (sign > 0 or txn_type in INCOME_T)
                merchant = desc.split('|')[0].split('\n')[0].strip()[:60]
                records.append({
                    'date': date_str, 'merchant_name': merchant,
                    'category': _auto_cat(merchant),
                    'amount': abs(amount), 'description': desc,
                    'transaction_type': 'income' if is_inc else 'expense',
                })
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df['transaction_id'] = [f'TXN{i:06d}' for i in range(len(df))]
                return df.sort_values('date').reset_index(drop=True), None

        # Generic table fallback
        if table_records:
            df = pd.DataFrame(table_records); r = _normalize(df)
            if r is not None and len(r)>0: return r, None

        return None, "Could not extract transactions. Try exporting as CSV from your bank app instead."
    except ImportError: return None, "pdfplumber not installed: pip install pdfplumber"
    except Exception as e: return None, f"PDF error: {e}"


def _auto_cat(desc):
    m = str(desc).lower()
    cats = {
        'Food & Dining':['foodpanda','kfc','pizza','burger','cafe','restaurant','biryani','cheetay'],
        'Utilities':['lesco','gepco','jazz','telenor','zong','ufone','ptcl','electric','sui gas'],
        'Transportation':['careem','indrive','pso','shell','petrol','bykea'],
        'Entertainment':['netflix','spotify','cinema','steam','skin.club','gaming','skin'],
        'Shopping':['daraz','telemart','amazon'],
        'Financial':['nayapay','jazzcash','easypaisa','hbl','meezan','ibft','raast','transfer'],
    }
    for cat,words in cats.items():
        if any(w in m for w in words): return cat
    return 'Other'


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k,v in [('chat_history',[]),('pending_prompt',None),('generated_df',None),('data_label','sample'),('ocr_transactions',[])]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="padding:22px 14px 14px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:4px;text-align:center;">
  <div style="display:flex;align-items:center;justify-content:center;gap:10px;">
    <svg width="30" height="30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="30" height="30" rx="7" fill="#1a2d4a"/>
      <path d="M7 20V10l7-4 7 4v10l-7 4-7-4z" stroke="#4a90d9" stroke-width="1.5" fill="none"/>
      <path d="M14 6v16M7 10l7 4 7-4" stroke="#38b2ac" stroke-width="1.2"/>
      <circle cx="14" cy="14" r="2" fill="#4a90d9"/>
    </svg>
    <div>
      <div style="font-size:24px;font-weight:800;color:#eef0f6;line-height:1.1;">Finance <span style="color:#4a90d9;">IQ</span></div>
      <div style="font-size:9px;color:#555f7a;letter-spacing:1.4px;text-transform:uppercase;">AI Platform</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown('<span class="sidebar-section-label">📂 Data Source</span>',unsafe_allow_html=True)

DATA_OPTIONS=[" Upload Bank Statement"," Sample Data"," Generate Demo Data"]
data_choice=st.sidebar.radio("dsr",DATA_OPTIONS,label_visibility="collapsed",key="data_source")

df=None

if data_choice==DATA_OPTIONS[0]:

    uf=st.sidebar.file_uploader("uf",type=["csv","xlsx","xls","pdf"],label_visibility="collapsed")
    if uf:
        if uf.name.lower().endswith('.pdf'):
            with st.spinner("Parsing PDF…"):
                df,err=load_pdf_data(uf.read())
            if df is not None:
                st.sidebar.success(f"✅ PDF: {len(df):,} transactions")
                st.session_state.data_label='live'
            else:
                st.sidebar.error(f"⚠️ {err}")
                st.sidebar.info("💡 Tip: Export as CSV from your bank app → Transaction History → Download")
        else:
            with st.spinner("Loading…"): df=load_user_data(uf)
            if df is not None:
                st.sidebar.success(f"✅ Loaded {len(df):,} real transactions")
                st.session_state.data_label='live'

elif data_choice==DATA_OPTIONS[1]:

    df=load_sample_data()
    if df is not None:
        st.sidebar.success(f"✅ Sample data: {len(df):,} transactions")
        st.session_state.data_label='sample'
    else:
        st.sidebar.warning("sample_transactions.csv not found in project folder.")

elif data_choice==DATA_OPTIONS[2]:

    ca,cb=st.sidebar.columns(2)
    with ca: gm=st.number_input("Months",1,12,3,key="gm")
    with cb: gi=st.number_input("Salary",20000,500000,80000,step=5000,key="gi")
    if st.sidebar.button("⚡ Generate",use_container_width=True):
        if ENHANCED_AI:
            with st.spinner("Generating…"):
                df=generate_realistic_dataset(months=int(gm),monthly_income=float(gi))
            st.session_state.generated_df=df; st.session_state.data_label='ai'
            st.sidebar.success(f"✅ Generated {len(df):,} transactions")
        else:
            st.sidebar.error("ai_utils_enhanced.py not found")
    if st.session_state.generated_df is not None: df=st.session_state.generated_df

# Fallback
if df is None:
    df=load_sample_data()
    if df is None:
        rng=np.random.default_rng(0); n=80
        df=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=n,freq='4D'),
            'merchant_name':rng.choice(['Netflix','Foodpanda','LESCO','Careem','Daraz','Monthly Salary'],n),
            'amount':rng.uniform(200,8000,n).round(2),
            'transaction_type':np.where(np.arange(n)%10==0,'income','expense'),
            'description':['Transaction']*n})
        df['category']=df['merchant_name'].apply(auto_categorize)
        df['transaction_id']=[f'TXN{i:06d}' for i in range(n)]

# Merge any OCR-captured transactions into main df
if st.session_state.ocr_transactions:
    ocr_df = pd.DataFrame(st.session_state.ocr_transactions)
    ocr_df['date'] = pd.to_datetime(ocr_df['date'], errors='coerce')
    ocr_df = ocr_df.dropna(subset=['date'])
    if len(ocr_df) > 0:
        df = pd.concat([df, ocr_df], ignore_index=True).sort_values('date').reset_index(drop=True)

# Computed
df_fraud=detect_fraud(df)
recurring_df=(detect_recurring_payments_enhanced(df) if ENHANCED_AI else detect_recurring_payments(df))

# Nav
st.sidebar.markdown('<div class="sidebar-divider"></div>',unsafe_allow_html=True)
st.sidebar.markdown('<span class="sidebar-section-label">🧭 Navigation</span>',unsafe_allow_html=True)
page=st.sidebar.radio("nav",["Dashboard","AI Chat Assistant","Analytics","Fraud Detection",
    "Recurring Payments","Financial Health","Smart Alerts","Budget Planning",
    "Monthly Report","Bill Reminders","OCR Receipt Upload","Reports & Export"],
    label_visibility="collapsed",key="navigation")
st.sidebar.markdown('<div class="sidebar-divider"></div>',unsafe_allow_html=True)

badge={'live':('🟢 Live Bank Data','color:#48bb78'),'sample':('🔵 Sample Data','color:#4a90d9'),'ai':('🟡 AI-Generated','color:#c9a84c')}
bl,bc=badge.get(st.session_state.data_label,badge['sample'])
fc=int(df_fraud['is_fraud'].sum())
st.sidebar.markdown(f"""<div style="padding:8px 16px 4px;">
  <div style="font-size:11px;color:var(--text-muted);line-height:2.2;">
    <div>Source: <strong style="{bc}">{bl}</strong></div>
    <div>Transactions: <strong style="color:var(--text-secondary)">{len(df):,}</strong></div>
    <div>Fraud alerts: <strong style="color:{'#e05c6a' if fc>0 else 'var(--text-secondary)'}">{fc}</strong></div>
    <div>Subscriptions: <strong style="color:var(--text-secondary)">{len(recurring_df)}</strong></div>
    <div>AI Chat: <strong style="color:{'#48bb78' if CHATBOT_ENABLED else '#e9c46a'}">{'🟢 Claude AI' if os.environ.get('ANTHROPIC_API_KEY') else ('🟢 HuggingFace AI' if os.environ.get('HUGGINGFACE_API_KEY') else '🟡 Offline mode')}</strong></div>
  </div>
</div>
<div style="padding:12px 18px 18px;margin-top:4px;border-top:1px solid rgba(255,255,255,0.06);text-align:center;">
  <div style="font-size:11px;color:var(--text-muted);">Made with <span style="color:#e05c6a;">♥</span> by<br>
  <span style="background:linear-gradient(90deg,#4a90d9,#38b2ac);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;font-size:12.5px;">Zain, Hunain &amp; Shaheer</span></div>
</div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════

# ── DASHBOARD ────────────────────────────────────────────────────
if page=="Dashboard":
    page_header("Financial Dashboard","Overview of your financial activity and key metrics")
    ti=df[df['transaction_type']=='income']['amount'].sum()
    ts=df[df['transaction_type']=='expense']['amount'].sum()
    bal=ti-ts; fa=df_fraud['is_fraud'].sum(); sr=(bal/ti*100) if ti>0 else 0

    c1,c2,c3,c4,c5=st.columns(5)
    for col,lbl,val,dlt,clr in [
        (c1,"Total Income",f"PKR {ti:,.0f}",f"{len(df[df['transaction_type']=='income'])} txns","teal"),
        (c2,"Total Expenses",f"PKR {ts:,.0f}",f"{len(df[df['transaction_type']=='expense'])} txns","blue"),
        (c3,"Net Balance",f"PKR {bal:,.0f}","Income − Expenses","gold" if bal>=0 else "red"),
        (c4,"Savings Rate",f"{sr:.1f}%","Of total income","green"),
        (c5,"Fraud Alerts",str(int(fa)),f"{fa/len(df)*100:.1f}% flagged","red" if fa>5 else "blue")]:
        with col: st.markdown(f'<div class="kpi-card {clr}"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-delta">{dlt}</div></div>',unsafe_allow_html=True)

    spacer(16)
    if ENHANCED_AI:
        alerts=generate_smart_alerts(df,recurring_df,df_fraud)
        urgent=[a for a in alerts if a['type'] in ('danger','warning')]
        if urgent: a=urgent[0]; status(a['type'],f"<strong>{a['icon']} {a['title']}:</strong> {a['message']}")

    c1,c2=st.columns([3,2],gap="medium")
    with c1:
        card_open("Monthly Spending Trend","Expense totals per month")
        ex=df[df['transaction_type']=='expense'].copy()
        if len(ex)>0:
            ex['month']=ex['date'].dt.to_period('M').dt.to_timestamp()
            mo=ex.groupby('month')['amount'].sum().reset_index().sort_values('month')
            fig=go.Figure(); fig.add_trace(go.Scatter(x=mo['month'],y=mo['amount'],mode='lines+markers',fill='tozeroy',
                fillcolor='rgba(74,144,217,0.09)',line=dict(color='#4a90d9',width=2.5),
                marker=dict(size=7,color='#4a90d9'),hovertemplate="<b>%{x|%b %Y}</b><br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=280,showlegend=False,xaxis=dict(type='date',gridcolor='rgba(255,255,255,0.05)'),hovermode='x unified')
        card_close()
    with c2:
        card_open("Spending by Category")
        cd=df[df['transaction_type']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
        if len(cd)>0:
            fig=go.Figure(data=[go.Pie(labels=cd.index,values=cd.values,hole=0.52,
                marker=dict(colors=COLOR_PALETTE,line=dict(color='#1c2336',width=2)),
                hovertemplate="<b>%{label}</b><br>PKR %{value:,.0f}<extra></extra>")])
            plotly_fig(fig,height=280,showlegend=True,legend=dict(orientation='v',x=1.0,y=0.5,font=dict(size=10)))
        card_close()

    card_open("Recent Transactions","Last 20 entries")
    rc=df.tail(20)[['date','merchant_name','category','amount','transaction_type']].copy()
    rc['date']=rc['date'].dt.strftime('%Y-%m-%d'); rc.columns=['Date','Merchant','Category','Amount (PKR)','Type']
    st.dataframe(rc,use_container_width=True,hide_index=True); card_close()

    card_open("AI-Powered Insights")
    if AI_ENABLED:
        st.write("**Financial Health Score**")
        with st.spinner("Calculating…"):
            health=(compute_financial_health_score(df,ti,ts,recurring_df) if ENHANCED_AI else generate_financial_health_assessment(df,ti,ts))
            try: sv=int(str(health.get('_health_score','0/100')).split('/')[0])
            except: sv=0
            sv=max(0,min(100,sv)); sl="Strong" if sv>=75 else ("Moderate" if sv>=50 else "Needs Improvement")
            st.metric("Health Score",f"{sv}/100",delta=sl); st.progress(sv/100)
            if '_assessment' in health: st.caption(health['_assessment'])
            if '_recommendations' in health: status('info',health['_recommendations'])
        x1,x2=st.columns(2)
        with x1:
            st.write("**Spending Analysis**")
            with st.spinner("…"): st.write(generate_spending_insights(df))
        with x2:
            st.write("**3-Month Forecast**")
            with st.spinner("…"):
                fc=predict_monthly_spending(df,months_ahead=3)
                if 'error' not in fc:
                    for k,v in fc.get('forecasts',{}).items():
                        if not str(k).startswith('_') and isinstance(v,(int,float,np.integer,np.floating)):
                            st.write(f"- {k}: PKR {v:,.0f}")
                    if '_interpretation' in fc: status('info',fc['_interpretation'])
                else: st.info("Need 3+ months data for forecast")
    else:
        status('info','Set <code>ANTHROPIC_API_KEY</code> for AI insights.')
    card_close()

# ── AI CHAT — FIXED ──────────────────────────────────────────────
elif page=="AI Chat Assistant":
    page_header("AI Financial Assistant","Ask anything about your finances in plain English")

    CLAUDE_KEY2 = os.environ.get("ANTHROPIC_API_KEY","")
    HF_KEY2     = os.environ.get("HUGGINGFACE_API_KEY","")
    if CLAUDE_KEY2:
        status('success','🟢 Claude AI connected — your transaction data is loaded as full context.')
    elif HF_KEY2:
        status('success','🟢 HuggingFace AI connected (free) — Mistral-7B financial assistant active.')
    elif ENHANCED_AI:
        status('warning',
            '🟡 Offline mode — smart rule-based answers.<br>'
            '<strong>To enable free AI:</strong> Get a free key at '
            '<a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a> '
            'and add <code>HUGGINGFACE_API_KEY=hf_...</code> to your .env file<br>'
            '<strong>Or for Claude:</strong> Add <code>ANTHROPIC_API_KEY=sk-ant-...</code>')
    else:
        status('info','Basic mode. Ensure ai_utils_enhanced.py is present.')

    spacer(8)

    # ── Process pending prompt FIRST before any UI renders ───────
    if st.session_state.pending_prompt:
        msg=st.session_state.pending_prompt
        st.session_state.pending_prompt=None
        with st.spinner("Thinking…"):
            reply=(chat_with_claude(msg,st.session_state.chat_history,df) if ENHANCED_AI
                   else f"Total expenses: PKR {df[df['transaction_type']=='expense']['amount'].sum():,.0f}. Set ANTHROPIC_API_KEY for full AI answers.")
        st.session_state.chat_history.append({'role':'user','content':msg})
        st.session_state.chat_history.append({'role':'assistant','content':reply})

    # ── Quick prompts (set pending_prompt and rerun) ──────────────
    card_open("Quick Questions","Click any question to get an instant AI answer")
    QP=["How much did I spend last month?","Which subscriptions are unnecessary?",
        "What category consumes most money?","How can I save more money?",
        "Which transactions look suspicious?","Predict my next month spending"]
    cols=st.columns(3)
    for i,p in enumerate(QP):
        with cols[i%3]:
            if st.button(p,use_container_width=True,key=f"qp_{i}"):
                st.session_state.pending_prompt=p
                st.rerun()
    card_close()

    # ── Conversation display ──────────────────────────────────────
    card_open("Conversation","Your AI financial advisor")
    if not st.session_state.chat_history:
        st.markdown("""<div class="chat-label-ai">🧠 AI ASSISTANT</div>
<div class="chat-ai">Hi! I'm your AI financial assistant. I have access to your transaction data.<br><br>
I can help you:<br>
• Understand where your money is going<br>
• Find savings and cut unnecessary subscriptions<br>
• Detect unusual or suspicious spending<br>
• Forecast next month's expenses<br><br>
Click a quick question above or type your own below!</div>""",unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn['role']=='user':
            st.markdown(f'<div class="chat-label-user">YOU</div><div class="chat-user">{turn["content"]}</div>',unsafe_allow_html=True)
        else:
            content=turn['content'].replace('\n','<br>').replace('•','&#8226;')
            st.markdown(f'<div class="chat-label-ai">🧠 AI</div><div class="chat-ai">{content}</div>',unsafe_allow_html=True)
    card_close()

    # ── Form input (clear_on_submit prevents double-send) ─────────
    with st.form("chat_form",clear_on_submit=True):
        ui=st.text_input("Question…",placeholder="e.g. How much did I spend on food this month?",label_visibility="collapsed")
        xc1,xc2=st.columns([5,1])
        with xc1: sb=st.form_submit_button("💬 Send",use_container_width=True,type="primary")
        with xc2: cb=st.form_submit_button("🗑️",use_container_width=True)

    if cb:
        st.session_state.chat_history=[]; st.rerun()
    if sb and ui.strip():
        with st.spinner("Thinking…"):
            reply=(chat_with_claude(ui,st.session_state.chat_history,df) if ENHANCED_AI
                   else f"Total expenses: PKR {df[df['transaction_type']=='expense']['amount'].sum():,.0f}. Set ANTHROPIC_API_KEY for full AI answers.")
        st.session_state.chat_history.append({'role':'user','content':ui})
        st.session_state.chat_history.append({'role':'assistant','content':reply})
        st.rerun()

# ── ANALYTICS ────────────────────────────────────────────────────
elif page=="Analytics":
    page_header("Detailed Analytics","Granular spending breakdowns")
    ex=df[df['transaction_type']=='expense'].copy()
    if len(ex)==0: status('warning','No expense transactions.'); st.stop()

    c1,c2=st.columns(2,gap="medium")
    with c1:
        card_open("Category Breakdown")
        ca=ex.groupby('category')['amount'].sum().sort_values(ascending=True)
        fig=go.Figure(go.Bar(x=ca.values,y=ca.index,orientation='h',marker=dict(color=COLOR_PALETTE[:len(ca)]),
            hovertemplate="<b>%{y}</b><br>PKR %{x:,.0f}<extra></extra>"))
        plotly_fig(fig,height=320,showlegend=False,xaxis_title="Amount (PKR)"); card_close()
    with c2:
        card_open("Spending by Day of Week")
        ex['dow']=ex['date'].dt.day_name()
        ds=ex.groupby('dow')['amount'].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
        fig=go.Figure(go.Bar(x=ds.index,y=ds.values,marker=dict(color='#38b2ac',opacity=0.8),
            hovertemplate="<b>%{x}</b><br>PKR %{y:,.0f}<extra></extra>"))
        plotly_fig(fig,height=320,showlegend=False,yaxis_title="Total (PKR)"); card_close()

    card_open("Top 15 Merchants")
    tm=ex.groupby('merchant_name')['amount'].sum().nlargest(15).sort_values()
    fig=go.Figure(go.Bar(x=tm.values,y=tm.index,orientation='h',marker=dict(color='#4a90d9',opacity=0.85),
        hovertemplate="<b>%{y}</b><br>PKR %{x:,.0f}<extra></extra>"))
    plotly_fig(fig,height=420,showlegend=False,xaxis_title="Total (PKR)"); card_close()

    card_open("Income vs Expenses Over Time")
    dm=df.copy(); dm['month']=dm['date'].dt.to_period('M').dt.to_timestamp()
    im2=dm[dm['transaction_type']=='income'].groupby('month')['amount'].sum()
    em2=dm[dm['transaction_type']=='expense'].groupby('month')['amount'].sum()
    am=sorted(set(im2.index)|set(em2.index))
    fig=go.Figure()
    fig.add_trace(go.Bar(name='Income',x=am,y=[im2.get(m,0) for m in am],marker_color='#48bb78'))
    fig.add_trace(go.Bar(name='Expenses',x=am,y=[em2.get(m,0) for m in am],marker_color='#e05c6a'))
    fig.update_layout(barmode='group')
    plotly_fig(fig,height=300,xaxis=dict(type='date',gridcolor='rgba(255,255,255,0.05)'),yaxis_title="PKR",hovermode='x unified')
    card_close()

# ── FRAUD DETECTION ───────────────────────────────────────────────
elif page=="Fraud Detection":
    page_header("Fraud Detection & Security","Isolation Forest anomaly detection")
    fc2=df_fraud['is_fraud'].sum(); fp=(fc2/len(df_fraud)*100) if len(df_fraud)>0 else 0

    if AI_ENABLED and st.checkbox("Enable Z-Score Statistical Analysis"):
        with st.spinner("Analysing…"):
            an=detect_anomalies_ai(df_fraud)
            if an:
                card_open("Statistical Anomalies (Z-Score > 2.5)")
                for a in an[:5]:
                    x1,x2=st.columns([3,2])
                    with x1: st.write(f"**{a['merchant']}** — Z: {a.get('z_score',0):.2f}")
                    with x2: st.metric("Amount",f"PKR {a['amount']:,.0f}",delta="High")
                card_close()

    x1,x2,x3=st.columns(3)
    with x1: st.metric("Suspicious",int(fc2))
    with x2: st.metric("Fraud Rate",f"{fp:.1f}%")
    with x3: st.metric("Safe",int(len(df_fraud)-fc2))
    spacer(8)
    if fp>10:   status('danger','High fraud rate. Review immediately.')
    elif fp>5:  status('warning','Moderate fraud rate. Monitor closely.')
    else:       status('success','Fraud rate within normal range.')

    card_open("Flagged Transactions")
    ft=df_fraud[df_fraud['is_fraud']].copy()
    if len(ft)>0:
        dc=['date','merchant_name','category','amount']
        if 'fraud_score' in ft.columns:
            ft['anomaly_%']=((ft['fraud_score'].min()-ft['fraud_score'])/(ft['fraud_score'].min()-ft['fraud_score'].max()+1e-9)*100).round(1)
            dc.append('anomaly_%')
        ft['date']=ft['date'].dt.strftime('%Y-%m-%d')
        dn=['Date','Merchant','Category','Amount (PKR)']+(['Anomaly Score (%)'] if 'anomaly_%' in dc else [])
        d=ft[dc].copy(); d.columns=dn
        st.dataframe(d,use_container_width=True,hide_index=True)
        status('warning','Contact your bank for any unrecognized charges.')
    else: status('success','No suspicious transactions detected.')
    card_close()

    card_open("Anomaly Scatter Plot")
    fig=go.Figure()
    sd=df_fraud[~df_fraud['is_fraud']]; fd2=df_fraud[df_fraud['is_fraud']]
    fig.add_trace(go.Scatter(x=sd['date'],y=sd['amount'],mode='markers',name='Safe',
        marker=dict(color='#48bb78',size=5,opacity=0.6),text=sd['merchant_name'],
        hovertemplate="%{text}<br>PKR %{y:,.0f}<extra></extra>"))
    if len(fd2)>0:
        fig.add_trace(go.Scatter(x=fd2['date'],y=fd2['amount'],mode='markers',name='Flagged',
            marker=dict(color='#e05c6a',size=10,symbol='x',line=dict(width=2,color='#e05c6a')),
            text=fd2['merchant_name'],hovertemplate="⚠️ %{text}<br>PKR %{y:,.0f}<extra></extra>"))
    plotly_fig(fig,height=300,xaxis=dict(type='date',gridcolor='rgba(255,255,255,0.05)'),yaxis_title="PKR",hovermode='closest')
    card_close()

# ── RECURRING PAYMENTS ────────────────────────────────────────────
elif page=="Recurring Payments":
    page_header("Recurring Payments & Subscriptions","Auto-detected regular charges")
    if len(recurring_df)>0:
        x1,x2,x3=st.columns(3)
        with x1: st.metric("Subscriptions",len(recurring_df))
        with x2: st.metric("Monthly Cost",f"PKR {recurring_df['avg_amount'].sum():,.0f}")
        with x3: st.metric("Annual Cost",f"PKR {recurring_df['total_annual'].sum():,.0f}")
        spacer()

        card_open("Your Subscriptions","Sorted by annual cost")
        sc=['merchant_name','category','count','avg_amount','total_annual']
        sl=['Merchant','Category','Times Seen','Avg (PKR)','Annual (PKR)']
        if 'billing_cycle' in recurring_df.columns: sc.insert(3,'billing_cycle'); sl.insert(3,'Cycle')
        if 'last_charge' in recurring_df.columns:   sc.append('last_charge');    sl.append('Last Charge')
        rd=recurring_df[sc].copy(); rd.columns=sl
        st.dataframe(rd,use_container_width=True,hide_index=True)
        status('warning',f'Annual subscription burden: <strong>PKR {recurring_df["total_annual"].sum():,.0f}</strong>')
        card_close()

        card_open("Cost Distribution")
        fig=go.Figure(data=[go.Pie(labels=recurring_df['merchant_name'],values=recurring_df['total_annual'],hole=0.5,
            marker=dict(colors=COLOR_PALETTE,line=dict(color='#1c2336',width=2)))])
        plotly_fig(fig,height=240,showlegend=True,legend=dict(orientation='v',x=1.0,y=0.5,font=dict(size=10)))
        card_close()

        if AI_ENABLED:
            card_open("AI Subscription Optimisation")
            if st.button("Analyse with AI"):
                with st.spinner("…"):
                    aa=analyze_subscriptions_ai(recurring_df)
                    if '_savings_opportunities' in aa: status('success',f'Savings: {aa["_savings_opportunities"]}')
                    if '_alternatives' in aa: status('info',f'Alternatives: {aa["_alternatives"]}')
            card_close()
    else:
        status('info','No recurring payments detected. Upload multiple months of data.')

# ── FINANCIAL HEALTH ──────────────────────────────────────────────
elif page=="Financial Health":
    page_header("Financial Health Score","Composite score 0–100 across 5 dimensions")
    ti2=df[df['transaction_type']=='income']['amount'].sum()
    ts2=df[df['transaction_type']=='expense']['amount'].sum()
    health=(compute_financial_health_score(df,ti2,ts2,recurring_df) if ENHANCED_AI
            else generate_financial_health_assessment(df,ti2,ts2))
    if not ENHANCED_AI:
        health.setdefault('total_score',int(str(health.get('_health_score','0/100')).split('/')[0]))
        health.setdefault('grade','A' if health['total_score']>=85 else 'B' if health['total_score']>=70 else 'C')
        health.setdefault('recommendations',[health.get('_recommendations','')])
        health.setdefault('sub_scores',{})

    sc2=int(health.get('total_score',0)); gr=health.get('grade','?')
    col=('#48bb78' if sc2>=75 else '#c9a84c' if sc2>=50 else '#e05c6a')

    x1,x2=st.columns([1,2],gap="large")
    with x1:
        st.markdown(f"""<div style="text-align:center;padding:28px 16
        px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-card);">
<div style="font-size:76px;font-weight:900;color:{col};line-height:1;">{sc2}</div>
<div style="font-size:13px;color:var(--text-muted);">out of 100</div>
<div style="font-size:46px;font-weight:800;color:{col};margin-top:6px;">Grade {gr}</div>
</div>""",unsafe_allow_html=True)
    with x2:
        st.progress(sc2/100); spacer(10)
        if health.get('sub_scores'):
            card_open("Score Breakdown")
            for k,(lbl,mx) in [('savings',('💰 Savings Ratio',30)),('consistency',('📊 Consistency',20)),
                ('subscription',('🔄 Subscription Load',20)),('risk',('⚠️ Risk Profile',15)),('income_stability',('💵 Income Stability',15))]:
                v=health['sub_scores'].get(k,0)
                st.markdown(f"**{lbl}** — {v:.1f}/{mx}")
                st.progress(min(1.0,v/mx))
            card_close()

    spacer(14)
    card_open("Recommendations")
    for r in health.get('recommendations',[]): 
        if r: status('info',f'💡 {r}')
    sr2=health.get('savings_rate',0)
    if sr2>=20: status('success',f"✅ Great savings rate of {sr2:.1f}%!")
    elif sr2>0: status('warning',f"📈 Savings rate {sr2:.1f}% — aim for 20%+.")
    card_close()

# ── SMART ALERTS ──────────────────────────────────────────────────
elif page=="Smart Alerts":
    page_header("Smart Alerts","Proactive financial notifications")
    if not ENHANCED_AI: status('warning','Enhanced AI module required.'); st.stop()
    alerts=generate_smart_alerts(df,recurring_df,df_fraud)
    da=[a for a in alerts if a['type']=='danger']; wa=[a for a in alerts if a['type']=='warning']; sa=[a for a in alerts if a['type']=='success']
    x1,x2,x3=st.columns(3)
    with x1: st.metric("🚨 Critical",len(da))
    with x2: st.metric("⚠️ Warnings",len(wa))
    with x3: st.metric("✅ All Clear",len(sa))
    spacer(10)
    card_open("Active Alerts")
    for a in alerts:
        act=f' <span style="font-size:11px;opacity:.7;">→ {a["action"]}</span>' if a['action'] else ''
        status(a['type'],f'<strong>{a["icon"]} {a["title"]}:</strong> {a["message"]}{act}')
    card_close()

# ── BUDGET PLANNING ───────────────────────────────────────────────
elif page=="Budget Planning":
    page_header("Budget Planning","Compare actual vs recommended spend")
    ex3=df[df['transaction_type']=='expense'].copy()
    ti3=df[df['transaction_type']=='income']['amount'].sum(); te3=ex3['amount'].sum()
    if ti3>0:
        ms=max(1,(df['date'].max()-df['date'].min()).days/30)
        ami=ti3/ms; ame=te3/ms; sr3=(ti3-te3)/ti3*100
        x1,x2,x3=st.columns(3)
        with x1: st.metric("Avg Monthly Income",f"PKR {ami:,.0f}")
        with x2: st.metric("Avg Monthly Spend",f"PKR {ame:,.0f}")
        with x3: st.metric("Savings Rate",f"{sr3:.1f}%")
        spacer()
        std={'Housing/Rent':0.25,'Utilities':0.08,'Groceries':0.12,'Transport':0.12,'Food & Dining':0.10,'Healthcare':0.06,'Entertainment & Shopping':0.15,'Subscriptions':0.04,'Savings':0.08}
        card_open("Standard Budget Allocations")
        bd=[{'Category':c,'Recommended %':f"{p*100:.0f}%",'Monthly (PKR)':f"PKR {ami*p:,.0f}"} for c,p in std.items()]
        st.dataframe(pd.DataFrame(bd),use_container_width=True,hide_index=True); card_close()
        card_open("Your Actual Spending")
        ac3=ex3.groupby('category')['amount'].agg(['sum','count','mean']).round(0)
        ac3.columns=['Total (PKR)','Txns','Avg (PKR)']; ac3['% of Income']=(ac3['Total (PKR)']/ti3*100).round(1)
        st.dataframe(ac3.sort_values('Total (PKR)',ascending=False),use_container_width=True); card_close()
        card_open("Budget Recommendations")
        rm={'Rent':'Housing/Rent','Utilities':'Utilities','Groceries':'Groceries','Transportation':'Transport',
            'Food & Dining':'Food & Dining','Healthcare':'Healthcare','Entertainment':'Entertainment & Shopping',
            'Shopping':'Entertainment & Shopping','Subscriptions':'Subscriptions'}
        found=False
        for ac4,rc4 in rm.items():
            if ac4 not in ac3.index: continue
            rm2=ami*std.get(rc4,0.1); am2=ac3.loc[ac4,'Total (PKR)']/ms
            if am2>rm2*1.2: status('warning',f'<strong>{ac4}:</strong> PKR {am2:,.0f}/mo — recommended PKR {rm2:,.0f}/mo'); found=True
        if not found: status('success','Spending aligns with recommended allocations.')
        card_close()
        if AI_ENABLED:
            card_open("AI Optimisation")
            if st.button("Generate AI Budget Recommendations"):
                with st.spinner("…"):
                    ar=generate_budget_recommendations(df,ti3)
                    if '_ai_tips' in ar: status('success',ar['_ai_tips'])
            card_close()
    else:
        status('warning','No income transactions found.')

# ── MONTHLY REPORT ────────────────────────────────────────────────
elif page=="Monthly Report":
    page_header("Monthly Financial Report","Detailed breakdown by month")
    months=sorted(df['date'].dt.to_period('M').unique(),reverse=True)
    sm=st.selectbox("Month",months,key="msel")
    mdf=df[df['date'].dt.to_period('M')==sm]
    mi=mdf[mdf['transaction_type']=='income']['amount'].sum(); me=mdf[mdf['transaction_type']=='expense']['amount'].sum()
    mb=mi-me; msr=(mb/mi*100) if mi>0 else 0
    x1,x2,x3,x4=st.columns(4)
    with x1: st.metric("Income",f"PKR {mi:,.0f}")
    with x2: st.metric("Expenses",f"PKR {me:,.0f}")
    with x3: st.metric("Balance",f"PKR {mb:,.0f}")
    with x4: st.metric("Savings Rate",f"{msr:.1f}%")
    spacer()
    x1,x2=st.columns(2,gap="medium")
    with x1:
        card_open(f"Categories — {sm}")
        cm=mdf[mdf['transaction_type']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
        if len(cm)>0:
            fig=go.Figure(go.Bar(x=cm.index,y=cm.values,marker=dict(color=COLOR_PALETTE[:len(cm)]),hovertemplate="<b>%{x}</b><br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=240,showlegend=False,yaxis_title="PKR")
        card_close()
    with x2:
        card_open(f"Daily Spending — {sm}")
        dl=mdf[mdf['transaction_type']=='expense'].groupby(mdf['date'].dt.day)['amount'].sum()
        if len(dl)>0:
            fig=go.Figure(go.Scatter(x=dl.index,y=dl.values,mode='lines+markers',fill='tozeroy',fillcolor='rgba(56,178,172,.09)',line=dict(color='#38b2ac',width=2),marker=dict(size=5),hovertemplate="Day %{x}<br>PKR %{y:,.0f}<extra></extra>"))
            plotly_fig(fig,height=240,showlegend=False,xaxis_title="Day",yaxis_title="PKR")
        card_close()
    card_open("Transactions")
    td=mdf[['date','merchant_name','category','amount','transaction_type']].copy()
    td['date']=td['date'].dt.strftime('%Y-%m-%d'); td.columns=['Date','Merchant','Category','Amount (PKR)','Type']
    st.dataframe(td,use_container_width=True,hide_index=True); card_close()

# ── BILL REMINDERS ────────────────────────────────────────────────
elif page=="Bill Reminders":
    page_header("Bill Reminders","Upcoming bills from recurring payment detection")
    if len(recurring_df)>0:
        today=pd.Timestamp.now()
        card_open("Upcoming Bills (Estimated)")
        bd=[{'Merchant':r['merchant_name'],'Category':r['category'],'Amount (PKR)':r['avg_amount'],
             'Est. Next Due':(today+timedelta(days=30)).strftime('%Y-%m-%d'),'Annual (PKR)':r['total_annual']}
            for _,r in recurring_df.iterrows()]
        st.dataframe(pd.DataFrame(bd),use_container_width=True,hide_index=True); card_close()
        x1,x2,x3=st.columns(3)
        with x1: st.metric("Monthly Total",f"PKR {recurring_df['avg_amount'].sum():,.0f}")
        with x2: st.metric("Annual Total",f"PKR {recurring_df['total_annual'].sum():,.0f}")
        with x3: st.metric("Bills",len(recurring_df))
        if AI_ENABLED:
            card_open("AI Bill Tips")
            if st.button("Get AI Bill Optimisation"):
                with st.spinner("…"):
                    r=generate_smart_bill_reminders(recurring_df,df)
                    if '_recommendations' in r: status('info',r['_recommendations'])
                    if '_urgency' in r: status('warning',r['_urgency'])
            card_close()
    else:
        status('info','No recurring bills detected. Upload multiple months of data.')

# ── OCR RECEIPT UPLOAD — FIXED ────────────────────────────────────
elif page=="OCR Receipt Upload":
    page_header("OCR Receipt / Screenshot Upload","Extract transaction details from any receipt photo")

    CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY","")

    # Show status based on available OCR method
    try:
        import pytesseract
        # Auto-detect Tesseract on Windows
        import os as _os
        for tp in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                   r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']:
            if _os.path.exists(tp): pytesseract.pytesseract.tesseract_cmd = tp; break
        pytesseract.get_tesseract_version()
        TESS_OK = True
    except Exception:
        TESS_OK = False

    if CLAUDE_KEY:
        status('success','🧠 Claude Vision AI active — reads any receipt, screenshot or slip with high accuracy.')
    elif TESS_OK:
        status('success','✅ Tesseract OCR is installed and ready. Upload a receipt image below.')
    else:
        status('warning',
            '⚠️ No OCR engine detected. <strong>Two options:</strong><br>'
            '&nbsp;1. <strong>Claude Vision (recommended, no install):</strong> Add <code>ANTHROPIC_API_KEY=sk-ant-...</code> to your .env file<br>'
            '&nbsp;2. <strong>Tesseract (local, free):</strong> Run in terminal as Administrator: '
            '<code>winget install UB-Mannheim.TesseractOCR</code> → restart VS Code → <code>pip install pytesseract Pillow</code>')

    spacer(8)

    card_open("Upload Receipt or Screenshot","Supported: JPG, PNG, WEBP, BMP — any bank, wallet, or payment app")
    img_file = st.file_uploader("ocr_up", type=['jpg','jpeg','png','webp','bmp'], label_visibility="collapsed")
    card_close()

    def run_ocr(img_bytes, mime):
        """Try Claude Vision first, then Tesseract, return (text, method_used)."""
        # 1. Claude Vision API
        if CLAUDE_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=CLAUDE_KEY)
                b64 = base64.b64encode(img_bytes).decode()
                resp = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400,
                    messages=[{"role":"user","content":[
                        {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}},
                        {"type":"text","text":
                         "This is a bank/wallet payment receipt. Extract:\n"
                         "MERCHANT: (who received money or service name)\n"
                         "AMOUNT: (number only, e.g. 170)\n"
                         "DATE: (e.g. May 12 2026)\n"
                         "TXN_ID: (transaction/reference ID if visible)\n"
                         "FROM: (sender name if visible)\n"
                         "TO: (recipient name if visible)\n"
                         "TYPE: (Transfer/Payment/Bill etc)\n"
                         "Write UNKNOWN for fields not visible."}
                    ]}])
                return resp.content[0].text, "Claude Vision AI"
            except Exception as e:
                pass  # Fall through to Tesseract

        # 2. Tesseract OCR
        try:
            from PIL import Image
            import pytesseract, io as _io
            for tp in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                       r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                       '/usr/bin/tesseract','/usr/local/bin/tesseract']:
                import os as _os
                if _os.path.exists(tp): pytesseract.pytesseract.tesseract_cmd = tp; break
            img = Image.open(_io.BytesIO(img_bytes))
            # Upscale small images for better accuracy
            w, h = img.size
            if w < 800: img = img.resize((w*2, h*2), Image.LANCZOS)
            text = pytesseract.image_to_string(img, config='--psm 6')
            return text.strip(), "Tesseract OCR"
        except ImportError:
            return "TESSERACT_NOT_INSTALLED", "none"
        except Exception as e:
            return f"OCR_ERROR: {e}", "none"

    def parse_ocr_result(text, method):
        """Parse extracted text into transaction fields."""
        r = {'merchant':None,'amount':None,'date':None,'txn_id':None,'from':None,'to':None,'type':None}

        if method == "Claude Vision AI":
            # Structured output from Claude
            for line in text.split('\n'):
                line = line.strip()
                for key in ['MERCHANT','AMOUNT','DATE','TXN_ID','FROM','TO','TYPE']:
                    if line.upper().startswith(key+':'):
                        v = line.split(':',1)[1].strip()
                        if v.upper() not in ('UNKNOWN','N/A','NONE',''):
                            k2 = key.lower()
                            if k2 == 'amount':
                                try: r['amount'] = float(re.sub(r'[^\d.]','',v))
                                except: pass
                            else: r[k2] = v
        else:
            # Tesseract raw text — smart extraction
            lines = text.split('\n')
            amt_pats = [
                re.compile(r'Rs\.?\s*([\d,]+\.?\d{0,2})', re.I),
                re.compile(r'PKR\s*([\d,]+\.?\d{0,2})', re.I),
                re.compile(r'^([\d,]+\.\d{2})$'),
            ]
            date_pat = re.compile(r'(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})', re.I)
            tid_pat  = re.compile(r'(?:TID|Transaction\s*ID|Ref|TxID)[:\s#]+([A-Za-z0-9]{6,})', re.I)
            to_pat   = re.compile(r'(?:Transferred\s+to|To)[:\s]+(.+)', re.I)
            from_pat = re.compile(r'From[:\s]+(.+)', re.I)

            for line in lines:
                line_s = line.strip()
                if not r['amount']:
                    for p in amt_pats:
                        m = p.search(line_s)
                        if m:
                            try: r['amount'] = float(m.group(1).replace(',','')); break
                            except: pass
                if not r['date']:
                    dm = date_pat.search(line_s)
                    if dm: r['date'] = dm.group(1)
                if not r['txn_id']:
                    tm = tid_pat.search(line_s)
                    if tm: r['txn_id'] = tm.group(1)
                if not r['to']:
                    tom = to_pat.search(line_s)
                    if tom: r['to'] = tom.group(1).strip()[:50]
                if not r['from']:
                    fm = from_pat.search(line_s)
                    if fm: r['from'] = fm.group(1).strip()[:50]

            # Detect service from text
            for svc in ['JazzCash','NayaPay','Easypaisa','HBL','Meezan','Foodpanda','Careem','Daraz','Netflix']:
                if svc.lower() in text.lower():
                    r['merchant'] = svc; break

        return r

    if img_file:
        raw = img_file.read()
        mime = f"image/{img_file.type.split('/')[-1]}" if img_file.type else "image/jpeg"

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(raw, caption="Uploaded image", use_column_width=True)

        with c2:
            with st.spinner("🔍 Extracting transaction data…"):
                txt, method_used = run_ocr(raw, mime)

            if txt in ("TESSERACT_NOT_INSTALLED",) or txt.startswith("OCR_ERROR"):
                status('danger',
                    f'<strong>OCR not available.</strong> {txt}<br><br>'
                    'Fix: Run as Administrator: <code>winget install UB-Mannheim.TesseractOCR</code><br>'
                    'Or set ANTHROPIC_API_KEY to use Claude Vision (no install needed).')
            else:
                status('info', f'Extracted using: <strong>{method_used}</strong>')
                card_open("Raw Extracted Text")
                st.text_area("raw_ocr", txt, height=160, label_visibility="collapsed")
                card_close()

                ext = parse_ocr_result(txt, method_used)
                if ext.get('amount'):
                    card_open("✅ Transaction Detected")
                    xa, xb = st.columns(2)
                    with xa:
                        st.metric("Amount",   f"PKR {ext['amount']:,.0f}")
                        st.metric("Merchant", ext.get('merchant') or ext.get('to') or 'Unknown')
                    with xb:
                        st.metric("Date",     ext.get('date') or 'Unknown')
                        st.metric("Type",     ext.get('type') or 'Transfer/Payment')
                    if ext.get('txn_id'): st.caption(f"TID: {ext['txn_id']}")
                    if ext.get('from'):   st.caption(f"From: {ext['from']}")
                    if ext.get('to'):     st.caption(f"To: {ext['to']}")

                    # ── Edit fields before adding ───────────────────
                    st.write("**Confirm before adding:**")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    with col_e1:
                        confirm_merchant = st.text_input("Merchant", value=ext.get('merchant') or ext.get('to') or 'Unknown', key="ocr_merchant")
                    with col_e2:
                        confirm_amount = st.number_input("Amount (PKR)", value=float(ext['amount']), min_value=0.0, key="ocr_amount")
                    with col_e3:
                        confirm_type = st.selectbox("Type", ["expense","income"], key="ocr_txn_type")

                    confirm_date = st.date_input("Date", value=datetime.now().date(), key="ocr_date")
                    confirm_cat  = st.selectbox("Category", ["Financial","Food & Dining","Utilities","Transportation","Shopping","Entertainment","Healthcare","Other"], key="ocr_cat")

                    if st.button("➕ Add to My Finances", use_container_width=True, type="primary"):
                        new_row = {
                            'date':             pd.Timestamp(confirm_date),
                            'merchant_name':    confirm_merchant,
                            'category':         confirm_cat,
                            'amount':           abs(confirm_amount),
                            'description':      f'OCR: {confirm_merchant}',
                            'transaction_type': confirm_type,
                            'transaction_id':   f'OCR{len(st.session_state.ocr_transactions):04d}',
                        }
                        st.session_state.ocr_transactions.append(new_row)
                        status('success', f'✅ Added PKR {confirm_amount:,.0f} from {confirm_merchant} to your financial data! Go to Dashboard to see it.')

                    if st.session_state.ocr_transactions:
                        st.caption(f"📊 {len(st.session_state.ocr_transactions)} OCR transaction(s) added to your finances this session.")
                        if st.button("🗑️ Clear OCR Transactions", key="clear_ocr"):
                            st.session_state.ocr_transactions = []
                            st.rerun()
                    card_close()
                else:
                    status('warning',
                        'Amount not detected. Tips:<br>'
                        '• Ensure the image is sharp and well-lit<br>'
                        '• Try a screenshot instead of a photo<br>'
                        '• Use Claude Vision (ANTHROPIC_API_KEY) for best results')


elif page=="Reports & Export":
    page_header("Reports & Export","Download financial reports")
    card_open("Configuration")
    xa,xb=st.columns(2)
    with xa: st.write("**Report Type**"); rt=st.selectbox("rt",["Monthly Summary","Annual Summary","Category Analysis","Subscription Audit"],label_visibility="collapsed")
    with xb: st.write("**Format**"); ef=st.selectbox("ef",["Excel (XLSX)","CSV"],label_visibility="collapsed")
    card_close(); card_open(rt)

    if rt=="Monthly Summary":
        months=sorted(df['date'].dt.to_period('M').unique(),reverse=True)
        sel=st.multiselect("Months",months,default=[months[0]] if months else [])
        if sel:
            rows=[]
            for m in sel:
                mdf=df[df['date'].dt.to_period('M')==m]
                inc=mdf[mdf['transaction_type']=='income']['amount'].sum()
                exp=mdf[mdf['transaction_type']=='expense']['amount'].sum()
                rows.append({'Month':str(m),'Income':f"PKR {inc:,.0f}",'Expenses':f"PKR {exp:,.0f}",'Balance':f"PKR {inc-exp:,.0f}",'Transactions':len(mdf)})
            rdf=pd.DataFrame(rows); st.dataframe(rdf,use_container_width=True,hide_index=True)
            if ef=="Excel (XLSX)":
                buf=pd.ExcelWriter('ms.xlsx',engine='openpyxl'); rdf.to_excel(buf,index=False); buf.close()
                with open('ms.xlsx','rb') as f: st.download_button("Download Excel",f.read(),f"monthly_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.download_button("Download CSV",rdf.to_csv(index=False),f"monthly_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    elif rt=="Annual Summary":
        ai2=df[df['transaction_type']=='income']['amount'].sum(); ae2=df[df['transaction_type']=='expense']['amount'].sum()
        sv2=ai2-ae2; sr4=(sv2/ai2*100) if ai2>0 else 0; pd2=(df['date'].max()-df['date'].min()).days; pm=max(1,pd2/30)
        sdf=pd.DataFrame({'Metric':['Total Income','Total Expenses','Net Savings','Savings Rate','Avg Monthly Income','Avg Monthly Expenses','Days Tracked'],
            'Amount':[f"PKR {ai2:,.0f}",f"PKR {ae2:,.0f}",f"PKR {sv2:,.0f}",f"{sr4:.1f}%",f"PKR {ai2/pm:,.0f}",f"PKR {ae2/pm:,.0f}",str(pd2)]})
        st.dataframe(sdf,use_container_width=True,hide_index=True)
        if ef=="Excel (XLSX)":
            buf=pd.ExcelWriter('as.xlsx',engine='openpyxl'); sdf.to_excel(buf,sheet_name='Annual',index=False)
            cs=df[df['transaction_type']=='expense'].groupby('category')['amount'].agg(['sum','count','mean']).reset_index()
            cs.columns=['Category','Total','Count','Avg']; cs.to_excel(buf,sheet_name='By Category',index=False); buf.close()
            with open('as.xlsx','rb') as f: st.download_button("Download Excel",f.read(),f"annual_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.download_button("Download CSV",sdf.to_csv(index=False),f"annual_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    elif rt=="Category Analysis":
        ca5=df[df['transaction_type']=='expense'].groupby('category').agg({'amount':['sum','count','mean','min','max']}).round(0)
        ca5.columns=['Total (PKR)','Count','Avg (PKR)','Min (PKR)','Max (PKR)']; ca5=ca5.sort_values('Total (PKR)',ascending=False)
        st.dataframe(ca5,use_container_width=True)
        if ef=="Excel (XLSX)":
            buf=pd.ExcelWriter('ca.xlsx',engine='openpyxl'); ca5.to_excel(buf); buf.close()
            with open('ca.xlsx','rb') as f: st.download_button("Download Excel",f.read(),f"categories_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.download_button("Download CSV",ca5.to_csv(),f"categories_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")

    elif rt=="Subscription Audit":
        if len(recurring_df)>0:
            au=recurring_df.copy().sort_values('total_annual',ascending=False)
            xa,xb,xc=st.columns(3)
            with xa: st.metric("Subscriptions",len(au))
            with xb: st.metric("Annual PKR",f"PKR {au['total_annual'].sum():,.0f}")
            with xc: st.metric("Monthly PKR",f"PKR {au['total_annual'].sum()/12:,.0f}")
            st.divider()
            dc2=['merchant_name','category','avg_amount','count','total_annual']
            dn2=['Subscription','Category','Monthly (PKR)','Count','Annual (PKR)']
            ad=au[dc2].copy(); ad.columns=dn2; st.dataframe(ad,use_container_width=True,hide_index=True)
            au['u']=(au['count']/au['count'].max()*100).round(0)
            un=au[au['u']<50]
            if len(un)>0: status('warning',f'<strong>{len(un)}</strong> underused subscription(s) — consider cancelling.')
            else: status('success','All subscriptions actively used.')
            if ef=="Excel (XLSX)":
                buf=pd.ExcelWriter('sa.xlsx',engine='openpyxl'); ad.to_excel(buf,sheet_name='Subscriptions',index=False); buf.close()
                with open('sa.xlsx','rb') as f: st.download_button("Download Excel",f.read(),f"subscriptions_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.download_button("Download CSV",ad.to_csv(index=False),f"subscriptions_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")
        else: status('info','No subscriptions detected.')
    card_close()
