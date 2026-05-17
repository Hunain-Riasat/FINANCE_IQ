# 🧠 AI Financial Intelligence Platform v2.0

> **Upgrade from**: Subscription Manager  
> **Upgrade type**: Additive — zero existing functionality removed

---

## What's New vs Original

| Feature | Original | v2.0 |
|---|---|---|
| Excel / CSV upload | ✅ | ✅ Preserved |
| Dashboard + KPIs | ✅ | ✅ + Smart Alerts banner |
| Analytics charts | ✅ | ✅ + Income vs Expense chart |
| Fraud Detection (Isolation Forest) | ✅ | ✅ + Confidence Score + Scatter plot |
| Recurring Payments | ✅ | ✅ + Billing cycle + Last charge |
| Budget Planning | ✅ | ✅ Preserved |
| Monthly Report | ✅ | ✅ Preserved |
| Bill Reminders | ✅ | ✅ Preserved |
| Reports & Export | ✅ | ✅ Preserved |
| **AI Chat Assistant** | ❌ | ✅ Claude API + memory |
| **Financial Health Score** | ❌ | ✅ 0–100 composite score |
| **Smart Alerts Engine** | ❌ | ✅ Spending spikes, fraud, subscriptions |
| **PDF Statement Parsing** | ❌ | ✅ pdfplumber table extraction |
| **OCR Receipt Upload** | ❌ | ✅ Tesseract + PIL |
| **Synthetic Demo Data Generator** | ❌ | ✅ Pakistani merchants |
| **Enhanced Subscription Detection** | ❌ | ✅ Billing cycle + consistency score |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements_enhanced.txt
```

### 2. Set API keys (optional but recommended)

```bash
# For AI Chat (Claude):
export ANTHROPIC_API_KEY="sk-ant-..."

# For Gemini fallback (optional):
export GEMINI_API_KEY="AIza..."
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

### 3. Run the app

```bash
# Enhanced version (recommended):
streamlit run app_enhanced.py

# Original version (still works):
streamlit run app.py
```

---

## New Files

| File | Purpose |
|---|---|
| `app_enhanced.py` | Main enhanced Streamlit app |
| `ai_utils_enhanced.py` | New AI features: chatbot, OCR, health score, alerts |
| `requirements_enhanced.txt` | Full dependency list |
| `UPGRADE_README.md` | This file |

**Original files untouched:** `app.py`, `ai_utils.py`, `requirements.txt`

---

## Feature Details

### 🤖 AI Chat Assistant
- Located at **AI Chat Assistant** in sidebar
- Powered by Claude API (`claude-sonnet-4-20250514`)
- Maintains conversational memory (last 10 turns)
- Injects your transaction data as context
- Quick-prompt buttons for common questions
- Falls back to Gemini → rule-based if no API key

### 💯 Financial Health Score
- Composite score 0–100 with grade (A–F)
- Five sub-scores: Savings, Consistency, Subscriptions, Risk, Income Stability
- Personalised recommendations per weakness
- Visual progress bars per dimension

### 🚨 Smart Alerts
- Month-over-month spending spikes (>20%)
- Fraud alerts from Isolation Forest
- Underused subscription detection
- Low savings rate warning
- All with action links to relevant pages

### 📄 PDF Statement Parsing
- Upload PDF bank statements directly
- Extracts tables using `pdfplumber`
- Auto-exports to CSV for re-upload and full analysis

### 📸 OCR Upload (NEW page)
- Upload receipts, screenshots, or transaction images
- Tesseract OCR extracts text
- Auto-detects amount, merchant, and date
- Works offline (no API needed)

### 🇵🇰 Pakistani Synthetic Data Generator
- Generates realistic transactions for demo/testing
- Covers: NayaPay, Foodpanda, Careem, LESCO, Daraz, Jazz, Telenor, Netflix, Spotify…
- Configurable months and income level
- Includes salary credits + expense patterns

### 🔄 Enhanced Subscription Detection
- Detects billing cycle (Weekly / Monthly / Quarterly / Annual)
- Shows last charge date
- Consistency score per merchant
- Improved coefficient of variation thresholds

---

## Architecture

```
Subscription_Manager/
├── app.py                    ← Original (unchanged)
├── app_enhanced.py           ← New enhanced entry point
├── ai_utils.py               ← Original ML functions (unchanged)
├── ai_utils_enhanced.py      ← New: chatbot, OCR, health score, alerts
├── requirements.txt          ← Original (unchanged)
├── requirements_enhanced.txt ← New full deps
├── sample_transactions.csv   ← Sample data (unchanged)
└── UPGRADE_README.md         ← This file
```

---

## API Keys Reference

| Key | Required? | Used For |
|---|---|---|
| `ANTHROPIC_API_KEY` | Recommended | AI Chat (Claude), financial analysis |
| `GEMINI_API_KEY` | Optional | Chatbot fallback if Claude unavailable |

Without any API keys, the app works fully in **offline mode** using rule-based responses.

---

## OCR Setup

Tesseract must be installed separately from Python:

- **Windows**: Download from [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

Then install the Python wrapper: `pip install pytesseract Pillow`

---

## Design Principles Preserved

- ✅ Same dark theme (`--bg-base: #0f1117`)
- ✅ Same color palette (blue, teal, gold, red, green)
- ✅ Same DM Sans + DM Mono fonts
- ✅ Same card component style
- ✅ Same status banner pattern
- ✅ Same sidebar branding and credits
- ✅ Same KPI card design
- ✅ Same Plotly chart styling
- ✅ No layout changes to existing pages

---

*Made with ❤️ by Zain, Hunain & Shaheer*
