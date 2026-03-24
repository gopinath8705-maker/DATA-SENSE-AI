# 📊 DataSense AI v2

> Advanced AI-powered data analytics platform — upload any dataset and get instant insights, ML predictions, anomaly detection, and a downloadable PDF report.

---

## ✨ What's New in v2

| Feature | Description |
|---|---|
| 📸 Image/PDF Upload (OCR) | Extract tables from images & PDFs via GPT-4o Vision or pytesseract |
| 🔮 ML Predictions | Auto train/evaluate 4 ML models + feature importance + forecast |
| 🎙️ Voice Queries | Record or upload audio — transcribed via OpenAI Whisper |
| 📄 PDF Report Export | One-click full analytics report with charts, stats & recommendations |
| 💡 AI Recommendations | Actionable data quality & feature engineering suggestions |
| 🚨 Anomaly Detection | Z-Score, IQR, and Isolation Forest methods |
| 📊 Quality Scoring | 5-dimension data quality score (0–100) with grade |
| ✨ NL Chart Generation | Type *"bar chart of sales by region"* and get an instant chart |
| 🤖 Chart Explanations | AI explains every chart's key insight in plain English |
| 📊 Dashboard | Auto KPI cards, filters, and multi-chart dashboard |

---

## 🗂️ Project Structure

```
datasense_ai/
├── app.py                        # Main app (9 tabs)
├── requirements.txt
├── README.md
├── utils/
│   ├── data_loader.py            # CSV/Excel loading
│   ├── ocr_extractor.py          # Image/PDF OCR table extraction
│   ├── analysis.py               # Stats, column typing, insights
│   ├── visualizations.py         # Plotly charts
│   ├── ai_engine.py              # LLM explanations & chat
│   ├── ml_engine.py              # ML training, evaluation, forecasting
│   ├── quality_anomaly.py        # Quality scoring + anomaly detection
│   ├── recommendations.py        # AI recommendations + NL chart gen
│   ├── voice_query.py            # Audio recording + Whisper transcription
│   └── pdf_report.py             # ReportLab PDF generation
└── components/
    ├── dashboard.py              # Auto dashboard
    └── ui_helpers.py             # CSS + insight cards
```

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Tesseract for OCR fallback
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501**

---

## 🔑 OpenAI API Key (optional but recommended)

| Feature | Without Key | With Key |
|---------|-------------|----------|
| Data preview & stats | ✅ | ✅ |
| Auto insights | ✅ | ✅ |
| ML predictions | ✅ | ✅ |
| Anomaly detection | ✅ | ✅ |
| PDF export | ✅ | ✅ |
| NL chart generation | ✅ rule-based | ✅ GPT-powered |
| Chart explanations | ✅ rule-based | ✅ GPT-powered |
| AI dataset explanation | ✅ rule-based | ✅ GPT-powered |
| Chat with data | ✅ rule-based | ✅ LangChain Agent |
| Image OCR | ✅ pytesseract | ✅ GPT-4o Vision |
| Voice queries | ❌ | ✅ Whisper |

Enter your key in the sidebar (never stored — session only).

---

## 📦 Tech Stack

- **Frontend/Backend**: Streamlit
- **Charts**: Plotly
- **ML**: scikit-learn
- **AI**: OpenAI GPT-3.5/4o + Whisper + LangChain
- **PDF**: ReportLab
- **OCR**: pdfplumber + pytesseract + GPT-4o Vision
