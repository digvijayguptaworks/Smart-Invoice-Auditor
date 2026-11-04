# 🧾 Smart Invoice Auditor

**An end-to-end machine learning system that predicts vendor freight costs and flags risky invoices for manual review.**

Built with Python, SQL, scikit-learn and Streamlit.

🔗 **Live App:** [smart-invoice-auditor.streamlit.app](https://smart-invoice-auditor.streamlit.app/)

---

## 📌 Problem Statement

Companies receive thousands of vendor invoices. Two things make this hard:

- **Freight cost is unknown until the invoice arrives.** This distorts landed cost, margins, inventory planning and vendor negotiation.
- **Manual invoice checking does not scale.** Abnormal charges, mismatched totals or delays can signal errors, disputes or overcharging, and money quietly leaks out.

## 💡 Solution

This project builds two machine learning models and a web app:

1. **Freight Cost Predictor (Regression):** predicts the expected freight for an invoice from its dollar value, so teams can budget before the bill arrives and spot unusually high freight.
2. **Invoice Risk Flagger (Classification):** predicts whether an invoice should be sent for **manual approval** or is **safe for auto-approval**, so finance teams focus only on risky invoices.
3. **Streamlit App:** a simple interface where anyone can enter invoice details and get instant predictions.

---

## 📸 App Preview

_Add screenshots here (for example `images/freight_page.png` and `images/flag_page.png`):_

```
![Freight Cost Prediction](images/freight_page.png)
![Invoice Risk Flagging](images/flag_page.png)
```

---

## 🗂️ Dataset

Data is stored in a SQLite database (`inventory.db`) with two main tables:

| Table | Description |
|---|---|
| `vendor_invoice` | What the vendor billed: quantity, dollars, freight, invoice and payment dates |
| `purchases` | What was ordered and received: brands, quantity, dollars, receiving date |

> ⚠️ The database file (about 400 MB) is not included in this repository because of GitHub's file size limit. Place your own copy at `data/inventory.db` to retrain the models. The deployed app only needs the saved models, not the database.

---

## 🔬 Approach

### Freight Cost Prediction
- Explored the data with a correlation heatmap and scatter plots
- Used a **T-test** to show that small orders pay more freight per unit than large orders
- Compared **Linear Regression, Decision Tree and Random Forest**
- Selected the best model by lowest MAE and saved it with `joblib`

### Invoice Risk Flagging
- Used SQL (`WITH`, `GROUP BY`, `LEFT JOIN`, `julianday`) to aggregate purchase data to one row per PO and join it with invoices
- Created the flag label using business rules:
  - Invoice total differs from PO item total by more than **$5**, or
  - Average receiving delay is more than **10 days**
- Used **T-tests** for feature selection
- Compared **Logistic Regression, Decision Tree and Random Forest**
- Tuned the Random Forest with **GridSearchCV** (the gain was small, so the deployed model is the standard Random Forest trained on five features)

---

## 📊 Results

| Model | Best Algorithm | Performance |
|---|---|---|
| Freight Prediction | Linear Regression | MAE ≈ $24, RMSE ≈ $124, R² ≈ 97% |
| Invoice Flagging | Random Forest | Accuracy 0.88, Precision 0.93, Recall 0.72, F1 0.81 |

When the flagger marks an invoice, it is right about **93%** of the time, and it catches about **72%** of risky invoices.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Database | SQLite, SQL |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistics | SciPy (T-tests) |
| Machine learning | scikit-learn |
| Model saving | Joblib |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version control | Git, GitHub |

---

## 📁 Project Structure

```
Smart-Invoice-Auditor/
├── app.py                          # Streamlit app
├── requirements.txt
├── inference/
│   ├── __init__.py
│   ├── predict_freight.py          # Freight prediction module
│   └── predict_invoice_flag.py     # Invoice flag prediction module
├── freight_cost_predcition/
│   ├── data_preprocessing.py       # Load, prepare and split data
│   ├── model_evaluation.py         # Train and evaluate models
│   ├── train.py                    # Train pipeline, saves best model
│   └── models/                     # Saved .pkl models
├── notebooks/                      # Exploration and experiments
└── README.md
```

---

## ▶️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/digvijayguptaworks/Smart-Invoice-Auditor.git
cd Smart-Invoice-Auditor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```

**4. (Optional) Retrain the freight model**

Place `inventory.db` in the `data/` folder, then:
```bash
cd freight_cost_predcition
python train.py
```

---

## ⚠️ Limitations

- The flag label is **rule-based** ($5 mismatch, 10-day delay), not taken from real audit history. The model learns to automate this rule rather than discover new fraud patterns.
- The flagging model cannot see average receiving delay, so it cannot catch invoices flagged only for late delivery, which limits recall.
- The freight model uses a single feature (invoice dollars), so it is less accurate for very large or unusual invoices.
- The app scores one invoice at a time from manual input; in production, the inference modules would be called automatically on new invoices.

## 🚀 Future Improvements

- Batch upload of invoices (CSV) with downloadable results
- Use real audit or approval history as the label
- Add more features such as vendor, brand and seasonality
- Add explainability (for example SHAP) to show why an invoice was flagged
- Connect the models to the invoice database for automatic scoring

---

## 👤 Author

**Digvijay Gupta**
GitHub: [digvijayguptaworks](https://github.com/digvijayguptaworks)
Portfolio: https://portfolio-25ehb76u0-digvijay14.vercel.app
