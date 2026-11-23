# 📉 Customer Churn Risk Predictor

An interactive Streamlit app that estimates how likely a telecom customer is to leave (churn), using a **Random Forest** trained on the [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) dataset.

It turns the work from the notebook (`customer_churn_project_2.ipynb`) into a tool a retention team can use: a risk score, a risk level, and concrete ideas to keep the customer.

## Features

| Tab | What it does |
|-----|--------------|
| **🔮 Single customer** | Describe one customer and get a churn probability on a gauge, a **LOW / MEDIUM / HIGH** badge, how they compare to the average customer, suggested retention actions, a *"what would lower this risk?"* table, and a same-customer contract comparison. Includes three one-click scenarios from the notebook. |
| **📂 Score a customer list** | Upload a CSV, get every customer scored and ranked (the notebook's *retention list*), with summary metrics, a risk distribution chart, and a downloadable result. |
| **🧠 About the model** | Top-10 feature importances, model facts, and a plain-language guide to reading the results (accuracy trap, precision vs recall, thresholds). |

The sidebar lets you move the **risk thresholds** (notebook defaults: MEDIUM ≥ 0.30, HIGH ≥ 0.50) and optionally upload a different model file.

## Project structure

```
churn-app/
├── app.py              # Streamlit UI (entry point for deployment)
├── churn_core.py       # Prediction logic: loading, encoding, risk levels, what-if
├── model.pkl           # ← YOUR trained model goes here (see below)
├── requirements.txt
├── .streamlit/config.toml
├── tests/test_core.py
├── .gitignore
└── README.md
```

## 1. Add your trained model

Put your saved model next to `app.py`. The app auto-detects these names (first match wins):
`model.pkl`, `model.joblib`, `churn_model.pkl`, `churn_model.joblib`, `random_forest.pkl`, `random_forest_model.pkl`, `rf_model.pkl`, or any `*.pkl` / `*.joblib` in the folder or in `models/`.

If you have not exported it yet, run this in the notebook after training:

```python
import joblib, sklearn
joblib.dump(model, "model.pkl")
print("scikit-learn version:", sklearn.__version__)   # ← pin this in requirements.txt
```

**The model must be the Random Forest trained on the notebook's one-hot-encoded features** (`X` after `pd.get_dummies`). The app re-creates that encoding for you. If your model was trained as a scikit-learn `Pipeline` that accepts the raw columns, that works too.

To use a different filename, set `MODEL_PATH` (environment variable, or a Streamlit secret):

```toml
# .streamlit/secrets.toml  (local)  or  App settings → Secrets  (Streamlit Cloud)
MODEL_PATH = "models/my_forest.pkl"
```

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## 3. Deploy with GitHub + Streamlit Community Cloud

1. **Pin scikit-learn** in `requirements.txt` to the version you trained with (a mismatch can break or silently change a pickled model).
2. **Create a GitHub repo and push:**
   ```bash
   git init
   git add .
   git commit -m "Churn risk predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**, choose your repo and branch, and set the main file path to **`app.py`**.
4. Click **Deploy**. Every `git push` to the branch redeploys automatically.

> **Model file size:** GitHub rejects files over 100 MB, and a 100-tree Random Forest can get large. If `model.pkl` is too big, save with compression (`joblib.dump(model, "model.pkl", compress=3)`), reduce `n_estimators` / `max_depth`, or use [Git LFS](https://git-lfs.com).

## Batch CSV format

One customer per row. `TotalCharges` is optional; blank values are estimated as `tenure × MonthlyCharges`, like in the notebook. Use the **Download CSV template** button in the app for a working example.

| Column | Allowed values |
|--------|----------------|
| `tenure` | whole months, 0–72 |
| `MonthlyCharges` | number |
| `TotalCharges` | number (optional) |
| `SeniorCitizen` | `0` or `1` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `PaperlessBilling` | `Yes`, `No` |
| `TechSupport` | `Yes`, `No`, `No internet service` |
| `OnlineSecurity` | `Yes`, `No`, `No internet service` |

## How it maps to the notebook

| Notebook | App |
|----------|-----|
| 4 numeric + 6 categorical features | Same 10 inputs |
| `pd.get_dummies(..., dtype=int)` | `churn_core.to_model_input` |
| Blank `TotalCharges` for brand-new customers | `churn_core.clean_raw` |
| `predict_proba(...)[:, 1]` | Gauge and probability |
| HIGH ≥ 0.5, MEDIUM ≥ 0.3, else LOW | Sidebar sliders (same defaults) |
| Retention list sorted by risk | Batch tab |
| `predict_churn_risk(...)` with typical-customer defaults | Single-customer tab (all 10 fields are editable) |
| Top-10 feature importances | About the model tab |

## Tests

```bash
pip install pytest
pytest -q
```

## Limitations

- The risk score reflects patterns in a public dataset from one telecom company. Re-train on your own customer data before using it for real decisions.
- The *what would lower this risk?* table shows how the model's score changes, **not** a proven effect. Correlation is not causation, so A/B test retention offers.
- Class imbalance means accuracy alone is misleading. Judge the model by recall and precision for churners, as in the notebook.

## Possible next steps

- Try `class_weight="balanced"`, more thresholds (0.2, 0.25), or extra features (`Partner`, `Dependents`, `StreamingTV`), as in the notebook's challenge.
- Add cost inputs (offer cost vs customer value) to choose a threshold by expected profit.
- Add SHAP explanations for per-customer drivers.
