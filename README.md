# 💰 Insurance Cost Prediction App

A **Streamlit-based web application** that predicts insurance costs based on an individual's health and lifestyle factors using **Random Forest** and **XGBoost** machine learning models.

This project includes:
- A **Streamlit app** for interactive predictions.
- A **Jupyter Notebook** demonstrating data preprocessing, model training, evaluation, and model export using `joblib`.

---

## 🚀 Features

- Predict insurance cost using **two powerful ML models**:
  - Random Forest Regressor
  - XGBoost Regressor
- Combines predictions to show an **average insurance cost estimate**.
- Displays **feature importance** for both models.
- Intuitive UI for entering health-related details.
- Built for easy deployment on **Streamlit Cloud** or **local systems**.

---

## 🧾 Input Parameters

| Feature Name | Type | Description |
|---------------|-------|-------------|
| **Age** | Numeric | Age of the individual (18–100 years) |
| **Diabetes** | Yes/No | Whether the person has diabetes |
| **Blood Pressure Problems** | Yes/No | Presence of blood pressure issues |
| **Known Allergies** | Yes/No | Known allergies (if any) |
| **Any Chronic Diseases** | Yes/No | Presence of chronic conditions |
| **Any Transplants** | Yes/No | Whether the person has had a transplant |
| **History of Cancer in Family** | Yes/No | Family history of cancer |
| **Number of Major Surgeries** | Numeric | Count of major surgeries undergone |
| **BMI** | Numeric | Body Mass Index value (10.0–60.0) |

All **Yes/No** values are internally converted to **1/0** before prediction.

---

## ⚙️ How It Works

1. User enters personal and medical details in the app.
2. Input data is converted into a numeric feature array.
3. Both the **Random Forest** and **XGBoost** models predict the insurance cost.
4. The app displays:
   - Individual model predictions.
   - Average prediction.
   - Feature importance charts for both models.

---

## 🧩 Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/insurance-cost-prediction.git
cd insurance-cost-prediction

python -m venv venv
venv\Scripts\activate       # for Windows
source venv/bin/activate    # for Mac/Linux

pip install -r requirements.txt

▶️ Running the Streamlit App
insurence_app.py
```
### Project Structure
Insurance-Cost-Prediction/
│
├── Insurence_cost_pred/
│   ├── insurance_app.py                  # Streamlit app for insurance cost prediction
│   ├── Insurence_Cost_Prediction_.ipynb  # Notebook for data preprocessing, training & evaluation
│   │
│   ├── rf_insurence.pkl                  # Trained Random Forest model
│   ├── xgb_insurence.pkl                 # Trained XGBoost model
│   │
│   │
│   ├── requirements.txt                  # Required Python packages
│   │
├── Insurence_cost_pred.pynb          # Additional notebook for experimentation or improvements
└── README.md                         # Project documentation (this file)

