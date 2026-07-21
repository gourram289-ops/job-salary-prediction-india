# India Tech Salary Predictor

An end-to-end machine learning application for estimating technology salaries across Indian cities in Lakhs Per Annum (LPA). The project combines a FastAPI backend, a Streamlit-based frontend, and a trained regression pipeline to provide both a REST API and an interactive dashboard.

The system is designed to predict compensation from profile attributes such as job title, education, primary skill, company size, work mode, city, state, and experience.

---

## ✨ What this project does

- Predicts software and data-role salaries in India using a trained regression model.
- Provides a REST API for programmatic predictions.
- Offers an interactive Streamlit UI for non-technical users.
- Benchmarks multiple regression models (Linear Regression, Random Forest, XGBoost) and saves the best-performing pipeline.

---

## 🧠 Model and architecture

The backend loads a serialized machine learning pipeline from the file [backend/salary_model.pkl](backend/salary_model.pkl). The pipeline includes:

- Preprocessing for categorical and numerical features
- One-hot encoding for text-based attributes
- Missing-value handling with imputation
- A trained regression model for salary prediction

The frontend sends user inputs to the API and displays the predicted range along with a simple market benchmark view.

---

## 📁 Project structure

```text
Indian_job_price_prediction/
├── backend/
│   ├── main.py                # FastAPI app and request/response schemas
│   ├── train.py               # Training script for regression models
│   ├── india_tech_jobs.csv    # Dataset used for training
│   ├── salary_model.pkl       # Serialized best-performing model
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   └── app.py                 # Streamlit dashboard for predictions
├── requirements.txt          # Root requirements for UI dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Tech stack

- Python
- FastAPI
- Streamlit
- Scikit-Learn
- XGBoost
- Pandas
- Joblib
- Pydantic

---

## ▶️ Local setup

1. Clone the repository and move into the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

---

## 🚀 Run the application

### Start the backend API

From the project root:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- http://127.0.0.1:8000/docs for Swagger documentation
- http://127.0.0.1:8000/predict for prediction requests

### Start the frontend UI

In a second terminal:

```bash
cd frontend
streamlit run app.py
```

The Streamlit app will open in your browser.

---

## 🔮 API usage example

You can send a request like this:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Data Scientist",
    "education": "M.Tech",
    "primary_skill": "Machine Learning",
    "city": "Hyderabad",
    "state": "Telangana",
    "company": "Microsoft",
    "company_size": "10000+",
    "work_mode": "Remote",
    "experience_years": 3.0,
    "skills_count": 8
  }'
```

Example response:

```json
{
  "predicted_salary_lpa": 22.8,
  "currency": "INR (Lakhs Per Annum)"
}
```

---

## 🧪 Retrain the model

If you want to re-train the model locally:

```bash
cd backend
python train.py
```

This script loads the dataset, compares multiple regressors, evaluates performance, and saves the best model to [backend/salary_model.pkl](backend/salary_model.pkl).

---

## 📊 Example model performance

The training script evaluates several algorithms and reports metrics such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared score (R²)

The current training workflow compares Linear Regression, Random Forest, and XGBoost to choose the best model for deployment.

---

## 🌐 Live deployments

You can test the live application and explore the backend REST endpoints through the links below:

- 🖥️ Frontend Interactive Dashboard: [Launch on Streamlit Cloud](https://job-salary-prediction-india.streamlit.app/)
- ⚡ Backend API Documentation (Swagger UI): [Explore API on Render](https://job-salary-prediction-india.onrender.com/docs)
- 🔍 Raw Health Check Endpoint: [https://job-salary-prediction-india.onrender.com](https://job-salary-prediction-india.onrender.com/)

---

## 🙋 Notes

- The app currently predicts salaries in Indian LPA terms.
- The model is best used as a rough benchmark rather than a guaranteed offer estimate.
- Inputs like negotiation style, bonus structure, and company-specific compensation policies may influence actual salaries beyond the model’s training data.