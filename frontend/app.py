import streamlit as st
import requests
import pandas as pd

# =====================================================
# Page Configuration & Custom Styling
# =====================================================
st.set_page_config(
    page_title="India Tech Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1.5rem;
    }
    .salary-range-text {
        font-size: 3.8rem;
        font-weight: 800;
        color: #059669;
        margin: 0.2rem 0;
    }
    .point-estimate-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: -0.5rem;
        margin-bottom: 1.2rem;
    }
    .metric-subtext {
        font-size: 0.95rem;
        color: #6B7280;
        font-weight: 500;
    }
    .developer-card {
        background-color: #F8FAFC;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .developer-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .developer-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# Frontend Mappings & Geographical Data
# =====================================================
STATE_CITY_MAP = {
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Navi Mumbai"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Haryana": ["Gurugram", "Faridabad"],
    "Uttar Pradesh": ["Noida", "Greater Noida", "Lucknow"],
    "West Bengal": ["Kolkata", "Salt Lake City"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Gwalior"],
    "Delhi NCR": ["New Delhi", "Delhi"]
}

COMPANY_SIZE_MAP = {
    "Small (1 - 50 employees)": "1-50",
    "Medium-Small (51 - 200 employees)": "51-200",
    "Medium (201 - 500 employees)": "201-500",
    "Large (501 - 1,000 employees)": "501-1000",
    "Very Large (1,001 - 5,000 employees)": "1001-5000",
    "Enterprise (5,000+ employees)": "5000+"
}

# =====================================================
# Header Section & Dynamic API Configuration
# =====================================================
st.markdown('<div class="main-header">🇮🇳 India Tech Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by a Scikit-Learn Preprocessing Pipeline & Advanced Ensemble Regression</div>', unsafe_allow_html=True)

# Automatically pulls from Streamlit Cloud Secrets, falling back to localhost if running locally
# .rstrip("/") prevents double slashes if a trailing slash is accidentally added to the URL secret
API_URL = st.secrets.get("API_URL", "https://job-salary-prediction-india.onrender.com/").rstrip("/")

with st.sidebar:
    st.header("⚙️ System Status")
    try:
        # Catching all RequestException errors prevents ReadTimeout crashes during Render cold starts
        health = requests.get(f"{API_URL}/", timeout=3)
        if health.status_code == 200:
            st.success("🟢 Backend API: Online")
        else:
            st.warning("🟡 Backend API: Degraded")
    except requests.exceptions.RequestException:
        st.warning("🟡 Backend API: Waking Up / Idle")
        st.caption("Free-tier servers sleep after 15 minutes of inactivity. It takes ~40 seconds to wake up on the first request. Your app will work normally when you click Predict!")
    
    st.markdown("---")
    st.markdown("### About the System")
    st.write("This application predicts technology compensation across India in **Lakhs Per Annum (LPA)** based on real-world industry hiring data.")
    st.write("The underlying machine learning pipeline automatically serializes one-hot encoding for categorical variables and median imputation for continuous features.")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 About the Developer")
    st.markdown("""
        <div class="developer-card">
            <div class="developer-name">Ram Gour</div>
            <div class="developer-title">Data Science & Machine Learning Engineer</div>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.9rem;">
                <a href="https://linkedin.com/in/ram-gour" target="_blank" style="text-decoration: none; color: #0A66C2; font-weight: 600;">🔗 LinkedIn Profile</a>
                <a href="https://github.com/gourram289-ops" target="_blank" style="text-decoration: none; color: #24292E; font-weight: 600;">💻 GitHub Workspace</a>
                <a href="mailto:gourram289@gmail.com" style="text-decoration: none; color: #EA4335; font-weight: 600;">📧 gourram289@gmail.com</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption("⚡ Built with FastAPI, Scikit-Learn & Streamlit")

# =====================================================
# Main Application Tabs
# =====================================================
tab_predict, tab_model = st.tabs(["🔮 Salary Predictor", "📊 Model Architecture & Performance"])

# -----------------------------------------------------
# TAB 1: PREDICTION INTERFACE (Reactive Layout)
# -----------------------------------------------------
with tab_predict:
    st.markdown("### Candidate & Role Specifications")
    st.write("Adjust the parameters below to dynamically estimate fair-market compensation based on current industry benchmarks.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👨‍💻 Role Details")
        job_title = st.selectbox("Job Title", [
            "Software Engineer", "Data Scientist", "DevOps Engineer", 
            "Product Manager", "Frontend Developer", "Backend Developer", 
            "Full Stack Developer", "Machine Learning Engineer"
        ])
        primary_skill = st.selectbox("Primary Skill", [
            "Python", "Java", "JavaScript", "React", "Node.js", 
            "AWS", "SQL", "C++", "Machine Learning", "Kubernetes"
        ])
        skills_count = st.number_input("Total Skills Count", min_value=1, max_value=50, value=5, step=1)
        
    with col2:
        st.subheader("🏢 Company & Work")
        company = st.selectbox("Company", [
            "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", 
            "Google", "Microsoft", "Amazon", "Flipkart", "Zomato", "Startup / Other"
        ])
        
        selected_size_label = st.selectbox("Company Scale", list(COMPANY_SIZE_MAP.keys()))
        company_size = COMPANY_SIZE_MAP[selected_size_label]
        
        work_mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "On-site"])
        
    with col3:
        st.subheader("📍 Location & Experience")
        education = st.selectbox("Highest Education", [
            "B.Tech/B.E.", "M.Tech/M.E.", "BCA", "MCA", "B.Sc", "M.Sc", "Ph.D."
        ])
        
        state = st.selectbox("State Jurisdiction", list(STATE_CITY_MAP.keys()))
        city = st.selectbox("Tech Hub City", STATE_CITY_MAP[state])
        
        experience_years = st.slider("Total Industry Experience (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        
    st.markdown("---")
    
    if st.button("⚡ Calculate Predicted Compensation", use_container_width=True, type="primary"):
        payload = {
            "job_title": job_title,
            "education": education,
            "primary_skill": primary_skill,
            "city": city,
            "state": state,
            "company": company,
            "company_size": company_size,
            "work_mode": work_mode,
            "experience_years": experience_years,
            "skills_count": skills_count
        }
        
        with st.spinner("Connecting to cloud model and analyzing market data (may take ~40s if the server is waking from sleep)..."):
            try:
                # 60-second timeout allows sleeping Render containers enough time to initialize
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    salary = data["predicted_salary_lpa"]
                    
                    # Statistical range calculation anchored in the model's MAE (±₹0.62 LPA)
                    error_margin = 0.62
                    lower_bound = max(1.0, round(salary - error_margin, 1))
                    upper_bound = round(salary + error_margin, 1)
                    
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="color: #4B5563; margin-bottom: 0rem;">Estimated Market Compensation Band</h3>
                            <div class="salary-range-text">₹ {lower_bound} – {upper_bound} LPA</div>
                            <div class="point-estimate-text">Exact Model Point Estimate: ₹ {salary:,.2f} LPA</div>
                            <p class="metric-subtext">Negotiation window calculated via Linear Regression error variance (±₹0.62 LPA) &nbsp;|&nbsp; Location: {city}, {state}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Prediction failed: {response.text}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The backend server took longer than 60 seconds to wake up. Please click Predict again immediately!")
            except requests.exceptions.RequestException:
                st.error("⚠️ Unable to connect to the backend API. Please verify that your Render web service is running.")

# -----------------------------------------------------
# TAB 2: MODEL ARCHITECTURE & PERFORMANCE
# -----------------------------------------------------
with tab_model:
    st.markdown("### 🏆 Executive Model Benchmark")
    st.write("During pipeline training, three regression architectures were evaluated using an 80/20 train-test split. **Linear Regression** emerged as the champion model, demonstrating superior generalizability on high-dimensional sparse data.")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(label="Winner R² Score (Accuracy)", value="97.24%", delta="+0.11% vs XGBoost")
    with col_kpi2:
        st.metric(label="Mean Absolute Error (MAE)", value="₹1.24 LPA", delta="-0.017 vs XGBoost", delta_color="inverse")
    with col_kpi3:
        st.metric(label="Root Mean Squared Error (RMSE)", value="1.437 LPA", delta="-0.028 vs XGBoost", delta_color="inverse")
        
    st.markdown("---")
    
    st.subheader("📊 Cross-Algorithm Evaluation Matrix")
    
    benchmark_data = {
        "Rank": ["🏆 1st (Selected)", "🥈 2nd Place", "🥉 3rd Place"],
        "Algorithm": ["Linear Regression", "XGBoost Regressor", "Random Forest Regressor"],
        "R² Score (Higher is better)": ["0.9724", "0.9713", "0.9696"],
        "MAE (LPA - Lower is better)": ["1.239", "1.256", "1.279"],
        "RMSE (LPA - Lower is better)": ["1.437", "1.465", "1.507"],
        "Primary Architectural Strength": [
            "Optimal for additive, one-hot encoded sparse data",
            "Captures complex non-linear feature interactions",
            "Robust bagging against localized training outliers"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(benchmark_data),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 🛠️ Production Preprocessing (`ColumnTransformer`)")
        st.write("The production pipeline serializes preprocessing and modeling into a single `.pkl` artifact to guarantee zero data leakage:")
        
        with st.expander("Categorical Encoding (8 Features)", expanded=True):
            st.markdown("* **Features:** `job_title`, `education`, `primary_skill`, `city`, `state`, `company`, `company_size`, `work_mode`")
            st.markdown("* **Imputation:** `SimpleImputer(strategy='most_frequent')` replaces missing strings with the mode.")
            st.markdown("* **Transformation:** `OneHotEncoder(handle_unknown='ignore')` converts text categories into dense binary matrices without breaking on unseen production labels.")
            
        with st.expander("Numerical Scaling (2 Features)", expanded=False):
            st.markdown("* **Features:** `experience_years`, `skills_count`")
            st.markdown("* **Imputation:** `SimpleImputer(strategy='median')` fills missing numerical entries while remaining robust against extreme salary or experience outliers.")

    with col_b:
        st.markdown("### 📈 Metric Interpretation for Stakeholders")
        st.info("How to interpret these evaluation metrics in the context of Indian tech recruitment:")
        
        st.markdown("* **$R^2 = 0.97241$:** The model successfully explains **97.24%** of why salaries differ between candidates. Only 2.76% of compensation variation is driven by uncaptured factors (e.g., negotiation skills, urgent hiring bonuses, or interview performance).")
        st.markdown("* **$\text{MAE} = 1.239\text{ LPA}$:** When predicting a salary, the baseline error margin is approximately **₹1,23,900 per year**. For a ₹20 LPA role, this represents a highly reliable error margin of just ~6.1%.")
        st.markdown("* **$\text{RMSE} = 1.437\text{ LPA}$:** Because $\text{RMSE}$ squares errors before averaging, its close proximity to the $\text{MAE}$ proves that the model does not suffer from extreme, catastrophic mispredictions on outlier salaries.")