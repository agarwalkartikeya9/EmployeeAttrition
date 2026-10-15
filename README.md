# Employee_Attrition_Model

# 🚀 Team NEXUS: HR Analytics Portal & Flight Risk Predictor

## 📌 Project Overview
The Team NEXUS HR Analytics Portal is an enterprise-grade Machine Learning application developed for the IIHMF Capstone Defense. This project leverages an advanced 10,000-record dataset to predict employee attrition (flight risk) using a robust 8-phase Machine Learning Lifecycle. 

The backend intelligence is powered by an optimized ensemble of classification algorithms, while the frontend provides HR Managers with a clean, interactive Streamlit dashboard to calculate real-time turnover probabilities for both existing and prospective employees.

---

## 🏗️ Project Architecture (The ML Lifecycle)

This project strictly adheres to the 8-phase data science methodology:

* **Phases 1 & 2: Data Acquisition & Problem Understanding**
  * Sourced a highly detailed 10,000-row proprietary HR dataset featuring complex psychological metrics (e.g., Work-Life Balance, Job Satisfaction, Absenteeism) and financial indicators.
* **Phase 3: Data Preprocessing**
  * Handled binary categorical translation using `LabelEncoder`.
  * Prevented ordinal mathematical bias by applying One-Hot Encoding (`pd.get_dummies`) to multi-class variables like Department and Job Role.
* **Phase 4: Exploratory Data Analysis (EDA)**
  * Generated Univariate (Histograms), Bivariate (Boxplots/Scatter Plots), and Correlation (Heatmaps) analyses using Seaborn and Matplotlib to prove psychological validity before model training.
* **Phase 5: Feature Engineering**
  * Optimized the feature space by isolating the target variable (`Attrition`) and dropping non-predictive identifiers to prevent model distraction.
* **Phases 6 & 7: Model Building & Evaluation**
  * Trained and evaluated three distinct classification models: **Logistic Regression, Random Forest, and XGBoost**.
  * Graded models strictly on classification metrics: Accuracy, Precision, Recall, and F1-Score to identify the most reliable predictive engine.
* **Phase 8: Model Deployment**
  * Exported the winning mathematical model as a `.pkl` artifact.
  * Wrapped the prediction engine in a user-friendly, interactive web application using the Streamlit framework.

---

## 🧠 Advanced Technical Implementation

**The Dynamic Baseline Mapping Architecture:**
During the transition from Jupyter/Colab to the Streamlit UI, a critical dimensionality challenge was resolved. The trained AI requires a 24+ feature input matrix due to One-Hot Encoding, but forcing a user to manually input 24 variables creates poor UI/UX. 

To bridge this gap without causing a matrix shape error or triggering false positives (which occurs if hidden numerical columns default to `0`), we engineered a dynamic baseline mapping system. The application establishes a "Baseline Employee" profile with company-average scores, overlays only the user's specific UI inputs, and uses `.reindex()` to safely handle the remaining structural columns. This prevents data leakage and ensures highly accurate probability scoring in production.

---

## 📂 Repository Structure

Ensure your directory matches this structure before execution:

```text
EmployeeAttrition_TeamNEXUS/
│
├── Dataset/
│   └── employee_attrition_dataset_10000.csv   # The raw data
│
├── Notebook/
│   └── Team_NEXUS_ML_Pipeline.ipynb           # Phases 1-7 Colab environment
│
├── Model/
│   ├── hr_model.pkl                           # The frozen AI brain
│   └── model_columns.pkl                      # The encoded structural matrix
│
├── Streamlit_App/
│   └── app.py                                 # The frontend UI script
│
└── README.md                                  # Project documentation
