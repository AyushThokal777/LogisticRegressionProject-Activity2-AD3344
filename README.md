# 🏥 Heart Disease Risk Prediction (Logistic Regression)

An end-to-end, beginner-friendly Machine Learning project using **Basic Logistic Regression** to solve a real-world healthcare challenge: predicting cardiovascular disease risk based on patient diagnostic indicators.

---

## 📌 Project Overview & Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of death globally. Early diagnosis and risk stratification allow doctors and healthcare professionals to initiate timely preventive treatments and lifestyle interventions.

This project implements a clean binary classification pipeline that:
- Analyzes clinical parameters (resting blood pressure, cholesterol, max heart rate, exercise angina, ST depression, etc.).
- Trains an interpretable **Logistic Regression** model without unnecessary black-box complexity.
- Evaluates diagnostic performance through accuracy, confusion matrix, precision, recall, and feature importance.

---

## 📁 Repository Structure

```text
├── Heart_Disease_Prediction_Logistic_Regression.ipynb   # Main Google Colab / Jupyter notebook (with pre-rendered outputs)
├── heart.csv                                           # Kaggle clinical dataset
├── Logistic_Regression_Model.ipynb                     # Reference baseline notebook
└── README.md                                           # Project documentation
```

---

## 📊 Dataset Details

* **Source:** Kaggle ([`johnsmith81/heart-disease-dataset`](https://www.kaggle.com/datasets/johnsmith81/heart-disease-dataset))
* **Observations:** 303 patient records
* **Target Variable (`target`):**
  * `0`: No heart disease / Healthy
  * `1`: Heart disease detected

### Key Clinical Features:
| Feature | Description |
| :--- | :--- |
| `age` | Patient age in years |
| `sex` | Patient sex (`1` = male, `0` = female) |
| `cp` | Chest pain type (0 to 3) |
| `trestbps` | Resting blood pressure (in mm Hg upon hospital admission) |
| `chol` | Serum cholesterol in mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (`1` = true, `0` = false) |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved during stress test |
| `exang` | Exercise-induced angina (`1` = yes, `0` = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0-3) colored by fluoroscopy |
| `thal` | Thalassemia status |

---

## ⚙️ Machine Learning Pipeline

1. **Data Acquisition:** Automatic download via `kagglehub` or fallback to local `heart.csv`.
2. **Exploratory Data Analysis (EDA):** Verifying shapes, data types, and ensuring zero missing values.
3. **Data Preprocessing:** Splitting independent variables ($X$) and target label ($y$).
4. **Data Splitting:** 80% training set and 20% testing set using `train_test_split(..., test_size=0.2, random_state=42)`.
5. **Feature Standardization:** Applying `StandardScaler` to normalize numeric ranges so large values (e.g., cholesterol) do not skew optimization.
6. **Model Training:** Fitting a baseline `LogisticRegression(max_iter=1000)` via scikit-learn.
7. **Evaluation:** Assessing performance via **Accuracy Score**, **Confusion Matrix**, and **Classification Report** (Precision, Recall, F1-Score).
8. **Interpretability:** Extracting learned weights/coefficients to discover top clinical risk factors.

---

## 📈 Performance & Results

- **Model Accuracy:** `~93.44%` on unseen test data.
- **Confusion Matrix:**
  * High True Negative and True Positive rates.
  * Low False Negative rate (vital in medical screening to prevent missing at-risk patients).
- **Key Predictors Identified:**
  * Fluoroscopy major vessels (`ca`), Maximum heart rate (`thalach`), ST depression (`oldpeak`), and Chest pain type (`cp`).

---

## 🚀 How to Run the Notebook

### Option A: Running on Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `Heart_Disease_Prediction_Logistic_Regression.ipynb`.
3. *(Optional)* Upload `heart.csv` into Colab's left sidebar file pane `/content/` if working offline.
4. Click **Runtime** > **Run all** (`Ctrl + F9`).

### Option B: Running Locally (Jupyter / VS Code)
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn kagglehub
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook Heart_Disease_Prediction_Logistic_Regression.ipynb
   ```

---

## 💡 Why Logistic Regression for this Problem?

In real-world healthcare diagnostics:
- **Interpretability:** Doctors need to understand *why* a prediction was made. Logistic Regression produces direct coefficients (odds ratios) for each medical test.
- **Low Overfitting Risk:** On small-to-medium clinical cohorts, simple linear decision boundaries generalize reliably.
- **Fast & Lightweight:** Requires minimal compute resources and can run directly in resource-constrained hospital hardware or edge devices.
