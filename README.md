# RockVsMine

# 🎯 Rock vs Mine Classification using Machine Learning

This project builds a binary classification model using the **Sonar Dataset** to identify whether a sonar signal is bounced back from a **rock** or a **mine**. It includes full preprocessing, model training, evaluation, and predictions using scikit-learn.

---

## 📂 Dataset Info

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- **Format**: CSV
- **Features**: 60 numerical attributes (each representing sonar signal energy in different frequencies)
- **Target Labels**:  
  - `R`: Rock  
  - `M`: Mine

---

## 🔧 Tech Stack

- **Language**: Python 🐍  
- **Libraries**:  
  - `pandas` – data handling  
  - `numpy` – numeric computations  
  - `matplotlib`, `seaborn` – data visualization  
  - `scikit-learn` – machine learning models

---

## 📊 ML Models Used

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Accuracy Evaluation with `train_test_split`

---

## 🧠 Steps Performed

1. **Data Loading**
2. **Exploratory Data Analysis (EDA)**
3. **Data Preprocessing**
   - Label Encoding (`R` → 0, `M` → 1)
   - Feature Scaling
4. **Model Training**
5. **Model Evaluation**
6. **Prediction on New Input**
7. **Model Comparison**

---

## ✅ Results

- Achieved over **90% accuracy** with SVM and Random Forest
- Clean classification with good performance metrics (Precision, Recall, F1-score)

---

## 📌 How to Run

1. Clone this repo  
2. Install dependencies  
```bash
pip install -r requirements.txt
Run the notebook
jupyter notebook Rock-vs-Mine.ipynb
📁 Project Structure

├── Rock-vs-Mine.ipynb
├── sonar.csv
├── README.md
├── requirements.txt
✨ Future Improvements

Add cross-validation
Try ensemble models (e.g., XGBoost)
Deploy using Streamlit for user input
🙋‍♂️ Author

Rahul Pandey
Aspiring Data Scientist | Passionate about Machine Learning & Real-world Problem Solving
