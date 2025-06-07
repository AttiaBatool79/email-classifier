# Email Classifier Project

This project is a **Machine Learning-based Email Classifier** that detects whether an email is **spam** or **not spam (ham)** using various natural language processing (NLP) and machine learning techniques. This solution helps in automatic email filtering for better productivity and email security.

---

## Repository Contents

| File Name                    | Description |
|-----------------------------|-------------|
| `ML_Project.ipynb`          | Main Jupyter notebook containing code for data preprocessing, model training, and evaluation |
| `spam.csv`                  | Dataset with labeled emails (spam/ham) |
| `ML Project presentation.pptx` | Presentation slides for project overview |
| `PROJECT PROPOSAL.pdf`      | Initial proposal describing the problem and methodology |
| `final ML project report.pdf` | Final written report including results, discussion, and conclusion |
| `README.md`                 | This documentation file |

---

## Project Objective

To develop and evaluate a machine learning model that can accurately classify emails as spam or ham, using text processing and classification algorithms. The end goal is to reduce the burden of spam on users.

---

## Tools and Technologies Used

- **Programming Language:** Python  
- **IDE:** Jupyter Notebook  
- **Libraries Used:**
  - `pandas` – for data handling
  - `numpy` – for numerical operations
  - `scikit-learn` – for model building and evaluation
  - `matplotlib` / `seaborn` – for data visualization
  - `nltk` – for text preprocessing (if used)

---

## Dataset Information

- **File:** `spam.csv`
- **Attributes:**
  - `label`: Indicates whether the message is spam or ham.
  - `message`: The content of the email.

---

## Workflow and Methodology

### 1. Data Preprocessing
- Removing unnecessary columns
- Converting all text to lowercase
- Removing punctuation, stopwords
- Tokenization and stemming/lemmatization

### 2. Feature Engineering
- Using **TF-IDF Vectorizer** to convert text into numerical features

### 3. Model Building
- Trained and evaluated multiple models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)

### 4. Model Evaluation
- **Metrics used:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## Results

- The model achieved **X% accuracy** on the test data. *(Update this from your results)*
- Logistic Regression and Naive Bayes gave competitive results, suitable for real-time deployment.

---

## How to Run This Project

1. Clone this repository:
   ```bash
   git clone https://github.com/AttiaBatool79/email-classifier.git
   cd email-classifier
