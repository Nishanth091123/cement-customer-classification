\# 🏗️ Cement Customer Classification



\## 📌 Project Overview



This project uses Machine Learning to classify cement customers into three categories:



\- Low Value

\- Regular

\- High Value



The model uses customer purchase behavior, sales performance, payment details, relationship history, and complaint information to predict the customer category.



\---



\## 🎯 Objective



The main objective of this project is to identify customer value categories using historical customer data.



This can help businesses:



\- Identify high-value customers

\- Understand regular customers

\- Identify low-value customers

\- Support customer relationship management

\- Improve business decision-making



\---



\## 📊 Dataset



Dataset contains \*\*1000 customer records\*\*.



\### Features



\- Customer Type

\- Region

\- Monthly Quantity

\- Purchase Frequency

\- Average Order

\- Payment Delay Days

\- Credit Limit

\- Relationship Years

\- Previous Month Sales

\- Current Month Sales

\- Growth Percentage

\- Complaint Count



\### Target



Customer Category:



\- Low Value

\- Regular

\- High Value



\---



\## 🤖 Machine Learning Models



The following classification models were evaluated:



1\. Logistic Regression

2\. Random Forest

3\. Decision Tree

4\. K-Nearest Neighbors (KNN)

5\. Support Vector Machine (SVM)



Decision Tree was further optimized using GridSearchCV.



\---



\## 🏆 Model Performance



| Model | Accuracy |

|---|---:|

| Tuned Decision Tree | 92.50% |

| Decision Tree | 91.50% |

| Logistic Regression | 87.50% |

| Random Forest | 87.50% |

| SVM | 84.00% |

| KNN | 78.00% |



\### Final Model



\*\*Tuned Decision Tree\*\*



\*\*Test Accuracy: 92.50%\*\*



\---



\## 🔧 Model Optimization



GridSearchCV was used to find the best Decision Tree parameters.



Best Parameters:



\- max\_depth = 7

\- min\_samples\_leaf = 1

\- min\_samples\_split = 2



The tuned model achieved \*\*92.50% test accuracy\*\*.



\---



\## 📈 Model Evaluation



The final model was evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix

\- Feature Importance



Weighted results:



\- Precision: 93%

\- Recall: 93%

\- F1 Score: 93%



\---



\## 💻 Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Streamlit

\- Jupyter Notebook



\---




## 🌐 Live Streamlit Application

The machine learning model is deployed as an interactive Streamlit web application.

**Live Application:** PASTE-YOUR-STREAMLIT-LINK-HERE

Users can enter customer details and get:

* Predicted Customer Category
* Prediction Probability
* Automatically calculated Growth Percentage
* Final Model Information

**Final Model:** Tuned Decision Tree
**Test Accuracy:** 92.50%


The project includes a Streamlit web application where users can enter customer details and receive:



\- Predicted Customer Category

\- Prediction Probability

\- Final Model Information



\---



\## 📁 Project Structure



```text

Cement Customer Classification/

│

├── app.py

├── cement\_customer\_classification.ipynb

├── cement\_customer\_classification\_1000.csv

├── cement\_customer\_final\_model.pkl

├── cement\_customer\_feature\_columns.pkl

├── cement\_customer\_scaler.pkl

└── model\_comparison\_results.csv

