  Credit_Risk_Probability
 A project to create a Credit Scoring Model using the data provided by the e-Commerce platform.

 Task-1
    Credit Scoring Business Understanding
1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
ANSWER
The Basel II Accord encourages financial institutions to adopt risk-sensitive practices, especially for credit risk measurement and capital allocation. This requires models that are not only accurate but also interpretable and auditable. A transparent credit scoring model ensures regulators can trace how risk scores are derived and confirm that risk is adequately accounted for in the bank’s capital reserves. Thus, our model must be well-documented, explainable (e.g., using logistic regression or decision trees), and aligned with Basel II’s requirements for validation, governance, and ongoing monitoring.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

ANSWER
Without a labeled "default" outcome in the dataset, it's impossible to train a supervised learning model directly on actual defaults. Instead, we engineer a proxy label based on customer behavior (e.g., Recency, Frequency, Monetary value) to identify disengaged users likely to default. While this allows us to approximate risk and build a working model, it introduces the risk of   label leakage or bias  —some low-risk users may be misclassified, leading to false rejections or poor lending decisions. The business must carefully validate that this proxy approximates real-world default behavior to avoid regulatory or financial consequences.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?


ANSWER

  -->Simple models   such as logistic regression with Weight of Evidence (WoE) encoding offer   interpretability, transparency, and ease of regulatory approval  . They allow financial analysts and auditors to understand the logic behind each decision. However, their predictive power is limited and may not capture nonlinear interactions.

  -->Complex models   like Gradient Boosting Machines (GBM) offer   higher predictive accuracy  , especially with diverse and nonlinear features. However, they act more like “black boxes,” making them harder to explain, validate, and regulate. In regulated contexts, the optimal strategy is often to balance both: start with interpretable models, validate their limitations, and then justify the use of more complex models with proper documentation, post-hoc explanation tools (like SHAP), and monitoring systems.


TASK-2

Based on the output of the EDA, here are the top 5 most important insights:

1.	Dominant Currency and Country: All transactions are in UGX and originate from country code 256. This suggests the data is specific to a particular region (likely Uganda).

2.	High Frequency of Financial Services and Airtime Transactions: The product categories 'financial_services' and 'airtime' make up the vast majority of transactions, indicating these are the core services being utilized.

3.	Channel 3 is the Most Used Channel: 'ChannelId_3' is the dominant channel for transactions, significantly more so than other channels.

4.	Transactions Peak During Business Hours: The plot of transactions by hour shows a clear peak during typical business hours, suggesting that transaction activity is correlated with the workday.

5.	Strong Correlation Between Amount and Value: There is a very high positive correlation (0.99) between the 'Amount' and 'Value' of transactions, indicating that these two features are nearly identical or directly proportional. Additionally, both 'Amount' and 'Value' show a moderate positive correlation with 'FraudResult', which could be a significant indicator for fraud detection.


Credit_Risk_Probabilit/
├── data/
│   └── data.csv               # Raw transaction data
│   └── processed_data.csv
│   └── processed_data_with_risk.csv  # Feature-engineered dataset
├── models/
│   └── random_forest_best.pkl     # Trained and saved model
├── src/
│   ├── Data_Processor.py          # Custom feature transformers
│   ├── Trainer/
|   |    └── Model_trainer.py      # Model training and tracking
│   ├──Task-4  
│   │    └── proxy_target_variable.py 
│   │  
│   ├──pipeline.py
│   │
│   └── api/
│       ├── main.py                # FastAPI service
│       └── pydantic_models.py     # Request/response validation
├── tests/
│   └── test_data_processing.py   # Unit tests
├── notebooks/
│   └── 1.0-eda.ipynb             # Exploratory Data Analysis
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD workflow
├── Dockerfile                     # Containerization
├── docker-compose.yml             # Local development orchestration
├── requirements.txt               # Python dependencies
└── README.md                      # This file



🧩 Task Breakdown 
Task 1
✔️ Understanding Credit Risk and Basel II Accord's impact on model interpretability

Task 2
✔️ EDA: Identified patterns, outliers, and insights from transaction data

Task 3
✔️ Feature Engineering: Built custom transformers for preprocessing

Task 4
✔️ Proxy Target Creation: Used KMeans clustering to create is_high_risk label

Task 5
✔️ Model Training and Tracking: Trained multiple models with hyperparameter tuning

Task 6
✔️ Deployment and CI/CD: Built FastAPI service and added testing + deployment


 🚀 How to Run 
1. 📥 Clone the Repository 
bash
git clone https://github.com/rediett/Credit_Risk_Probabilit.git 
cd Credit_Risk_Probabilit
 
2. 🔧 Install Dependencies 
Make sure you have Python 3.9+ installed. 
bash 
pip install -r requirements.txt
 
3. 🧹 Preprocess Data 
Run your feature engineering pipeline: 
bash
python src/Data_Processor.py
 
4. 🧪 Train Models 
Train and evaluate models using: 
bash
cd src/Trainer
python Model_trainer.py 

This script will: 

 - Load data
 - Build pipelines
 - Train Random Forest and Logistic Regression
 - Track experiments using MLflow  

5. 🐳 Run the API Locally 

Start the FastAPI service: 
bash
uvicorn src.api.main:app --reload

Then open:
👉 http://localhost:8000/docs  

Test the /predict endpoint with sample input. 

