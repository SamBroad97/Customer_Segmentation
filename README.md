# Customer Segmentation Using Behavioural & Demographic Data

This project builds a customer segmentation model using behavioural, transactional, demographic, and RFM‑derived features. The aim is to identify meaningful customer groups that support targeted marketing, commercial strategy, and customer‑experience planning.

The workflow includes data preparation in Python, machine learning with scikit‑learn, and an interactive Tableau dashboard designed for both technical and non‑technical stakeholders.

---

## Tableau Dashboard
https://public.tableau.com/app/profile/sam.broad/viz/CustomerSegmentationUsingRFMAnalysis_17696135990530/Dashboard14

---

## Objectives
- Identify distinct customer groups using machine learning  
- Understand behavioural and demographic differences across segments  
- Support marketing, retention, and commercial decision‑making  
- Deliver a clear, stakeholder‑friendly dashboard  
- Demonstrate an end‑to‑end analytical workflow  

---
## Dataset

This project uses a synthetic dataset sourced from Kaggle, created for demonstration and learning purposes.

---

## Data and Features

The clustering model uses 12 behavioural and demographic features.

### Behavioural / Transactional
- total_spent  
- avg_order_value  
- loyalty_points  
- number_of_orders  
- support_tickets  
- recency  
- tenure  

### Demographic
- age  
- income  

### RFM‑Derived Scores
- R_score  
- F_score  
- M_score  

All features were scaled before modelling.

---

## Methodology

### 1. Data Preparation
- Cleaned and validated the dataset  
- Engineered behavioural and demographic features  
- Scaled numeric variables  
- Prepared data for clustering  

### 2. Machine Learning
- Applied K‑Means clustering to the 12‑feature dataset  
- Evaluated cluster separation and interpretability  
- Assigned each customer to a segment  
- Exported cluster labels for visualisation  

### 3. Visualisation
The Tableau dashboard includes a ternary plot, cluster summaries, key metrics, and interactive filtering.

---

## Cluster Profiles

### Cluster 0 — Low‑Frequency Regulars
Low spend, low frequency, low loyalty, mid‑range recency.  
Customers with modest engagement and value.

### Cluster 1 — VIP Loyal Customers
Highest spend, high frequency, very high loyalty, strong recency, long tenure.  
Most valuable and engaged customers.

### Cluster 2 — Dormant / At‑Risk Customers
Very old recency, very low frequency, low monetary value.  
Customers who are disengaged and at risk of churn.

### Cluster 3 — Occasional High‑Spenders
High total spend, very high average order value, low–moderate frequency, good recency.  
Infrequent but high‑value purchasers.

---

## Files in This Repository
- `customersegmentation_analysis.py` — Python script for data prep, feature engineering, scaling, and clustering  
- `customer_segments.csv` — Dataset or sample dataset  
- `README.md` — Project documentation  

---

## How to Reproduce

### 1. Install Python 3.x

### 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Add the dataset
Place the dataset in the project folder.

### 4. Run the script
python customersegmentation_analysis.py

### 5. Explore results
Open the Tableau dashboard to view the cluster profiles and interactive visualisation.



