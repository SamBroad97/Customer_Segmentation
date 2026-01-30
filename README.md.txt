# Customer Segmentation Using RFM and KMeans

This project performs customer segmentation on a synthetic retail dataset.  
The goal is to identify distinct customer groups based on purchasing behaviour, engagement, and value to the business.

The workflow includes:
- data cleaning and preparation  
- feature engineering (recency, tenure, AOV)  
- RFM scoring  
- scaling and clustering with KMeans  
- cluster profiling and persona creation  
- exporting the final dataset for visualisation in Tableau  

---

## 1. Dataset Overview

The dataset contains transaction‑level customer information, including:
- purchase dates  
- total spend  
- loyalty points  
- support interactions  
- demographic fields (age, income, subscription type)  

The data is aggregated to one row per customer for analysis.

---

## 2. Feature Engineering

Key engineered features:
- **recency**: days since last purchase  
- **tenure**: days since first purchase  
- **number_of_orders**: total purchases  
- **avg_order_value**  
- **R_score, F_score, M_score** (1–5)  
- **RFM_score** (combined metric)

These features help standardise behaviour before clustering.

---

## 3. Clustering Approach

KMeans clustering is applied to scaled numerical features.  
Silhouette scores were tested for k = 2 to 9.  
Although k = 2 produced the highest score, **k = 4** was selected to provide more interpretable customer groups.

---

## 4. Cluster Profiles & Personas

### **Cluster 0 — Low‑Frequency Regulars**
- Lower spend  
- Lower frequency  
- Lower loyalty  
**→ Occasional buyers with modest value**

### **Cluster 1 — VIP Loyal Customers**
- High spend  
- High frequency  
- High loyalty points  
- Strong long‑term engagement  
**→ Most valuable and engaged customers**

### **Cluster 2 — Dormant / At‑Risk Customers**
- Very old recency  
- Low frequency  
- Low monetary value  
**→ Customers at risk of churn**

### **Cluster 3 — Big Basket Buyers**
- High average order value  
- Moderate recency  
- Lower frequency  
**→ Infrequent but high‑spend customers**

---

## 5. Tableau Dashboard

The final dataset (`customer_segments.csv`) is used to create an interactive Tableau dashboard showing:
- cluster scatterplots  
- cluster size breakdown  
- heatmap of feature means  
- persona summaries  

**Tableau link:** *[Add your Tableau Public link here]*

---

## 6. Files in This Repository

- `customer_segmentation.py` — full analysis script  
- `customer_segments.csv` — final dataset for Tableau  
- `README.md` — project documentation  

---

## 7. Conclusion

This project demonstrates an end‑to‑end customer segmentation workflow using Python and KMeans.  
The resulting personas can support targeted marketing, retention strategies, and customer value analysis.
