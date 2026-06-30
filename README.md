# Customer Segmentation & Churn Prediction — RFM Analysis

🔗 **Live Demo:** [Try the app here](https://customer-segmentation-rfm-churn-vcbsf4j7p76d3z5uv3whea.streamlit.app/)

## Overview
An end-to-end customer intelligence project combining RFM (Recency, Frequency, Monetary) analysis, unsupervised customer segmentation, and churn prediction — built on the UCI Online Retail dataset. Deployed as a live, interactive Streamlit app for real-time churn risk scoring.

## Problem Statement
Rather than building a generic churn classifier, this project asks a more practical business question: **which customer segments matter most, and where should retention efforts be focused?** RFM segmentation identifies *who* customers are; churn prediction identifies *who's at risk*; combined, they tell a business *what to do about it*.

## Dataset
- **Source:** UCI Online Retail Dataset
- **Size:** ~540,000 transaction line-items, 4,372 unique customers (post-cleaning)
- **Cleaning:** Removed rows with missing CustomerID, filtered out cancelled/returned orders (negative quantity), removed customers with net $0 spend (full-order cancellations)

## RFM Feature Engineering
- **Recency** — days since each customer's last purchase (relative to the dataset's latest date)
- **Frequency** — count of unique invoices (orders) per customer
- **Monetary** — total amount spent (Quantity × UnitPrice, summed per customer)

## Customer Segmentation (KMeans)
- Standardized RFM features, used the **Elbow Method** to select K=5 clusters
- Each cluster profiled by mean Recency/Frequency/Monetary and labeled by business meaning:

| Cluster | Profile | Label |
|---------|---------|-------|
| 0 | High recency, low frequency, low spend | Lost / Inactive |
| 1 | Moderate recency, low frequency | Potential Loyalists |
| 3 | Low recency, high frequency, high spend | Loyal Customers |
| 4 | Very low recency, highest frequency | High-Value Customers |
| 2 | Near-zero recency, 213 avg orders, $67K avg spend | Champions |

**Investigation highlight:** Cluster 2's extreme order frequency initially suggested wholesale/reseller activity. Direct investigation of transaction-level quantities showed only 0.47% of their orders exceeded 100 units — ruling out bulk buying and confirming these are genuinely high-engagement individual customers, not business accounts.

## Churn Prediction
- Defined churn as Recency > 180 days (customers inactive for 6+ months)
- **Deliberately excluded Recency as a model feature** to prevent direct label leakage — model predicts churn using only Frequency, Monetary, and Cluster
- Built with `ColumnTransformer` + `Pipeline` (SimpleImputer, StandardScaler, OneHotEncoder, RandomForestClassifier with `class_weight='balanced'`)

### Results
| Metric | Not Churned (0) | Churned (1) |
|--------|------------------|--------------|
| Precision | 0.95 | 0.77 |
| Recall | 0.94 | 0.79 |
| F1-Score | 0.95 | 0.78 |

**91% overall accuracy, 79% recall on churned customers** — achieved without directly using Recency, suggesting historical engagement patterns alone carry meaningful churn signal.

## Business Recommendation
Marketing focus should prioritize **Cluster 1 (Potential Loyalists)** over already-lost Cluster 0 customers — intervening before disengagement fully sets in offers higher ROI than costly win-back campaigns after the fact.

## Deployment
Built and deployed a **live Streamlit web app** for real-time churn risk prediction:
- 🔗 **[Try it live here](https://customer-segmentation-rfm-churn-vcbsf4j7p76d3z5uv3whea.streamlit.app/)**
- Takes Frequency, Monetary, and Cluster as input
- Includes a cluster reference guide for non-technical users
- Returns color-coded churn risk (✅ Low Risk / ⚠️ High Risk)

Run locally instead:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Libraries Used
pandas, numpy, matplotlib, scikit-learn, streamlit, joblib

## Files
- `customer_segmentation_churn.ipynb` — full analysis, clustering, and model training
- `app.py` — Streamlit deployment app
- `churn_model.pkl` — trained pipeline (preprocessing + model)
- `requirements.txt` — dependencies for deployment
