# Ctrl+Price: Taking Control of Used Car Valuations with ML

**Reducing Mispricing to Accelerate Sales and Improve Profitability**

---

## Overview

This project aims to **optimize used car prices in Saudi Arabia** using advanced machine learning. By providing accurate, data-driven price recommendations, we help sellers on online marketplaces (like syarah.com) reduce pricing errors, speed up sales, and increase platform revenue.

---

## Business Problem

* **Stakeholder:** Online used-car marketplace (syarah.com)
* **Pain Points:**

  * Overpriced cars don’t sell (lost commission, stale inventory).
  * Underpriced cars lose seller & platform revenue.
  * Manual pricing is slow, error-prone, and inconsistent.
* **Goal:**
  Build a machine learning model to estimate the *fair market price* for any used car listing in Saudi Arabia, so sellers price correctly on day one.

---

## Solution

* **Machine Learning Model:** CatBoost Regressor (tree-based, handles raw categoricals, robust to outliers)
* **Features Used:** Make, Model, Year, Car Type, Options, Engine Size, Region, and more
* **Key Feature Engineering:**

  * Car Age (2025 - Year)
  * Outlier handling with business logic (vintage flag, engine size cap, mileage/year sanity checks)
* **Core Deliverables:**

  * ML-powered price prediction API/app (Streamlit demo)
  * Business analytics on mispricing, potential revenue gains, and actionable recommendations

---

## Data

* **Source:** Real used car listing data from Saudi Arabia
* **Shape:** \~30,000 rows, 13 features (see Data Dictionary)
* **Cleaning/EDA:**

  * Remove “Negotiable”/0-price listings
  * Light outlier removal (documented business logic)
  * Preserve genuine market diversity; keep rare luxury/vintage/unique cases (documented strategy)

---

## Modeling Approach

* **Benchmarking:**

  * Tested KNN, Decision Tree, Random Forest, XGBoost, LightGBM
  * Chose CatBoost due to best MAE, categorical support, and business interpretability
* **Cross-validation:** 5-fold CV, stratified by price segment
* **Main Metric:**

  * Mean Absolute Error (MAE) in SAR (target ≈ 2,800 SAR, \~5% of average price)
  * Median Absolute Percentage Error (MdAPE) as companion

---

## Business Impact

* Over 50% of listings are mispriced by ±10% vs. ML fair value
* Implementing the model could recover **>SAR 500,000** in lost commission (from just the test set)
* **Actionable:**

  * Instantly flag under/overpriced listings for sellers & moderators
  * Provide dynamic price recommendations at listing time

---

## Deployment

* **App Demo:** Streamlit interface for instant price prediction & batch CSV uploads
* **Interpretability:** SHAP values & plots to explain individual and global predictions to users & business

---

## Results

* **Best Model:** CatBoost Regressor (25-180k SAR)

  * **Test MAE:** \~14,000 SAR
  * **MAPE:** \~15.72%
* **Robustness:**

  * Validated on holdout set, stable across price segments
  * Interpretability via SHAP confirms model uses business-relevant features (Make, Type, Age, Options)

---

## 🧑‍💻 Authors & Credits

* **Lead:** [Rio Pramana](https://github.com/yourusername)
* **Mentors:** Purwadhika DTI ML Bootcamp

---

## Disclaimer

This project is for educational use. All code is reproducible.
