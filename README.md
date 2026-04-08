# FareFocus-Boston: Dynamic Rideshare Pricing Prediction

FareFocus is a high-performance predictive analytics project that estimates Uber and Lyft fares in Boston, MA. By leveraging a dataset of over **630,000 rides**, this project demonstrates how transitioning from linear statistical baselines to advanced ensemble gradient boosting can slash prediction error by **51%**.

## 📊 Numerical Achievements & Results

The primary goal was to minimize the variance between predicted and actual fares. The transition from a linear baseline (Ridge Regression) to a boosted architecture (XGBoost) yielded the following results:

### Key Success Metrics
* **50.8% Reduction in RMSE:** The standard deviation of prediction errors dropped from **$6.33** to **$3.11**.
* **34.2% Improvement in MAE:** The average absolute prediction error was reduced from **$1.78** to **$1.17**.
* **High Precision:** Achieved an **R-Squared of 0.964**, meaning the model explains 96.4% of the variance in pricing.

### Model Performance Comparison
| Model | RMSE (Error) | MAE (Avg Dev) | R² (Accuracy) |
| :--- | :--- | :--- | :--- |
| **Baseline (Ridge Regression)** | $6.33 | $1.78 | 0.927 |
| **XGBoost Regressor** | **$3.11** | **$1.17** | **0.964** |
| **Stacking Ensemble** | $3.12 | $1.17 | 0.964 |

> **Note:** While the Stacking Ensemble (Random Forest + XGBoost) provided a robust architecture, the optimized XGBoost model alone achieved the most significant leap in performance relative to computational cost.

## 🧠 Technical Architecture

The project utilizes a multi-stage machine learning pipeline:

1.  **Data Cleaning:** Filtered over 55,000 entries, removing standard Taxi rides and null price values to ensure model integrity.
2.  **Preprocessing:** * **Numeric:** `StandardScaler` for distance and surge multipliers.
    * **Categorical:** `OneHotEncoder` for cab types, specific service names (e.g., "Lux Black"), and temporal features like the hour of the day.
3.  **Advanced Modeling:** Implemented an **XGBoost architecture** with a learning rate of 0.1 and a max depth of 6 to capture non-linear pricing triggers like surge multipliers and weather conditions.



## 🛠️ Tech Stack
* **Core:** Python
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Data Source:** [Uber and Lyft Dataset (Boston, MA)](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)

## 🚀How to Run in your System

### 1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/FareFocus-Boston.git](https://github.com/Akshat-Singh-Kshatriya/FareFocus-Boston.git)
   cd FareFocus-Boston
   ```
### 2. Install Dependies 
 ```bash
   pip install -r requirements.txt
   ```
### 3. Run the Notebook
```bash
   jupyter notebook uber_data_analytics.ipynb
```
 
