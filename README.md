# House Price Prediction — Mumbai

## Problem
Predict residential property prices across Mumbai and deploy a live interactive prediction app.

## Dataset
~12,000 property listings sourced from 99acres.com and Housing.com via Kaggle. Features: location, carpet area (sqft), bedrooms, bathrooms, parking, amenities (gym, pool, club house), building type, possession status.

## What I Built
- Data cleaning: duplicate removal, outlier capping at 95th percentile, missing value handling
- Feature engineering: location one-hot encoding, area normalization, binary amenity features
- Three regression models trained and compared: Linear Regression, Random Forest Regressor, SVM
- Model evaluation using MAE, RMSE, and R²
- Deployed interactive **Streamlit app** — users input area, bedroom count, and location, and receive a live property price estimate in INR

## Results
| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 82,288 | 102,279 |

**Example Streamlit predictions:**
- 3,000 sqft · 2 bed · Mumbai Central → ₹5,58,17,587
- 2,500 sqft · 3 bed · Bandra West → ₹9,00,40,082

## Tech Stack
`Python` `pandas` `scikit-learn` `LinearRegression` `RandomForestRegressor` `SVM` `Matplotlib` `Seaborn` `Streamlit`

## Files
- `House_price_prediction_in_mumbai.py` — full model
- `Mumbai_House_Data.csv` — dataset
