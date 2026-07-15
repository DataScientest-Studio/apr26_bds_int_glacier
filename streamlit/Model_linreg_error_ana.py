import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv(r"C:\Users\HWHah\OneDrive\Desktop\Python_Prog\DS_Glacier\ML_Model\mod_feature_meteohydrodata_hourly.csv", index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

# define reduced data set
#df.drop(columns=['PM_relative_humidity','PM_precipitation_cum','PM_wind_direction'], inplace=True)
#df = df[["water_level",'summer_seasonality','lag_1']]
#normalization for modeling not applicable : data leakage!!!!!!
#df_norm = (df - df.min()) / (df.max() - df.min())

#Data split
# sorting acc. to time and remove time zone
df = df.sort_index()
df.index = df.index.tz_localize(None)

# Split acc. to fixed date / time series
train_start_date = '2019-12-31'
train_end_date = '2023-04-30'

test_end_date = '2023-10-30'
test_start_dt = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
test_start_date = test_start_dt.strftime("%Y-%m-%d %H:%M:%S")
train_df = df[:train_end_date]
#train_df = df_norm[train_start_date:train_end_date]  
test_df = df[test_start_date:test_end_date]   

target_col = "water_level"

# y is target
y_test = test_df[target_col]
y_train = train_df[target_col]

# X explanatory variables
X_test = test_df.drop(columns=[target_col])
X_train = train_df.drop(columns=[target_col])


print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} rows)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} rows)")

# normalisation of the data (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# model impelentation
model = LinearRegression()
#y_train_log = np.log1p(y_train) # captures the exponential growth pattern in the data
model.fit(X_train, y_train)  # Fit the model on the training data


# model run
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
#y_test_pred_log = model.predict(X_test)
#y_test_pred = np.expm1(y_test_pred_log)
#y_train_pred_log = model.predict(X_train)
#y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.clip(y_test_pred, a_min=0.0, a_max=None)

# Performance evaluation
rmse_test = root_mean_squared_error(y_test, y_test_pred)
rmse_train = root_mean_squared_error(y_train, y_train_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Error (RMSE) : {rmse_test}')
print(f'Error (RMSE) : {rmse_train}')
print(f'R² (Test) : {r2_test}')
print(f'R² (Train) : {r2_train}')

#Analysis

# generate series type
y_pred_series = pd.Series(y_test_pred, index=y_test.index)

plt.figure(figsize=(12, 6))

# 1. Train data (z.B. in Blau)
plt.plot(y_train.index, y_train, 
         label='Historical Data (Train)', color='#1f77b4', linewidth=2)

# 2. Prediction data (z.B. in Orange)
plt.plot(y_pred_series.index, y_pred_series, 
         label=f'Model Prediction (Forecast)\n (RMSE : {rmse_test:.4f},R² = {r2_test:.4f})', color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.8)

plt.plot(y_test.index, y_test, 
         label='Actual Data (y_test)', color='#1f77b4', linewidth=2, alpha=0.5)

# 3. Diagram-Details 
plt.title('Water Level Forecast with Distinct Data Phases')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()

#Error analysis
from sklearn.metrics import root_mean_squared_error

# 1. Create analysis DataFrame
error_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
}, index=y_test.index).sort_index()

# Calculate local deviation (Negative = Model underestimates the peak)
error_df['Deviation'] = error_df['Predicted'] - error_df['Actual']

# 2. Get the top 5 largest underestimations
top_5_errors = error_df.sort_values(by='Deviation').head(5)

# 3. Initialize plot
plt.figure(figsize=(15, 7))

# Plot time series for actual and predicted values
plt.plot(error_df.index, error_df['Actual'], label='Actual (Measured)', color='blue', alpha=0.6, linewidth=1.5)
plt.plot(error_df.index, error_df['Predicted'], label='Predicted (Model)', color='orange', alpha=0.8, linewidth=1.5)

# 4. Highlight and annotate the top 5 peak errors
for idx, row in top_5_errors.iterrows():
    # Red dot on the actual peak
    plt.scatter(idx, row['Actual'], color='red', s=60, zorder=5, edgecolors='black')
    
    # Two-line label: Actual value and the local error deviation
    label_text = f"Act: {row['Actual']:.1f}\nΔ: {row['Deviation']:.1f}"
    
    # Position the text box next to the dot
    plt.annotate(
        label_text, 
        xy=(idx, row['Actual']), 
        xytext=(10, 10),              # 10 pixels offset to the top-right
        textcoords='offset points', 
        fontsize=8, 
        fontweight='bold',
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8) # White background box
    )

# 5. Chart layout and formatting (English)
#gesamt_rmse = root_mean_squared_error(y_test, y_pred)

plt.title(f"Glacier Modeling: Visualization of Maximum Peak Underestimations\n(RMSE : {rmse_test:.4f},R² = {r2_test:.4f})", fontsize=12, fontweight='bold')
plt.xlabel("Date / Time")
plt.ylabel("Value")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()

