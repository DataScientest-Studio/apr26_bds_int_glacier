import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv('mod_feature_meteohydrodata2026-06-19.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')

#normalization
df_norm = (df - df.min()) / (df.max() - df.min())
#df_norm = df

#Data split
# sorting acc. to time and remove time zone
df_norm = df_norm.sort_index()
df_norm.index = df_norm.index.tz_localize(None)

#Split acc. to fix dates
train_start_date = '2020-01-01'
train_end_date = '2023-10-31'
test_end_date = '2024-10-30'
test_end_dt = pd.to_datetime(test_end_date)

test_start_dt = pd.to_datetime(train_end_date) + pd.Timedelta(days=1)
test_start_date = test_start_dt.strftime('%Y-%m-%d')

#Define train and test data
train_df = df_norm[train_start_date:train_end_date]
test_df = df_norm[test_start_date:test_end_date]

print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} Zeilen)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} Zeilen)")

#Target and discibtive variables
target_col = "water_level"

# y is target
y_test = test_df[target_col]
y_train = train_df[target_col]

# X are explanatory values
X_test = test_df.drop(columns=[target_col])
X_train = train_df.drop(columns=[target_col])

print(f"Training set: {train_df.index.min()} bis {train_df.index.max()} ({len(train_df)} Zeilen)")
print(f"Test set:     {test_df.index.min()} bis {test_df.index.max()} ({len(test_df)} Zeilen)")


# model impelentation

y_train = y_train.asfreq('D')
model = sm.tsa.SARIMAX(
    y_train,
    exog=X_train[['summer_seasonality','lag_1']],
    order=(1,1,1)
)
sarima=model.fit()
print(sarima.summary())

start_date = y_test.index[0]
end_date = y_test.index[-1]

# model run

pred = sarima.predict(
    start=start_date, 
    end=end_date, 
    exog=X_test[['summer_seasonality','lag_1']]  
)

rmse = root_mean_squared_error(y_test, pred)
print(f'Erreur quadratique moyenne (RMSE) : {rmse}')


# Visualisation

water_level = pd.concat([y_train, pred])

plt.figure(figsize=(12, 6))

# 1. plot train data  (z.B. in Blau)
plt.plot(y_train.index, y_train, 
         label='Historical Data (Train)', color='#1f77b4', linewidth=2)

# 2. plot prediction (z.B. in Orange)
plt.plot(pred.index, pred, 
         label='Model Prediction (Forecast)', color='#ff7f0e', linestyle='--', linewidth=2)

plt.plot(y_test.index, y_test, 
         label='Actual Data (y_test)', color='#2ca02c', linewidth=1, alpha=0.5)

# 3. plot indicator
plt.axvline(x=pd.Timestamp(train_end_date), 
            color='red', linestyle=':', linewidth=1.5, label='Start of Prediction')

# 4. Diagram-Details 
plt.title('Water Level Forecast with Distinct Data Phases')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()

