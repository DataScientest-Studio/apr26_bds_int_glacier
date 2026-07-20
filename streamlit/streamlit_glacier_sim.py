#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:57:48 2026

@author: theresa
"""

import pandas as pd
import numpy as np
import os
import streamlit as st
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# ==========================================
# DEFINITION DES BILDER-GRIDS (Ganz oben oder bei den Funktionen einfügen)
# ==========================================
def zeige_gletscher_bildergalerie():
    import os
    
    # 1. Pfade clever ermitteln
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Wo liegt dieses Skript?
    BASIS_ORDNER = os.getcwd()                               # Wo wurde das Terminal gestartet?
    
    # Strategie A: Versuche es relativ zum Skript-Standort (Sehr zuverlässig)
    img_dir = os.path.join(SCRIPT_DIR, "plots")
    
    # Strategie B: Falls das nicht klappt, versuche den langen Pfad ab Terminal-Start
    if not os.path.exists(img_dir):
        img_dir = os.path.join(BASIS_ORDNER, "apr26_bds_int_glacier-main", "streamlit", "plots")
        
    # Strategie C: Fallback, falls direkt im streamlit-Ordner gestartet
    if not os.path.exists(img_dir):
        img_dir = os.path.join(BASIS_ORDNER, "plots")
    
    # --- DIAGNOSE-BOX (Kannst du löschen, wenn es läuft!) ---
    # st.info(f"Der Code sucht deine Bilder aktuell in: `{img_dir}`")
    # --------------------------------------------------------

    image_filenames = [
    #    "WL_FE_plain.png",  
        "WL_FE_plain_detail.png",               
    #    "WL_FE_saisona_y_d.png",
        "WL_FE_saisona_y_d_detail.png",
    #    "WL_FE_saisona_y_d_snowmem.png",
        "WL_FE_saisona_y_d_snowmem_detail.png",
    #    "WL_FE_saisona_y_d_snowmem_Lag.png",
        "WL_FE_saisona_y_d_snowmem_Lag_detail.png"
    ]
    
    COLS_PER_ROW = 1
    ROWS = 4
    
    # 2. Grid aufbauen
    for row_idx in range(ROWS):
        cols = st.columns(COLS_PER_ROW)
        for col_idx in range(COLS_PER_ROW):
            image_index = row_idx * COLS_PER_ROW + col_idx
            
            if image_index < len(image_filenames):
                img_name = image_filenames[image_index]
                img_path = os.path.join(img_dir, img_name)
                
                with cols[col_idx]:
                    with st.container(border=True):
                        display_title = os.path.splitext(img_name)[0].replace("_", " ")
                        
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.warning(f"Bild nicht gefunden: {img_name}")


st.title('Modelling the Water Level')
st.header('of a Glacier Stream using Meteorological Data')
st.sidebar.title("Table of contents")
pages = ['Project Description', 
         'Data Visualization',
         'Data Preprocessing',
         'Data Analysis',
         'Feature Engineering',
         'Modeling',
         'Conclusion']
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.header("Project Description")
    st.image('csm_8_Vernagtxxxx_6853154684.jpg', caption='Vernagtferner in Tyrol (2900 - 3600 m)')
    st.subheader('Goal')
    st.write('To set up a model to compute the water level of the stream Vernagtbach from the meteorological variables')
    st.markdown('- air temperature')
    st.markdown('- relative humidity')
    st.markdown('- wind speed and direction')
    st.markdown('- precipitation and snow height')
    st.markdown('- short wave radiation downwards and upwards')
    st.write('measured at two measurement stations at 2640 m and 3080 m in the vicinity of the glacier from January 1, 2013 to December 31, 2024 in steps of 5 minutes.')
    st.subheader('Issues')
    st.write('The problem of the project is not a conventional data science problem:')
    st.markdown('- It is not a typical regression problem because the time dependence spoils the independence between the observations.')
    st.markdown('- It is not a typical time series problem because it is not the prediction of the water level from the previous water levels.')

if page == pages[1]: 
    st.header("Data Visualization of some variables")
    
    st.subheader('Water Level')
    st.image('plots/overview_water_level.png')
    
    st.subheader('Air Temperatures')
    st.image('plots/overview_temperature.png')
    
    st.subheader('Snow Height')
    st.image('plots/overview_snow.png')
    
    st.subheader('Wind Direction')
    st.image('plots/overview_wind_direction_2days.png')
    
if page == pages[2]:
    st.header("Data Preprocessing")
    st.subheader('Preprocessing of the target variable water_level')
    st.markdown('- water_level is set to 5 cm in winter')
    st.markdown('- other large time gaps of water_level are filled with adjusted values of water_level_ultrasound')
    st.markdown('- remaining small gaps are interpolated')
    st.subheader('Preprocessing of the other variables')
    st.markdown('- Forward Filling for longer gaps (larger than ca.6 hours) of the variables')
    st.markdown('- Forward Shifting for gaps of less than ca. 6 hours')
    st.image('plots/Temp_cutout_mod_exp.png')
    st.subheader('Measurement variables from PM station after preprocessing')
    st.image('plots/Variab_PM_pre_processed.png')
    
if page == pages[3]:
    st.header("Data Analysis")
    st.image('Corr_matrix.png', width='stretch')
    st.subheader('Cross Correlation Analysis (Covariance with time lag)')
    st.image('plots/time_lag_wat_lev_temp.png')
    st.image('plots/time_lag_wat_lev_SWD.png')
    st.subheader('Seasonality Analysis')
    st.markdown('- Water Level shows a strong seasonality with a maximum in summer and a minimum in winter')
    st.image('plots/deco_season_WL.png', width='stretch')
    st.markdown('- Temperature ')
    st.image('plots/deco_season_PM_temp.png', width='stretch')


if page == pages[4]:
    st.header("Feature Engineering")
    st.subheader("Feature Engineering for Machine Learning Models")
    st.markdown('- Data is cummulated to 1 hour steps')
    st.markdown('- Creation Lag Features for the variables with high covariance with the target variable water_level'
    ' and a significant time shift in cross correlation, see Data Analysis')
    st.image('plots/Lag3_Temp.png', width='stretch')
    st.markdown('- Generation of synthetic features for yearly and daily saesonality')
    st.image('plots/syn_seasonality.png')
    st.image('plots/syn_day.png')
    st.markdown('- Generation of feature to sensing snow melt in summer')
    st.image('plots/snow_melting.png') 
    st.markdown('- List of features used for the Machine Learning Models')

    st.subheader('Feature Engineering for Deep Learning Models')
    st.markdown('- Data is left in steps of 5 minutes')
    st.markdown('- Creation of a new time variable to keep through the modell as time reference')
    st.markdown('- Conversion of the wind variables of speed and direction in speed in X direction and speed in Y direction')
    st.markdown('- Conversion of the cumulative precipitation in variables precipitation and evaporation')
    st.markdown('- Deletion of variables because of faulty and redundant measurements')

if page == pages[5]:
    st.header("Modeling")
    st.subheader('Model Application of Machine Learning Models')
    st.subheader('Linear Regression')

    # Hier wird die Galerie jetzt genau EINMAL gezielt gezündet:
    zeige_gletscher_bildergalerie()



    st.subheader('Linear Regression')
    # use_container_width sorgt dafür, dass das Bild die volle Breite ausfüllt
    st.image('plots/Metric_LinReg.png', use_container_width=True)

    st.subheader('GammaRegressor with Feature Engineering for seasonality, snow memory and lag features')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('Results ')
        st.image('plots/Gamma_WL_FE_saisona_y_d_snowmem_Lag_detail.png', use_container_width=True)

    with col2:
        st.markdown(' Q-Q plot of residuals')
        st.image('plots/Q-Q_Ti_comp.png', use_container_width=True)
        
    st.image('plots/Metric_GammaReg.png')

    st.subheader('Lazypredictor for classification best ML model')
    st.image('plots/LaysyClas.png', width='stretch')

    st.subheader('Best ML model for regression: HistGradientBoostingRegressor / Best tradeoff between accuracy and computation time')
    st.image('plots/HGBR_WL_FE_saison_snowmem_lag_Ti_comp.png', width='stretch')
    st.image('plots/HGBR_WL_FE_saison_snowmem_lag_Ti_comp_detail.png', width='stretch')
    st.image('plots/Metric_HbGBR.png')

    st.set_page_config(
    page_title="Glacier Runoff Simulation Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    # Note: Using standard triple-quotes (NOT f-strings) to avoid curly brace parsing conflicts.
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f4f8;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #005088;
            margin-bottom: 10px;
        }
        .stButton>button {
            background-color: #005088 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #11caa0 !important;
            transform: scale(1.02);
        }
        .highlight-box {
            background-color: #e8f4fd;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #b3d7ff;
        }
    </style>
    """, unsafe_allow_html=True)

    def find_csv_file(filename="mod_feature_meteohydrodata_hourly.csv"):
        """
        Recursively searches for the CSV file in current and parent directories
        to avoid path issues across different machine setups.
        """
        search_paths = [
            filename,
            os.path.join("..", filename),
            os.path.join("ML_Model", filename),
            os.path.join("..", "ML_Model", filename),
            os.path.join("C:\\Users\\HWHah\\OneDrive\\Desktop\\Python_Prog\\DS_Glacier", filename),
            os.path.join("C:\\Users\\HWHah\\OneDrive\\Desktop\\Python_Prog\\DS_Glacier\\ML_Model", filename)
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
                
        # Fallback recursive walk
        for root, dirs, files in os.walk("."):
            if filename in files:
                return os.path.join(root, filename)
                
        return None

    @st.cache_data
    def load_and_prepare_data():
        csv_path = find_csv_file()
        if not csv_path:
            st.error("❌ The file 'mod_feature_meteohydrodata_hourly.csv' could not be found! Please place it in your project folder.")
            st.stop()
            
        df = pd.read_csv(csv_path)
        
        # Locate date/timestamp column
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break
                
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
            
        # CRITICAL: Strip timezone to prevent comparison errors with naive datetime strings!
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Sort index for chronological split processing
        df.sort_index(inplace=True)
        
        # Fill gaps (essential for lag calculation robustness)
        df = df.ffill().bfill()
        
        return df

    # Load data
    try:
        df_raw = load_and_prepare_data()
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        st.stop()

    st.sidebar.header("📂 1. Data & Variables")

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Pre-select target variable logically
    default_target_idx = 0
    for i, col in enumerate(numeric_cols):
        if any(keyword in col.lower() for keyword in ["water_level", "pegel", "abfluss", "q_sim", "runoff", "discharge"]):
            default_target_idx = i
            break

    target_var = st.sidebar.selectbox(
        "Target Variable (Y):",
        options=numeric_cols,
        index=default_target_idx
    )

    # Automatically pre-select all other numeric columns as inputs (X)
    default_features = [col for col in numeric_cols if col != target_var]
    selected_features = st.sidebar.multiselect(
        "Predictive Features (X):",
        options=numeric_cols,
        default=default_features,
        help="Select the environmental variables your model will use as inputs."
    )

    if not selected_features:
        st.sidebar.warning("⚠️ Please select at least one feature!")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("📅 2. Time Periods (Splits)")

    min_data_date = df_raw.index.min().to_pydatetime()
    max_data_date = df_raw.index.max().to_pydatetime()

    # Configure the exact default ranges from your script
    default_train_start = pd.to_datetime("2013-01-01").to_pydatetime()
    default_train_end = pd.to_datetime("2022-12-31").to_pydatetime()
    default_test_start = pd.to_datetime("2023-01-01").to_pydatetime()
    default_test_end = pd.to_datetime("2024-06-30").to_pydatetime()

    train_start = st.sidebar.date_input(
        "Training Period Start",
        value=max(min_data_date, default_train_start),
        min_value=min_data_date,
        max_value=max_data_date
    )

    train_end = st.sidebar.date_input(
        "Training Period End",
        value=min(max_data_date, default_train_end),
        min_value=min_data_date,
        max_value=max_data_date
    )

    test_start = st.sidebar.date_input(
        "Testing Period/Simulation Start",
        value=max(min_data_date, default_test_start),
        min_value=min_data_date,
        max_value=max_data_date
    )

    test_end = st.sidebar.date_input(
        "Testing Period/Simulation End",
        value=min(max_data_date, default_test_end),
        min_value=min_data_date,
        max_value=max_data_date
    )

    # Validation check for bounds
    if train_start >= train_end:
        st.error("Error: Training Start must be before Training End!")
        st.stop()
    if test_start >= test_end:
        st.error("Error: Testing Start must be before Testing End!")
        st.stop()

    @st.cache_resource
    def train_custom_model(target, features_tuple, t_start, t_end):
        """
        Trains the regressor live based on sidebar choices.
        Uses a features tuple to allow Streamlit caching to work.
        """
        features_list = list(features_tuple)
        
        # Filter dataset for training range
        train_mask = (df_raw.index >= pd.to_datetime(t_start)) & (df_raw.index <= pd.to_datetime(t_end))
        df_train = df_raw[train_mask]
        
        X_train = df_train[features_list]
        y_train = df_train[target]
        
        # Train HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions on training subset for training metrics
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        return model, train_rmse, train_r2, len(df_train)

    # Start live training process
    features_tuple = tuple(selected_features)
    model, train_rmse, train_r2, train_rows = train_custom_model(
        target_var, 
        features_tuple, 
        train_start, 
        train_end
    )

    # Prepare hold-out validation dataset
    test_mask = (df_raw.index >= pd.to_datetime(test_start)) & (df_raw.index <= pd.to_datetime(test_end))
    df_test = df_raw[test_mask].copy()

    X_test = df_test[selected_features]
    y_test = df_test[target_var]

    # Compute true test-set performance metrics
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    st.title("🌊 Glacier Runoff: Live Model Simulation Pro")

    st.markdown(f"""
    This interactive application runs a **live model execution** of your trained *Histogram-based Gradient Boosting Regressor*. 
    You can dynamically configure variables, target objectives, and date splits in the left sidebar to match your offline runs and inspect predictions in real-time.
    """)

    st.subheader("📊 Model Performance Comparison")

    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Datapoints (N)", f"{train_rows:,} rows")
        st.metric("Model Train $R^2$", f"{train_r2:.4f}")
        st.metric("Model Train RMSE", f"{train_rmse:.4f} cm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Testing Datapoints (N)", f"{len(df_test):,} rows")
        st.metric("Model Test $R^2$", f"{test_r2:.4f}")
        st.metric("Model Test RMSE", f"{test_rmse:.4f} cm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("##### 💡 Understanding Different Metric Values")
        st.markdown(f"""
        * **Training Window:** The offline script covers **10 years** (2013–2022, ~87,638 lines). To perfectly match your offline output, set the sidebar training dates to this 10-year range to obtain your exact test R² of **{test_r2:.4f}** and RMSE.
        * **Features Selection:** Ensure you selected exactly the same features (X-variables) in the sidebar as you did in your offline python script.
        * **Overfitting Effect:** Restricting training to a smaller subset (e.g. starting from 2019) will falsely yield an extremely high training R² (~0.98), as the model memorizes fewer points, but it will suffer from reduced generalization capability on unseen test phases.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ Interactive Scenario Simulation")

    st.sidebar.markdown("---")
    st.sidebar.header("🎮 3. Simulation Control")

    # Select a simulation start point within test boundaries
    sim_start_point = st.sidebar.date_input(
        "Simulation Start Date",
        value=pd.to_datetime("2023-07-15"),
        min_value=test_start,
        max_value=test_end - pd.Timedelta(days=5)
    )

    sim_days = st.sidebar.slider("Simulation Duration (Days)", min_value=1, max_value=7, value=3)
    sim_speed = st.sidebar.slider("Speed (Seconds per Step)", min_value=0.01, max_value=0.50, value=0.05)

    # Climate Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.header("🌡️ Climate Scenario")
    temp_shift = st.sidebar.slider("Temperature Anomaly (°C)", min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
                                help="Simulates global warming or regional cooling periods during this simulation run.")

    sim_start_dt = pd.to_datetime(sim_start_point)
    sim_end_dt = sim_start_dt + pd.Timedelta(days=sim_days)
    sim_slice = df_test.loc[sim_start_dt:sim_end_dt].copy()

    # Apply temperature shifted scenario if a temperature feature exists
    temp_cols = [col for col in selected_features if "temp" in col.lower() or "temperature" in col.lower()]
    if temp_cols and temp_shift != 0:
        for tc in temp_cols:
            sim_slice[tc] = sim_slice[tc] + temp_shift

    # Trigger Button
    start_button = st.sidebar.button("▶️ Start Live Simulation", use_container_width=True)

    # Placeholder objects for real-time reporting metrics
    m1, m2, m3, m4 = st.columns(4)
    metric_temp = m1.metric("Simulated Temp", "---")
    metric_actual = m2.metric("Actual Runoff", "---")
    metric_pred = m3.metric("Model Prediction", "---")
    metric_rmse = m4.metric("Running RMSE", "---")

    chart_placeholder = st.empty()

    if start_button:
        simulated_times = []
        actual_values = []
        predicted_values = []
        
        progress_bar = st.sidebar.progress(0)
        total_steps = len(sim_slice)
        
        if total_steps == 0:
            st.warning("No data points found inside the selected simulation date range. Please select another date.")
        else:
            for idx, (timestamp, row) in enumerate(sim_slice.iterrows()):
                # Prepare feature input frame for this timestep
                features_single = pd.DataFrame([row[selected_features].to_dict()])
                
                # Predict output using loaded regressor
                prediction = model.predict(features_single)[0]
                
                # Aggregate timeseries arrays
                simulated_times.append(timestamp)
                actual_values.append(row[target_var])
                predicted_values.append(prediction)
                
                # Calculate running performance metric (RMSE)
                if len(actual_values) > 1:
                    current_rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                    rmse_display = f"{current_rmse:.2f} cm"
                else:
                    rmse_display = "---"
                    
                # Read first detected temperature column to update metric card
                temp_display = f"{row[temp_cols[0]]:.1f} °C" if temp_cols else "N/A"
                
                # Refresh metric displays
                metric_temp.metric("Simulated Temp", temp_display)
                metric_actual.metric("Actual Runoff", f"{row[target_var]:.1f} cm")
                metric_pred.metric("Model Prediction", f"{prediction:.1f} cm")
                metric_rmse.metric("Running RMSE", rmse_display)
                
                # Re-draw the interactive plot dynamically
                fig = go.Figure()
                
                # Observed historical telemetry curve
                fig.add_trace(go.Scatter(
                    x=simulated_times, 
                    y=actual_values,
                    mode='lines',
                    name='Measured Runoff (Actual)',
                    line=dict(color='#005088', width=2)
                ))
                
                # Simulated model projection curve
                fig.add_trace(go.Scatter(
                    x=simulated_times, 
                    y=predicted_values,
                    mode='lines+markers',
                    name='Model Simulation (Predicted)',
                    line=dict(color='#11caa0', width=3, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Real-Time Stream Simulation ({timestamp.strftime('%d.%m.%Y %H:%M')})",
                    xaxis_title="Timeline",
                    yaxis_title=f"{target_var} (cm)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=450,
                    xaxis=dict(range=[sim_start_dt, sim_end_dt]),
                    yaxis=dict(range=[min(df_test[target_var]) - 5, max(df_test[target_var]) + 10])
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Increment progress bar
                progress_bar.progress((idx + 1) / total_steps)
                
                # Rest time to fluidly render steps
                time.sleep(sim_speed)
                
            # Compile post-sim reports
            final_sim_rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            final_sim_r2 = r2_score(actual_values, predicted_values)
            st.sidebar.success("✅ Simulation completed successfully!")
            
            st.markdown("---")
            st.subheader("🎯 Simulation Summary Report")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Simulation RMSE", f"{final_sim_rmse:.3f} cm")
            sc2.metric("Simulation R²", f"{final_sim_r2:.3f}")
            sc3.info(f"Scenario modification: **{temp_shift:+.1f} °C** applied to temperature signals.")

    else:
        # Default standby state before user clicks "Start Simulation"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_slice.index, 
            y=sim_slice[target_var],
            mode='lines',
            name='Observed Measurements (Historical)',
            line=dict(color='#6d6d6b', width=1.5)
        ))
        fig.update_layout(
            title="Ready for Simulation — Click 'Start Live Simulation' in the left sidebar to begin",
            xaxis_title="Timeline",
            yaxis_title=f"{target_var} (cm)",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    st.subheader('Model Application of Deep Learning Models')
    st.markdown('- To feed data to a neural net a CustomDataset was defined in PyTorch.')   
    st.subheader('Model Application of Deep Learning Models')
    st.markdown('- To feed data to a neural net a CustomDataset was defined in Pytorch:')
    st.markdown('- It divides the input data into continuous chunks of data on the time axis with a given time span.')
    st.markdown('- The chunks overlap each other.')
    st.markdown('- A model is trained with these chunks.')
    st.markdown('- For the metric and the output of the model an output of the model is computed by averaging the output chunks of the model on their position of the time axis.')
    st.markdown('- The chosen metric for validation and testing is mean absolute error.')
    st.markdown('- Training Dataset: 70\%, Validation Dataset: 15\%, Test Dataset: 15\%')
    st.markdown('- Architecture of first model:')

    st.write('Sequential(')
    st.write('(0): Conv2d(1, 2, kernel_size=(3, 1), stride=(1, 1))')
    st.write('(1): ReLU()')
    st.write('(2): MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=0, dilation=1, ceil_mode=False)')
    st.write('(3): Conv2d(2, 4, kernel_size=(7, 1), stride=(1, 1))')
    st.write('(4): ReLU()')
    st.write('(5): MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=0, dilation=1, ceil_mode=False)')
    st.write('(6): Flatten(start_dim=1, end_dim=-1)')
    st.write('(7): Linear(in_features=768, out_features=288, bias=True)')

    st.markdown('- Graphical presentation of the result of the first model:')
    st.image('model_PM_0_result.png', width='stretch')

    st.markdown('- Results for 4 different models:')
    results = pd.DataFrame(
        {
        "model_PM_0": [8.73, 10.74, '1 day'],
        "model_all_0": [6.7, 9.2, '1 day'],
        "model_pm_1": [5.87, 8.41, '7 days'],
        "model_all_1": [6.49, 9.38, '7 days'],
        },
        index=["mean absolute error", "rmse", "time span"])
    st.table(results)

if page == pages[6]:
    st.header("Conclusion and Outlook")
    st.subheader('Conclusion for Machine Learning Models')
    st.image('plots/Concl_metrics.png', width='stretch')
    
    st.subheader('Conclusion on Deep Learning Models')
    st.markdown('- It was possible to get a Deep Learning Model running.')
    st.markdown('- The results are only as good a linear regression.')
    st.markdown('- There is very likely improvement on the data loading part possible leading to much lower computation times.')
    st.markdown('- There are a lot of model architectures to be tried out.')
    st.subheader('Outlook')
    st.markdown('- Gain more experience on neural networks and their set up')
    st.markdown('- Try again')
        