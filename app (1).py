import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
import plotly.graph_objects as go

# --- Custom Theme ---
st.set_page_config(
    page_title="Restful Nights: Sleep Analyzer",
    page_icon="🌙",
    layout="centered"
)

# --- Utility Functions ---

@st.cache_resource
def load_models():
    models = {}
    try:
        models["Classic Model"] = load("Xgb_model.joblib")
    except Exception:
        models["Classic Model (Demo)"] = None
    try:
        models["Optimized Model"] = load("XGB_tunned_model.joblib")
    except Exception:
        models["Optimized Model (Demo)"] = None
    return models

def create_features(df):
    df = df.copy()
    df['series_id'] = 1
    for col, win, func in [
        ('enmo_std_1m', 12, 'std'),
        ('anglez_std_1m', 12, 'std'),
        ('enmo_mean_2m', 24, 'mean'),
        ('anglez_mean_2m', 24, 'mean')
    ]:
        base = col.split('_')[0]
        if func == 'std':
            df[col] = df.groupby('series_id')[base].rolling(win).std().reset_index(level=0, drop=True)
        else:
            df[col] = df.groupby('series_id')[base].rolling(win).mean().reset_index(level=0, drop=True)
        df[col].fillna(df[base], inplace=True)
    # Clustering
    try:
        X = df[['anglez', 'enmo']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, n_init=10)
        df['activity_cluster'] = kmeans.fit_predict(X_scaled)
    except Exception:
        df['activity_cluster'] = 1
    return df

def predict(model, X):
    if model is None:
        # Demo: alternate sleep/awake
        return np.tile([0, 1], len(X)//2 + 1)[:len(X)]
    try:
        return model.predict(X).astype(int)
    except Exception:
        return np.zeros(len(X))

def prepare_X(df):
    feats = ['anglez_std_1m', 'enmo_std_1m', 'anglez_mean_2m', 'enmo_mean_2m', 'anglez', 'enmo', 'activity_cluster']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])
    return X

# --- Sidebar Navigation ---
st.sidebar.title("🌙 SmartSleep")
page = st.sidebar.radio("Navigate", ["Overview", "Analyze Data", "About"])

# --- Overview Page ---
if page == "Overview":
    st.title("😴AI FOR SLEEP DETECTION")
    st.info("Upload your wearable device data and discover your sleep patterns with our advanced models.")

    st.markdown("""
    - **Upload** your CSV or enter data manually
    - **Choose** a prediction model
    - **Visualize** your sleep/awake cycles
    - **Download** your results
    """)
    st.success("Ready to get started? Go to 'Analyze Data' in the sidebar!")

# --- Analyze Data Page ---
elif page == "Analyze Data":
    st.header("🛌 Sleep Pattern Analyzer")
    models = load_models()
    model_names = list(models.keys())
    model_choice = st.selectbox("Select Prediction Model", model_names, index=0)
    st.caption("Tip: The Optimized Model is tuned for higher accuracy.")

    data_mode = st.radio("How would you like to provide your data?", ["Upload CSV", "Manual Entry"])

    df = None  # <-- Add this line to initialize df

    if data_mode == "Upload CSV":
        file = st.file_uploader("Upload CSV with 'timestamp', 'anglez', 'enmo'", type="csv")
        if file:
            df = pd.read_csv(file)
            st.write("Preview:", df.head())
    elif data_mode == "Manual Entry":
        st.write("Add your data points below:")
        if "manual_df" not in st.session_state:
            st.session_state.manual_df = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])
        with st.form("manual_form"):
            c1, c2, c3 = st.columns(3)
            dt = c1.date_input("Date", datetime.date.today())
            tm = c1.time_input("Time", datetime.datetime.now().time())
            anglez = c2.number_input("Angle Z", value=0.0)
            enmo = c3.number_input("ENMO", value=0.0, min_value=0.0)
            add = st.form_submit_button("Add Row")
        if add:
            ts = datetime.datetime.combine(dt, tm)
            st.session_state.manual_df = pd.concat([
                st.session_state.manual_df,
                pd.DataFrame({'timestamp': [ts], 'anglez': [anglez], 'enmo': [enmo]})
            ], ignore_index=True)
        if not st.session_state.manual_df.empty:
            st.write(st.session_state.manual_df)
            if st.button("Clear All"):
                st.session_state.manual_df = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])
        df = st.session_state.manual_df if not st.session_state.manual_df.empty else None

    if df is not None and not df.empty:
        if st.button("Analyze Sleep"):
            with st.spinner("Analyzing..."):
                feat_df = create_features(df)
                X = prepare_X(feat_df)
                preds = predict(models[model_choice], X)
                feat_df['Predicted State'] = preds
                st.success("Analysis Complete!")
                # Pie chart
                sleep_pct = (preds == 1).mean() * 100
                awake_pct = 100 - sleep_pct
                st.plotly_chart(go.Figure(
                    data=[go.Pie(labels=["Sleep", "Awake"], values=[sleep_pct, awake_pct], hole=0.5)],
                    layout=go.Layout(title="Sleep vs Awake (%)")
                ), use_container_width=True)
                # Timeline
                st.plotly_chart(go.Figure(
                    data=[go.Scatter(
                        x=feat_df['timestamp'] if 'timestamp' in feat_df else feat_df.index,
                        y=feat_df['Predicted State'],
                        mode='lines+markers',
                        line=dict(color="#4b6cb7"),
                        name="Sleep State"
                    )],
                    layout=go.Layout(
                        title="Predicted Sleep Timeline",
                        yaxis=dict(tickvals=[0, 1], ticktext=["Awake", "Sleep"]),
                        xaxis_title="Time",
                        yaxis_title="State"
                    )
                ), use_container_width=True)
                # Download
                st.download_button(
                    "Download Results",
                    feat_df.to_csv(index=False),
                    file_name="restful_nights_results.csv",
                    mime="text/csv"
                )
                with st.expander("Show Data Table"):
                    st.dataframe(feat_df)

# --- About Page ---
elif page == "About":
    st.header("About SmartSleep")   
    st.markdown("""
    **Smartsleep** is a modern sleep analysis tool powered by machine learning.
    - Predicts sleep and awake states from wearable device data
    -The goal is to develop a model that detects sleep onset or wakeup  based on accelerometer data from wearable devices in children. 

                
    - Built using Streamlit 
    - Created for personal wellness and research
    """)
    st.info("All predictions are for informational purposes only.")

st.markdown("---")
st.caption("SmartSleep | 2025")
