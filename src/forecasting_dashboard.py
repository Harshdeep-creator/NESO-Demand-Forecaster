import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Energy Demand Forecasting Platform",
    layout="wide"
)

DATA_FOLDER = "data/processed"
PLOTS_FOLDER = "results/plots"

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("Energy Demand Forecasting Platform")

st.markdown(
"""
Deep learning forecasting system trained on National Energy System Operator (NESO) GB Grid demand data.

Models implemented

- Statistical baselines  
- LSTM neural network  
- Transformer architecture  

Evaluation includes walk-forward validation and statistical significance testing.
"""
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Explorer",
        "Model Performance",
        "Forecast Gallery",
        "Backtest Results",
        "Model Leaderboard",
        "Statistical Test"
    ],
    key="main_nav"
)

# -------------------------------------------------
# OVERVIEW
# -------------------------------------------------

if page == "Overview":

    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset Range", "2019 – 2024")
    c2.metric("Lookback Window", "90 Days")
    c3.metric("Forecast Horizon", "7 Days")
    c4.metric("Backtest Windows", "71")

    st.markdown("---")

    st.subheader("Pipeline")

    st.markdown(
    """
    1. Data preprocessing  
    2. Baseline statistical models  
    3. Deep learning forecasting models  
    4. Walk-forward validation  
    5. Diebold–Mariano statistical testing  
    """
    )

# -------------------------------------------------
# DATASET EXPLORER
# -------------------------------------------------

elif page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    if os.path.exists(DATA_FOLDER):

        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

        if not files:
            st.warning("No datasets found.")

        else:

            display_names = {
                f: "NESO GB Grid Demand (2019–2024)" if "2019_2026" in f else f
                for f in files
            }

            selected_display = st.selectbox(
                "Select Dataset",
                list(display_names.values())
            )

            selected_file = [
                k for k, v in display_names.items() if v == selected_display
            ][0]

            df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))

            st.caption("Dataset: NESO GB Grid Demand (2019–2024)")

            st.dataframe(df.head(), use_container_width=True)

            c1, c2 = st.columns(2)

            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])

            st.markdown("Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            st.markdown("Time Series Visualization")

            if "date" in df.columns:

                df["date"] = pd.to_datetime(df["date"])

                fig = px.line(
                    df,
                    x="date",
                    y=df.columns[1],
                    title="Energy Demand Over Time"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                fig = px.line(df, y=df.columns[0])
                st.plotly_chart(fig, use_container_width=True)

    else:

        st.error("data/processed folder not found")

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

elif page == "Model Performance":

    st.subheader("Model Performance Comparison")

    df = pd.DataFrame({

        "Model": [
            "Naive",
            "Seasonal Naive",
            "Moving Average",
            "LSTM",
            "Transformer"
        ],

        "MAE": [
            0.203201,
            0.225035,
            0.179111,
            0.098599,
            0.069000
        ],

        "RMSE": [
            0.256921,
            0.273033,
            0.218067,
            0.127598,
            0.095000
        ],

        "MAPE": [
            100.198158,
            131.126709,
            118.195213,
            62.505573,
            34.207000
        ],

        "Directional Accuracy": [
            None,
            None,
            None,
            72.604,
            84.478
        ]

    })

    st.dataframe(df, use_container_width=True)

    st.markdown("Error Metrics Comparison")

    metric = st.selectbox(
        "Select Metric",
        ["MAE", "RMSE", "MAPE"]
    )

    fig = px.bar(
        df,
        x="Model",
        y=metric,
        color="Model",
        title=f"{metric} Comparison Across Models"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Directional Accuracy (Deep Learning Models)")

    acc_df = df[df["Directional Accuracy"].notna()]

    fig2 = px.bar(
        acc_df,
        x="Model",
        y="Directional Accuracy",
        color="Model",
        title="Directional Accuracy: LSTM vs Transformer"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# FORECAST GALLERY
# -------------------------------------------------

elif page == "Forecast Gallery":

    st.subheader("Model Forecast Plots")

    if os.path.exists(PLOTS_FOLDER):

        plots = [f for f in os.listdir(PLOTS_FOLDER) if f.endswith(".png")]

        if not plots:
            st.warning("No plots found.")

        else:

            mode = st.radio(
                "Display Mode",
                ["Single Plot", "All Plots"]
            )

            if mode == "Single Plot":

                selected_plot = st.selectbox("Select Plot", plots)

                path = os.path.join(PLOTS_FOLDER, selected_plot)

                st.image(path, caption=selected_plot, use_container_width=True)

            else:

                cols = st.columns(2)

                for i, plot in enumerate(plots):

                    path = os.path.join(PLOTS_FOLDER, plot)

                    cols[i % 2].image(
                        path,
                        caption=plot,
                        use_container_width=True
                    )

    else:

        st.error("results/plots folder not found")

# -------------------------------------------------
# BACKTEST RESULTS
# -------------------------------------------------

elif page == "Backtest Results":

    st.subheader("Walk Forward Backtest")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", "0.074")
    c2.metric("RMSE", "0.101")
    c3.metric("MAPE", "30.52%")
    c4.metric("Directional Accuracy", "82.63%")

    st.info("Total backtest windows: 71")

# -------------------------------------------------
# MODEL LEADERBOARD
# -------------------------------------------------

elif page == "Model Leaderboard":

    st.subheader("Model Leaderboard")

    df = pd.DataFrame({

        "Model": [
            "Naive",
            "Seasonal Naive",
            "Moving Average",
            "LSTM",
            "Transformer"
        ],

        "MAE": [0.203, 0.225, 0.179, 0.099, 0.069],
        "RMSE": [0.256, 0.273, 0.218, 0.127, 0.095],
        "MAPE": [100.19, 131.12, 118.19, 62.50, 34.20]
    })

    df = df.sort_values("MAE")

    st.dataframe(df, use_container_width=True)

    fig = px.bar(
        df,
        x="Model",
        y="RMSE",
        color="Model",
        title="RMSE Comparison Across Models"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# STATISTICAL TEST
# -------------------------------------------------

elif page == "Statistical Test":

    st.subheader("Diebold–Mariano Statistical Test")

    st.write("Comparison: LSTM vs Naive Baseline")

    c1, c2 = st.columns(2)

    c1.metric("DM Statistic", "-14.957")
    c2.metric("p-value", "0.000")

    st.success("Result: LSTM significantly outperforms the naive baseline.")
