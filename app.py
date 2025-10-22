import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import numpy as np
from sqlalchemy import create_engine
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Custom utils ---
from utils.forecast import forecast_sales
from utils.data_quality import data_quality_report
from utils.insights import generate_insight  # <— make sure this file exists
from utils.sql_generator import generate_sql_query
from utils.auto_viz import auto_viz
from utils.sql_generator import generate_sql_query as generate_sql_from_prompt


# --- Streamlit setup ---
st.set_page_config(page_title="InsightIQ", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv", encoding="latin-1")
    return df

df = load_data()

# --- Create SQL DB ---
conn = sqlite3.connect("insightiq.db", check_same_thread=False)
df.to_sql("sales", conn, if_exists="replace", index=False)

# --- Load NLP model ---
@st.cache_resource
def load_text_to_sql_model():
    model_name = "mrm8488/t5-base-finetuned-wikiSQL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_text_to_sql_model()

# --- Sidebar Navigation ---
st.sidebar.image("assets/logo.png", width=150)
st.sidebar.title("🔍 InsightIQ Dashboard")
page = st.sidebar.radio("Navigate", ["Ask AI", "KPI Dashboard", "Forecasting", "Data Quality Report"])


# --- PAGE 1: AI Query ---
if page == "Ask AI":
    st.title("💬 AI-Powered Data Insights")
    user_query = st.text_input("Ask your data a question (e.g., 'Top 5 products by profit in 2020')")

    if st.button("Generate Insights"):
        with st.spinner("Generating SQL query and insights... ⏳"):
            sql_query = generate_sql_from_prompt(user_query)
            st.code(sql_query, language="sql")

            try:
                result_df = pd.read_sql_query(sql_query, conn)
                if not result_df.empty:
                    st.dataframe(result_df)
                    # Try automatic visualization
                    try:
                        if len(result_df.columns) >= 2:
                            st.plotly_chart(
                                auto_viz(result_df)
                            )
                    except Exception:
                        st.info("⚙️ Could not auto-plot data.")
                    # Generate insights
                    insight = generate_insight(result_df, user_query)
                    st.success(insight)
                else:
                    st.warning("No data returned for this query.")
            except Exception as e:
                st.error(f"❌ Could not execute query: {e}")


# --- PAGE 2: KPI Dashboard ---
elif page == "KPI Dashboard":
    st.title("📊 Business Performance Overview")

    def find_col(df, keywords):
        for col in df.columns:
            if all(k in col.lower() for k in keywords):
                return col
        return None

    sales_col = find_col(df, ["sales"])
    profit_col = find_col(df, ["profit"])
    discount_col = find_col(df, ["discount"])
    order_col = find_col(df, ["order", "id"])
    region_col = find_col(df, ["region"])
    category_col = find_col(df, ["category"])

    total_sales = df[sales_col].sum() if sales_col else 0
    total_profit = df[profit_col].sum() if profit_col else 0
    avg_discount = df[discount_col].mean() if discount_col else 0
    num_orders = df[order_col].nunique() if order_col else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
    col3.metric("🎯 Avg Discount", f"{avg_discount:.2%}")
    col4.metric("🧾 Total Orders", f"{num_orders:,}")

    if region_col and sales_col:
        region_data = df.groupby(region_col)[sales_col].sum().reset_index()
        st.plotly_chart(px.bar(region_data, x=region_col, y=sales_col, title="Sales by Region", text_auto=True))
    if category_col and profit_col:
        cat_data = df.groupby(category_col)[profit_col].sum().reset_index()
        st.plotly_chart(px.bar(cat_data, x=category_col, y=profit_col, title="Profit by Category", text_auto=True))


# --- PAGE 3: Forecasting ---
elif page == "Forecasting":
    st.title("📅 Predict Future Sales Trends")

    try:
        forecast_df = forecast_sales(df)
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"],
                                 mode="lines", name="Predicted Sales", line=dict(color="royalblue", width=2)))
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"],
                                 mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"],
                                 mode='lines', name='Lower Bound', fill='tonexty',
                                 fillcolor='rgba(65,105,225,0.2)', line=dict(width=0), showlegend=False))
        fig.update_layout(title="📈 30-Day Sales Forecast",
                          xaxis_title="Date", yaxis_title="Predicted Sales ($)", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast Data", csv, "sales_forecast.csv", "text/csv")

        # Insights
        first, last = forecast_df.iloc[-30]["yhat"], forecast_df.iloc[-1]["yhat"]
        change = ((last - first) / first) * 100 if first != 0 else 0
        if change > 0:
            st.success(f"🚀 Sales expected to rise by {change:.1f}% in next 30 days.")
        elif change < 0:
            st.warning(f"📉 Sales may drop by {abs(change):.1f}% in next 30 days.")
        else:
            st.info("⚖️ Sales likely to remain stable.")

    except Exception as e:
        st.error(f"Forecasting failed: {e}")


# --- PAGE 4: Data Quality ---
elif page == "Data Quality Report":
    st.title("🧾 Data Quality and Profiling")
    dq = data_quality_report(df)
    st.dataframe(dq)
    st.download_button("📥 Download Report", dq.to_csv(index=False), "data_quality_report.csv")
