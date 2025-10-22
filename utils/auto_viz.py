import plotly.express as px
import streamlit as st
import pandas as pd

def auto_viz(df: pd.DataFrame):
    """Automatically visualize a given DataFrame in Streamlit using Plotly."""
    if df.empty:
        st.warning("No data to visualize.")
        return

    # --- Detect data types ---
    df = df.reset_index(drop=True)
    num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    # --- Priority logic ---
    if len(date_cols) > 0 and len(num_cols) > 0:
        col_x, col_y = date_cols[0], num_cols[0]
        fig = px.line(df, x=col_x, y=col_y, title=f"{col_y} Over Time")
    elif len(cat_cols) > 0 and len(num_cols) > 0:
        col_x, col_y = cat_cols[0], num_cols[0]
        fig = px.bar(df, x=col_x, y=col_y, text_auto=True, title=f"{col_y} by {col_x}")
    elif len(num_cols) == 1:
        fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
    elif len(cat_cols) == 1:
        fig = px.pie(df, names=cat_cols[0], title=f"Proportion of {cat_cols[0]}")
    else:
        st.info("⚙️ Unable to automatically determine visualization type.")
        st.dataframe(df)
        return

    st.plotly_chart(fig, use_container_width=True)
