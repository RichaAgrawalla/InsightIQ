import pandas as pd
from prophet import Prophet

def forecast_sales(df):
    # --- Standardize column names: snake_case, strip spaces ---
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # --- Detect date and sales columns ---
    date_col = next((c for c in df.columns if "order_date" in c.lower()), None)
    sales_col = next((c for c in df.columns if "sales" in c.lower()), None)

    if not date_col or not sales_col:
        raise ValueError("Could not find 'Order Date' or 'Sales' column.")

    # --- Prepare data for Prophet ---
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])
    sales_df = df.groupby(date_col)[sales_col].sum().reset_index()
    sales_df.columns = ["ds", "y"]

    # --- Fit Prophet model ---
    model = Prophet()
    model.fit(sales_df)

    # --- Forecast next 30 days ---
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
