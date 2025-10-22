import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Load a lightweight T5 model ---
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# --- 1️⃣ Generate SQL from natural prompt ---
def generate_sql(prompt):
    """
    Uses FLAN-T5 to convert a natural-language question into an SQL-like query.
    """
    try:
        input_text = f"Translate this question into an SQL query: {prompt}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=80)
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up any unwanted tokens
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return sql_query
    except Exception as e:
        return f"Error generating SQL: {e}"


# --- 2️⃣ Fix SQL to match dataset columns dynamically ---
def fix_sql_columns(sql, df):
    """
    Replaces column names in SQL to match the DataFrame’s actual column names.
    """
    corrected_sql = sql
    for col in df.columns:
        col_clean = col.replace(" ", "_").lower()
        corrected_sql = corrected_sql.replace(col_clean, f"`{col}`")
        corrected_sql = corrected_sql.replace(col.lower(), f"`{col}`")
    return corrected_sql


# --- 3️⃣ Generate contextual business insights ---
def generate_insight(df, user_prompt):
    """
    Automatically generates short insights based on the result and user query.
    """
    try:
        insight = ""

        # Profit-based insights
        if "profit" in user_prompt.lower():
            top_region = df.groupby("Region")["Profit"].sum().idxmax()
            profit_value = df.groupby("Region")["Profit"].sum().max()
            insight = f"💰 The **{top_region}** region generated the highest profit (${profit_value:,.2f})."

        # Sales-based insights
        elif "sales" in user_prompt.lower():
            if "TotalSales" in df.columns:
                # Aggregated data
                total_sales = df["TotalSales"].sum()
                if "Region" in df.columns:
                    best_region = df.loc[df["TotalSales"].idxmax(), "Region"]
                    insight = f"📊 Total sales are **${total_sales:,.0f}**, with the **{best_region}** region leading."
                else:
                    insight = f"📊 Total sales are **${total_sales:,.0f}**."
            else:
                # Raw data
                total_sales = df["Sales"].sum()
                best_region = df.groupby("Region")["Sales"].sum().idxmax()
                insight = f"📊 Total sales are **${total_sales:,.0f}**, with the **{best_region}** region leading."

        # Discount / Customer behavior
        elif "discount" in user_prompt.lower():
            avg_discount = df["Discount"].mean()
            insight = f"🎯 The average discount across orders is **{avg_discount:.2%}**."

        # Region or Segment
        elif "region" in user_prompt.lower() or "segment" in user_prompt.lower():
            region_sales = df.groupby("Region")["Sales"].sum().idxmax()
            insight = f"🌍 The **{region_sales}** region shows the best overall performance."

        # Category-based prompt
        elif "category" in user_prompt.lower():
            best_cat = df.groupby("Category")["Profit"].sum().idxmax()
            insight = f"🏷️ The **{best_cat}** category is the top performer in profitability."

        # Default insight
        else:
            insight = "✅ Query executed successfully. Data summary generated."

        return insight

    except Exception as e:
        return f"⚠️ Could not generate insight automatically: {e}"
