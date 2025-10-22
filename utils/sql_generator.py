import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load T5 model (offline-friendly)
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_sql_query(prompt: str) -> str:
    """Generate valid SQL from natural language prompt with robust fallback."""
    try:
        # Try NLP model first
        input_text = f"translate English to SQL: {prompt}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=150)
        sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- Cleanup ---
        sql = sql.strip().replace('"', "'").replace(";", "")
        sql = sql.replace("table", "sales").replace("[", "").replace("]", "")

        # Fix known column names
        column_map = {
            "order id": "Order ID", "order date": "Order Date", "ship date": "Ship Date",
            "ship mode": "Ship Mode", "customer id": "Customer ID", "customer name": "Customer Name",
            "segment": "Segment", "country": "Country", "city": "City", "state": "State",
            "postal code": "Postal Code", "region": "Region", "product id": "Product ID",
            "category": "Category", "sub-category": "Sub-Category", "product name": "Product Name",
            "sales": "Sales", "quantity": "Quantity", "discount": "Discount", "profit": "Profit"
        }
        for k, v in column_map.items():
            sql = re.sub(rf"\b{k}\b", f"[{v}]", sql, flags=re.IGNORECASE)

        # Post-processing for aggregates and common issues
        # Replace "Total [Sales]" with SUM([Sales]) AS TotalSales
        sql = re.sub(r'Total\s+\[Sales\]', 'SUM([Sales]) AS TotalSales', sql, flags=re.IGNORECASE)
        # Remove tautological WHERE clauses like WHERE [Region] = [Region]
        sql = re.sub(r'\s+WHERE\s+\[Region\]\s*=\s*\[Region\]', '', sql, flags=re.IGNORECASE)
        # Add GROUP BY [Region] if prompt contains "by region" and GROUP BY not present
        if "by region" in prompt.lower() and "group by" not in sql.lower():
            sql += " GROUP BY [Region]"
        # Ensure table name is lowercase "sales"
        sql = re.sub(r'FROM\s+\[Sales\]', 'FROM sales', sql, flags=re.IGNORECASE)

        # Fix "Top N" queries
        match = re.search(r'top (\d+) (.+) by (.+)', prompt.lower())
        if match:
            num = match.group(1)
            field = match.group(3).strip()
            column = "[Profit]" if "profit" in field else "[Sales]"
            # Replace SELECT part
            sql = re.sub(r'SELECT Top \d+ .+ by \[.+\]', f'SELECT [Product Name], {column}', sql, flags=re.IGNORECASE)
            # Add ORDER BY and LIMIT if not present
            if 'ORDER BY' not in sql.upper():
                sql = sql.rstrip(';') + f' ORDER BY {column} DESC LIMIT {num};'

        # Fix year filtering
        sql = re.sub(r'WHERE Year = (\d+)', r"WHERE strftime('%Y', [Order Date]) = '\1'", sql, flags=re.IGNORECASE)

        # Ensure SELECT and FROM exist
        if not sql.lower().startswith("select") or "from" not in sql.lower():
            raise ValueError("Invalid SQL from model")

        return sql + ";"

    except Exception:
        # --- Smart fallback ---
        q = prompt.lower()
        if "sales" in q and "region" in q:
            return "SELECT [Region], SUM([Sales]) AS TotalSales FROM sales GROUP BY [Region];"
        elif "top" in q and "customer" in q:
            return "SELECT [Customer Name], SUM([Sales]) AS TotalSales FROM sales GROUP BY [Customer Name] ORDER BY TotalSales DESC LIMIT 5;"
        elif "profit" in q and "category" in q:
            return "SELECT [Category], SUM([Profit]) AS TotalProfit FROM sales GROUP BY [Category] ORDER BY TotalProfit DESC;"
        elif "average" in q and "discount" in q:
            return "SELECT AVG([Discount]) AS AvgDiscount FROM sales;"
        elif "month" in q or "trend" in q or "time" in q:
            return "SELECT strftime('%Y-%m', [Order Date]) AS Month, SUM([Sales]) AS MonthlySales FROM sales GROUP BY Month ORDER BY Month;"
        else:
            return "SELECT * FROM sales LIMIT 10;"
