from utils.sql_generator import generate_sql_query

# Test the query that was failing
prompt = "Top 5 products by profit in 2020"
sql = generate_sql_query(prompt)
print(f"Generated SQL: {sql}")

# Expected: SELECT [Product Name], [Profit] FROM sales WHERE strftime('%Y', [Order Date]) = '2020' ORDER BY [Profit] DESC LIMIT 5;
