import pandas as pd

def data_quality_report(df):
    report = pd.DataFrame({
        'Column': df.columns,
        'MissingValues': df.isnull().sum(),
        'UniqueValues': df.nunique(),
        'DataType': df.dtypes.astype(str)
    })
    return report
