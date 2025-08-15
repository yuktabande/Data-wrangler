import pandas as pd

def analyze_excel(file_path):
    excel_file = pd.ExcelFile(file_path)
    summary = {}

    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        summary[sheet_name] = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

    return summary