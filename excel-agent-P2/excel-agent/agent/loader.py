import pandas as pd
from typing import Dict, Any

def _describe_df(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns.tolist())),
        "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
    }

def summarize_excel(file_path: str) -> Dict[str, Any]:
    xls = pd.ExcelFile(file_path)
    summary = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        summary[sheet] = _describe_df(df)
    return summary

def sample_sheets(file_path: str, n: int = 5) -> Dict[str, Any]:
    """Return first n rows of each sheet as records (safe to show in prompts)."""
    xls = pd.ExcelFile(file_path)
    samples = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # keep only first n rows and convert to records for readability
        samples[sheet] = df.head(n).to_dict(orient="records")
    return samples