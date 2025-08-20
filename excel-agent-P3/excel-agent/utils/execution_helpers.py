# excel-agent/utils/execution_helpers.py
"""
Additional utilities for Phase 3 execution
"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

class InstructionParser:
    """
    Enhanced rule-based instruction parser for fallback scenarios
    """
    
    ACTION_KEYWORDS = {
        'merge': ['join', 'merge', 'combine', 'link', 'connect'],
        'filter': ['filter', 'where', 'only', 'remove', 'exclude', 'select'],
        'aggregate': ['group', 'aggregate', 'sum', 'count', 'average', 'total', 'summarize'],
        'pivot': ['pivot', 'crosstab', 'cross-tab', 'transpose'],
        'visualize': ['chart', 'plot', 'visualize', 'graph', 'show'],
        'clean': ['clean', 'fix', 'standardize', 'normalize', 'dedupe'],
        'sort': ['sort', 'order', 'arrange', 'rank']
    }
    
    @classmethod
    def extract_sheet_names(cls, instruction: str, available_sheets: List[str]) -> List[str]:
        """Extract sheet names mentioned in instruction"""
        mentioned_sheets = []
        instruction_lower = instruction.lower()
        
        for sheet in available_sheets:
            if sheet.lower() in instruction_lower:
                mentioned_sheets.append(sheet)
        
        return mentioned_sheets if mentioned_sheets else available_sheets[:1]
    
    @classmethod
    def extract_column_names(cls, instruction: str, available_columns: Dict[str, List[str]]) -> List[str]:
        """Extract column names mentioned in instruction"""
        mentioned_columns = []
        instruction_lower = instruction.lower()
        
        all_columns = []
        for sheet_cols in available_columns.values():
            all_columns.extend(sheet_cols)
        
        for col in set(all_columns):
            if col.lower() in instruction_lower:
                mentioned_columns.append(col)
        
        return mentioned_columns
    
    @classmethod
    def detect_action(cls, instruction: str) -> str:
        """Detect primary action from instruction"""
        instruction_lower = instruction.lower()
        
        for action, keywords in cls.ACTION_KEYWORDS.items():
            if any(keyword in instruction_lower for keyword in keywords):
                return action
        
        return 'summary'  # Default action

class ResultFormatter:
    """
    Format execution results for better display
    """
    
    @staticmethod
    def format_execution_result(result: Dict[str, Any]) -> str:
        """Format execution result for console display"""
        if result["status"] == "error":
            return f"❌ Error: {result['message']}"
        
        output = [f"✅ {result['message']}"]
        
        # Add specific details based on result type
        if "output_path" in result:
            output.append(f"📁 Saved to: {result['output_path']}")
        
        if "result_shape" in result:
            rows, cols = result["result_shape"]
            output.append(f"📊 Result: {rows:,} rows × {cols} columns")
        
        if "cleaning_steps" in result:
            output.append("🧹 Cleaning performed:")
            for step in result["cleaning_steps"]:
                output.append(f"   • {step}")
        
        if "chart_types" in result:
            output.append(f"📈 Visualizations: {', '.join(result['chart_types'])}")
        
        return "\n".join(output)
    
    @staticmethod
    def create_execution_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of multiple execution results"""
        summary = {
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r.get("execution_result", {}).get("status") == "success"),
            "failed_tasks": sum(1 for r in results if r.get("execution_result", {}).get("status") == "error"),
            "outputs_generated": []
        }
        
        for result in results:
            exec_result = result.get("execution_result", {})
            if exec_result.get("status") == "success" and "output_path" in exec_result:
                summary["outputs_generated"].append(exec_result["output_path"])
        
        return summary

class DataQualityChecker:
    """
    Automated data quality assessment
    """
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame, sheet_name: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        assessment = {
            "sheet_name": sheet_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues": [],
            "recommendations": []
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.1]  # More than 10% missing
        
        if len(high_missing) > 0:
            assessment["issues"].append(f"High missing values in: {list(high_missing.index)}")
            assessment["recommendations"].append("Consider data imputation or column removal")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            assessment["issues"].append(f"{duplicate_count} duplicate rows found")
            assessment["recommendations"].append("Remove duplicate rows")
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        outlier_cols = []
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            assessment["issues"].append(f"Potential outliers in: {outlier_cols}")
            assessment["recommendations"].append("Review and potentially remove outliers")
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns
        potential_numeric = []
        
        for col in object_cols:
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                potential_numeric.append(col)
            except:
                pass
        
        if potential_numeric:
            assessment["recommendations"].append(f"Consider converting to numeric: {potential_numeric}")
        
        return assessment

def generate_automated_suggestions(sheets: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Generate automated suggestions based on data analysis
    """
    suggestions = []
    sheet_names = list(sheets.keys())
    
    # Multi-sheet suggestions
    if len(sheet_names) > 1:
        # Look for common columns for joining
        all_columns = {}
        for name, df in sheets.items():
            all_columns[name] = set(df.columns)
        
        # Find common columns between sheets
        for i, sheet1 in enumerate(sheet_names):
            for sheet2 in sheet_names[i+1:]:
                common_cols = all_columns[sheet1] & all_columns[sheet2]
                if common_cols:
                    for col in list(common_cols)[:2]:  # Suggest up to 2 joins
                        suggestions.append(f"Merge {sheet1} and {sheet2} on '{col}' column")
    
    # Single sheet suggestions
    for name, df in sheets.items():
        # Data quality suggestions
        quality_check = DataQualityChecker.assess_data_quality(df, name)
        if quality_check["issues"]:
            suggestions.append(f"Clean data quality issues in {name}")
        
        # Visualization suggestions
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            suggestions.append(f"Create correlation analysis for {name}")
        
        # Aggregation suggestions
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            suggestions.append(f"Aggregate {name} by {cat_col}")
    
    return suggestions[:6]  # Return top 6 suggestions