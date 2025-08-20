# excel-agent/agent/executor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import os
from pathlib import Path
import google.generativeai as genai
from config import GOOGLE_API_KEY

class TaskExecutor:
    """
    Phase 3: Task Executor Agent
    Handles both suggested automations and custom user instructions
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.sheets = {}
        self.load_sheets()
        self._configure_gemini()
    
    def load_sheets(self):
        """Load all sheets into memory"""
        xls = pd.ExcelFile(self.file_path)
        for sheet_name in xls.sheet_names:
            self.sheets[sheet_name] = xls.parse(sheet_name)
        print(f"📊 Loaded {len(self.sheets)} sheets: {list(self.sheets.keys())}")
    
    def _configure_gemini(self):
        """Configure Gemini for instruction parsing"""
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Parse natural language instruction into structured format using Gemini
        """
        # Get sheet info for context
        sheet_info = {}
        for name, df in self.sheets.items():
            sheet_info[name] = {
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "sample": df.head(2).to_dict('records') if not df.empty else []
            }
        
        prompt = f"""
        You are a data analysis instruction parser. Parse the user instruction into a structured JSON format.
        
        Available sheets and columns:
        {json.dumps(sheet_info, indent=2, default=str)}
        
        User Instruction: "{instruction}"
        
        Parse this into JSON with these possible actions:
        - "merge": Join datasets
        - "filter": Filter data
        - "aggregate": Group and summarize
        - "pivot": Create pivot table
        - "visualize": Create charts
        - "clean": Data cleaning operations
        - "summary": Generate summaries
        
        Return ONLY valid JSON in this format:
        {{
            "action": "action_type",
            "sheets": ["sheet1", "sheet2"],
            "columns": ["col1", "col2"],
            "conditions": {{"filter_conditions": "value"}},
            "operation": "specific_operation",
            "output_name": "result_name"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._fallback_parse(instruction)
        except Exception as e:
            print(f"⚠️ LLM parsing failed: {e}")
            return self._fallback_parse(instruction)
    
    def _fallback_parse(self, instruction: str) -> Dict[str, Any]:
        """Simple rule-based parsing as fallback"""
        instruction_lower = instruction.lower()
        
        # Detect action type
        if any(word in instruction_lower for word in ['join', 'merge', 'combine']):
            action = "merge"
        elif any(word in instruction_lower for word in ['filter', 'where', 'only']):
            action = "filter"
        elif any(word in instruction_lower for word in ['group', 'aggregate', 'sum', 'count']):
            action = "aggregate"
        elif any(word in instruction_lower for word in ['pivot', 'cross']):
            action = "pivot"
        elif any(word in instruction_lower for word in ['chart', 'plot', 'visualize', 'graph']):
            action = "visualize"
        elif any(word in instruction_lower for word in ['clean', 'remove', 'drop']):
            action = "clean"
        else:
            action = "summary"
        
        # Find sheet names mentioned
        sheets = [name for name in self.sheets.keys() 
                 if name.lower() in instruction_lower]
        
        return {
            "action": action,
            "sheets": sheets if sheets else list(self.sheets.keys()),
            "columns": [],
            "conditions": {},
            "operation": instruction,
            "output_name": f"{action}_result"
        }
    
    def execute_instruction(self, instruction_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute parsed instruction and return results
        """
        action = instruction_dict.get("action", "summary")
        
        try:
            if action == "merge":
                return self._execute_merge(instruction_dict)
            elif action == "filter":
                return self._execute_filter(instruction_dict)
            elif action == "aggregate":
                return self._execute_aggregate(instruction_dict)
            elif action == "pivot":
                return self._execute_pivot(instruction_dict)
            elif action == "visualize":
                return self._execute_visualize(instruction_dict)
            elif action == "clean":
                return self._execute_clean(instruction_dict)
            else:
                return self._execute_summary(instruction_dict)
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Execution failed: {str(e)}",
                "action": action
            }
    
    def _execute_merge(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute merge/join operations"""
        sheets = instruction.get("sheets", [])
        if len(sheets) < 2:
            return {"status": "error", "message": "Need at least 2 sheets to merge"}
        
        df1 = self.sheets[sheets[0]]
        df2 = self.sheets[sheets[1]]
        
        # Find common columns for join
        common_cols = list(set(df1.columns) & set(df2.columns))
        if not common_cols:
            return {"status": "error", "message": "No common columns found for merge"}
        
        # Use first common column as join key
        join_key = common_cols[0]
        
        try:
            result = pd.merge(df1, df2, on=join_key, how='inner', suffixes=('_left', '_right'))
            output_path = self._save_dataframe(result, f"merged_{sheets[0]}_{sheets[1]}")
            
            return {
                "status": "success",
                "message": f"Merged {sheets[0]} and {sheets[1]} on '{join_key}'",
                "output_path": output_path,
                "result_shape": result.shape,
                "join_key": join_key
            }
        except Exception as e:
            return {"status": "error", "message": f"Merge failed: {str(e)}"}
    
    def _execute_filter(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute filter operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            return {"status": "error", "message": "No sheets specified"}
        
        df = self.sheets[sheets[0]].copy()
        original_shape = df.shape
        
        # Simple filtering based on instruction text
        operation = instruction.get("operation", "").lower()
        
        # Remove null values
        if "null" in operation or "empty" in operation:
            df = df.dropna()
        
        # Remove duplicates
        if "duplicate" in operation:
            df = df.drop_duplicates()
        
        # Remove zero values (if numeric columns exist)
        if "zero" in operation:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df = df[df[col] != 0]
        
        output_path = self._save_dataframe(df, f"filtered_{sheets[0]}")
        
        return {
            "status": "success",
            "message": f"Filtered {sheets[0]}. Rows: {original_shape[0]} → {df.shape[0]}",
            "output_path": output_path,
            "original_shape": original_shape,
            "result_shape": df.shape
        }
    
    def _execute_aggregate(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            return {"status": "error", "message": "No sheets specified"}
        
        df = self.sheets[sheets[0]]
        
        # Find categorical and numeric columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not cat_cols or not num_cols:
            return {"status": "error", "message": "Need both categorical and numeric columns for aggregation"}
        
        # Group by first categorical column, sum numeric columns
        group_col = cat_cols[0]
        
        try:
            result = df.groupby(group_col)[num_cols].agg(['sum', 'mean', 'count']).round(2)
            result.columns = ['_'.join(col) for col in result.columns]
            result = result.reset_index()
            
            output_path = self._save_dataframe(result, f"aggregated_{sheets[0]}")
            
            return {
                "status": "success",
                "message": f"Aggregated {sheets[0]} by '{group_col}'",
                "output_path": output_path,
                "group_column": group_col,
                "result_shape": result.shape
            }
        except Exception as e:
            return {"status": "error", "message": f"Aggregation failed: {str(e)}"}
    
    def _execute_pivot(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pivot table operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            return {"status": "error", "message": "No sheets specified"}
        
        df = self.sheets[sheets[0]]
        
        # Find suitable columns for pivot
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(cat_cols) < 2 or not num_cols:
            return {"status": "error", "message": "Need at least 2 categorical and 1 numeric column for pivot"}
        
        try:
            pivot_result = pd.pivot_table(
                df, 
                index=cat_cols[0], 
                columns=cat_cols[1], 
                values=num_cols[0], 
                aggfunc='sum', 
                fill_value=0
            )
            
            # Reset index to make it a regular DataFrame
            pivot_result = pivot_result.reset_index()
            
            output_path = self._save_dataframe(pivot_result, f"pivot_{sheets[0]}")
            
            return {
                "status": "success",
                "message": f"Created pivot table for {sheets[0]}",
                "output_path": output_path,
                "pivot_columns": [cat_cols[0], cat_cols[1], num_cols[0]],
                "result_shape": pivot_result.shape
            }
        except Exception as e:
            return {"status": "error", "message": f"Pivot failed: {str(e)}"}
    
    def _execute_visualize(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            return {"status": "error", "message": "No sheets specified"}
        
        df = self.sheets[sheets[0]]
        
        # Create visualizations
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Data Analysis: {sheets[0]}', fontsize=16)
        
        try:
            # 1. Numeric columns distribution
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                df[num_cols[0]].hist(bins=20, ax=axes[0,0])
                axes[0,0].set_title(f'Distribution of {num_cols[0]}')
                axes[0,0].set_xlabel(num_cols[0])
                axes[0,0].set_ylabel('Frequency')
            
            # 2. Correlation heatmap
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                sns.heatmap(corr, annot=True, ax=axes[0,1], cmap='coolwarm')
                axes[0,1].set_title('Correlation Matrix')
            
            # 3. Categorical analysis
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                cat_col = cat_cols[0]
                value_counts = df[cat_col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[1,0])
                axes[1,0].set_title(f'Top 10 {cat_col} Values')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Scatter plot if we have 2+ numeric columns
            if len(num_cols) >= 2:
                df.plot.scatter(x=num_cols[0], y=num_cols[1], ax=axes[1,1], alpha=0.6)
                axes[1,1].set_title(f'{num_cols[0]} vs {num_cols[1]}')
            
            # Save plot
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plot_path = output_dir / f"visualization_{sheets[0]}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "status": "success",
                "message": f"Created visualizations for {sheets[0]}",
                "output_path": str(plot_path),
                "chart_types": ["histogram", "correlation", "bar_chart", "scatter"]
            }
            
        except Exception as e:
            plt.close()
            return {"status": "error", "message": f"Visualization failed: {str(e)}"}
    
    def _execute_clean(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data cleaning operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            return {"status": "error", "message": "No sheets specified"}
        
        df = self.sheets[sheets[0]].copy()
        original_shape = df.shape
        cleaning_steps = []
        
        try:
            # Remove duplicates
            before_dup = len(df)
            df = df.drop_duplicates()
            after_dup = len(df)
            if before_dup != after_dup:
                cleaning_steps.append(f"Removed {before_dup - after_dup} duplicate rows")
            
            # Handle missing values
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            
            for col in cols_with_nulls.index:
                if df[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        cleaning_steps.append(f"Filled {col} nulls with mode: {mode_val[0]}")
                else:
                    # Fill numeric with median
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    cleaning_steps.append(f"Filled {col} nulls with median: {median_val}")
            
            # Remove outliers from numeric columns (using IQR method)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_outlier = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_outlier = len(df)
                
                if before_outlier != after_outlier:
                    cleaning_steps.append(f"Removed {before_outlier - after_outlier} outliers from {col}")
            
            output_path = self._save_dataframe(df, f"cleaned_{sheets[0]}")
            
            return {
                "status": "success",
                "message": f"Cleaned {sheets[0]}. Shape: {original_shape} → {df.shape}",
                "output_path": output_path,
                "cleaning_steps": cleaning_steps,
                "original_shape": original_shape,
                "result_shape": df.shape
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Cleaning failed: {str(e)}"}
    
    def _execute_summary(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summary operations"""
        sheets = instruction.get("sheets", [])
        if not sheets:
            sheets = list(self.sheets.keys())
        
        summary_data = {}
        
        for sheet_name in sheets:
            df = self.sheets[sheet_name]
            
            sheet_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": {},
                "categorical_summary": {}
            }
            
            # Numeric columns summary
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                sheet_summary["numeric_summary"] = df[num_cols].describe().to_dict()
            
            # Categorical columns summary
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                sheet_summary["categorical_summary"][col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
            
            summary_data[sheet_name] = sheet_summary
        
        # Save summary
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / "detailed_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return {
            "status": "success",
            "message": f"Generated detailed summary for {len(sheets)} sheet(s)",
            "output_path": str(summary_path),
            "sheets_analyzed": sheets
        }
    
    def _save_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """Save DataFrame to Excel file"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{name}.xlsx"
        
        df.to_excel(output_path, index=False)
        return str(output_path)

# Convenience functions for backward compatibility
def execute_instruction(sheets: dict, instruction: dict):
    """
    Legacy function signature for backward compatibility
    """
    # This would need the file path - for now, raise helpful error
    raise NotImplementedError(
        "Use TaskExecutor class instead:\n"
        "executor = TaskExecutor('your_file.xlsx')\n"
        "result = executor.execute_instruction(instruction_dict)"
    )

def execute_natural_language(file_path: str, instruction: str) -> Dict[str, Any]:
    """
    Main entry point for natural language instructions
    """
    executor = TaskExecutor(file_path)
    parsed_instruction = executor.parse_instruction(instruction)
    result = executor.execute_instruction(parsed_instruction)
    
    return {
        "parsed_instruction": parsed_instruction,
        "execution_result": result
    }