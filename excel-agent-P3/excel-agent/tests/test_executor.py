# excel-agent/tests/test_executor.py
"""
Test suite for the Task Executor Agent (Phase 3)
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.executor import TaskExecutor, execute_natural_language

class TestTaskExecutor:
    """Test cases for TaskExecutor class"""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample Excel file for testing"""
        # Create sample data
        sales_data = pd.DataFrame({
            'Order_ID': [1, 2, 3, 4, 5],
            'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
            'Quantity': [10, 5, 8, 12, 6],
            'Price': [100.0, 200.0, 100.0, 150.0, 200.0],
            'Date': pd.date_range('2024-01-01', periods=5)
        })
        
        returns_data = pd.DataFrame({
            'Order_ID': [2, 4],
            'Return_Quantity': [2, 3],
            'Return_Reason': ['Defective', 'Wrong Size'],
            'Return_Date': pd.date_range('2024-01-10', periods=2)
        })
        
        # Save to temporary Excel file
        excel_path = tmp_path / "test_data.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            sales_data.to_excel(writer, sheet_name='Sales', index=False)
            returns_data.to_excel(writer, sheet_name='Returns', index=False)
        
        return str(excel_path)
    
    def test_initialization(self, sample_data):
        """Test TaskExecutor initialization"""
        executor = TaskExecutor(sample_data)
        
        assert len(executor.sheets) == 2
        assert 'Sales' in executor.sheets
        assert 'Returns' in executor.sheets
        assert len(executor.sheets['Sales']) == 5
        assert len(executor.sheets['Returns']) == 2
    
    def test_parse_instruction_fallback(self, sample_data):
        """Test fallback instruction parsing"""
        executor = TaskExecutor(sample_data)
        
        # Test merge instruction
        merge_instruction = "join Sales and Returns on Order ID"
        parsed = executor._fallback_parse(merge_instruction)
        
        assert parsed['action'] == 'merge'
        assert 'Sales' in parsed['sheets']
        assert 'Returns' in parsed['sheets']
    
    def test_execute_merge(self, sample_data):
        """Test merge execution"""
        executor = TaskExecutor(sample_data)
        
        instruction = {
            'action': 'merge',
            'sheets': ['Sales', 'Returns'],
            'columns': ['Order_ID'],
            'operation': 'merge Sales and Returns'
        }
        
        result = executor.execute_instruction(instruction)
        
        assert result['status'] == 'success'
        assert 'output_path' in result
        assert result['join_key'] == 'Order_ID'
        
        # Check if file was created
        assert Path(result['output_path']).exists()
    
    def test_execute_filter(self, sample_data):
        """Test filter execution"""
        executor = TaskExecutor(sample_data)
        
        instruction = {
            'action': 'filter',
            'sheets': ['Sales'],
            'operation': 'remove null values'
        }
        
        result = executor.execute_instruction(instruction)
        assert result['status'] == 'success'
    
    def test_execute_aggregate(self, sample_data):
        """Test aggregation execution"""
        executor = TaskExecutor(sample_data)
        
        instruction = {
            'action': 'aggregate',
            'sheets': ['Sales'],
            'operation': 'group by Product'
        }
        
        result = executor.execute_instruction(instruction)
        assert result['status'] == 'success'
        assert 'output_path' in result
    
    def test_execute_summary(self, sample_data):
        """Test summary execution"""
        executor = TaskExecutor(sample_data)
        
        instruction = {
            'action': 'summary',
            'sheets': ['Sales'],
            'operation': 'summarize Sales data'
        }
        
        result = executor.execute_instruction(instruction)
        assert result['status'] == 'success'
        assert 'output_path' in result
        
        # Check if summary file exists and contains expected data
        summary_path = Path(result['output_path'])
        assert summary_path.exists()
        
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        assert 'Sales' in summary_data
        assert summary_data['Sales']['shape'] == [5, 6]  # 5 rows, 6 columns

    def test_natural_language_interface(self, sample_data):
        """Test the high-level natural language interface"""
        instruction = "merge Sales and Returns tables"
        
        result = execute_natural_language(sample_data, instruction)
        
        assert 'parsed_instruction' in result
        assert 'execution_result' in result
        assert result['execution_result']['status'] == 'success'

def create_test_data():
    """Helper function to create test data for manual testing"""
    # Create more complex test data
    np.random.seed(42)
    
    sales_data = pd.DataFrame({
        'Order_ID': range(1, 101),
        'Customer_ID': np.random.randint(1, 21, 100),
        'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100),
        'Product_Name': [f'Product_{i}' for i in np.random.randint(1, 51, 100)],
        'Quantity': np.random.randint(1, 20, 100),
        'Unit_Price': np.round(np.random.uniform(10, 500, 100), 2),
        'Discount': np.round(np.random.uniform(0, 0.3, 100), 2),
        'Sales_Date': pd.date_range('2024-01-01', periods=100, freq='D')
    })
    
    # Calculate total price
    sales_data['Total_Price'] = sales_data['Quantity'] * sales_data['Unit_Price'] * (1 - sales_data['Discount'])
    
    customer_data = pd.DataFrame({
        'Customer_ID': range(1, 21),
        'Customer_Name': [f'Customer_{i}' for i in range(1, 21)],
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 20),
        'Age': np.random.randint(18, 80, 20),
        'Customer_Segment': np.random.choice(['Premium', 'Standard', 'Basic'], 20)
    })
    
    returns_data = pd.DataFrame({
        'Order_ID': np.random.choice(range(1, 101), 15, replace=False),
        'Return_Quantity': np.random.randint(1, 5, 15),
        'Return_Reason': np.random.choice(['Defective', 'Wrong Size', 'Not as Expected', 'Damaged'], 15),
        'Return_Date': pd.date_range('2024-02-01', periods=15, freq='3D')
    })
    
    # Save to Excel
    output_path = Path('data') / 'test_comprehensive.xlsx'
    output_path.parent.mkdir(exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        sales_data.to_excel(writer, sheet_name='Sales', index=False)
        customer_data.to_excel(writer, sheet_name='Customers', index=False)
        returns_data.to_excel(writer, sheet_name='Returns', index=False)
    
    print(f"Test data created: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    # Create comprehensive test data
    test_file = create_test_data()
    
    # Run manual test
    print("Running manual test...")
    try:
        executor = TaskExecutor(test_file)
        
        # Test various instructions
        test_instructions = [
            "merge Sales and Customers on Customer_ID",
            "show me total sales by Product_Category",
            "create visualization for Sales data",
            "clean the Sales data",
            "filter Sales data to remove low quantities"
        ]
        
        for instruction in test_instructions:
            print(f"\n{'='*60}")
            print(f"Testing: {instruction}")
            result = execute_natural_language(test_file, instruction)
            
            print("Parsed:", result['parsed_instruction'])
            print("Result:", result['execution_result']['message'])
            
    except Exception as e:
        print(f"Error in manual test: {e}")