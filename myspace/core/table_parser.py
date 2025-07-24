"""
Table parsing and processing utilities
"""
import pandas as pd
import json
import csv
from typing import List, Dict, Any, Union, Tuple
from io import StringIO
import os


class TableParser:
    """
    Utility class for parsing and processing various table formats
    """
    
    def __init__(self):
        """Initialize the table parser"""
        pass
    
    def load_table(self, file_path: str) -> pd.DataFrame:
        """
        Load table from various file formats
        
        Args:
            file_path: Path to the table file
            
        Returns:
            Pandas DataFrame
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def parse_table_from_string(self, table_string: str, format_type: str = "auto") -> pd.DataFrame:
        """
        Parse table from string representation
        
        Args:
            table_string: String representation of table
            format_type: Format type ('csv', 'json', 'markdown', 'auto')
            
        Returns:
            Pandas DataFrame
        """
        if format_type == "auto":
            format_type = self._detect_format(table_string)
        
        if format_type == "csv":
            return pd.read_csv(StringIO(table_string))
        elif format_type == "json":
            data = json.loads(table_string)
            return pd.DataFrame(data)
        elif format_type == "markdown":
            return self._parse_markdown_table(table_string)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _detect_format(self, table_string: str) -> str:
        """
        Automatically detect table format
        
        Args:
            table_string: String representation of table
            
        Returns:
            Detected format type
        """
        table_string = table_string.strip()
        
        # Check for JSON
        if table_string.startswith('[') or table_string.startswith('{'):
            try:
                json.loads(table_string)
                return "json"
            except:
                pass
        
        # Check for Markdown table
        if '|' in table_string and '---' in table_string:
            return "markdown"
        
        # Default to CSV
        return "csv"
    
    def _parse_markdown_table(self, markdown_string: str) -> pd.DataFrame:
        """
        Parse Markdown table format
        
        Args:
            markdown_string: Markdown table string
            
        Returns:
            Pandas DataFrame
        """
        lines = markdown_string.strip().split('\n')
        
        # Find table lines (lines containing |)
        table_lines = [line for line in lines if '|' in line]
        
        if len(table_lines) < 2:
            raise ValueError("Invalid Markdown table format")
        
        # Extract header
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        # Skip separator line and extract data rows
        data_rows = []
        for line in table_lines[2:]:  # Skip header and separator
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) == len(headers):
                data_rows.append(row)
        
        return pd.DataFrame(data_rows, columns=headers)
    
    def table_to_string(self, df: pd.DataFrame, format_type: str = "markdown") -> str:
        """
        Convert DataFrame to string representation
        
        Args:
            df: Pandas DataFrame
            format_type: Output format ('markdown', 'csv', 'json')
            
        Returns:
            String representation of table
        """
        if format_type == "markdown":
            return df.to_markdown(index=False)
        elif format_type == "csv":
            return df.to_csv(index=False)
        elif format_type == "json":
            return df.to_json(orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def chunk_table(self, df: pd.DataFrame, chunk_size: int = 10) -> List[pd.DataFrame]:
        """
        Split table into smaller chunks for better retrieval
        
        Args:
            df: Pandas DataFrame
            chunk_size: Number of rows per chunk
            
        Returns:
            List of DataFrame chunks
        """
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        return chunks
    
    def get_table_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the table
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.to_dict(),
            "has_null_values": df.isnull().any().any(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        # Add sample data
        if len(df) > 0:
            summary["sample_rows"] = df.head(3).to_dict('records')
        
        return summary
    
    def filter_table_by_columns(self, df: pd.DataFrame, relevant_columns: List[str]) -> pd.DataFrame:
        """
        Filter table to keep only relevant columns
        
        Args:
            df: Pandas DataFrame
            relevant_columns: List of column names to keep
            
        Returns:
            Filtered DataFrame
        """
        available_columns = [col for col in relevant_columns if col in df.columns]
        if not available_columns:
            return df  # Return original if no matches
        return df[available_columns]
    
    def search_table_content(self, df: pd.DataFrame, search_terms: List[str]) -> pd.DataFrame:
        """
        Search table content for specific terms
        
        Args:
            df: Pandas DataFrame
            search_terms: List of terms to search for
            
        Returns:
            Filtered DataFrame with matching rows
        """
        if not search_terms:
            return df
        
        # Convert all columns to string for searching
        df_str = df.astype(str)
        
        # Create boolean mask for rows containing any search term
        mask = pd.Series([False] * len(df))
        
        for term in search_terms:
            term_mask = df_str.apply(lambda row: row.str.contains(term, case=False, na=False).any(), axis=1)
            mask = mask | term_mask
        
        return df[mask]
    
    def normalize_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize table data for better processing
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        # Fill NaN values with empty strings
        df_normalized = df_normalized.fillna('')
        
        # Convert all columns to string type for consistent processing
        for col in df_normalized.columns:
            df_normalized[col] = df_normalized[col].astype(str)
        
        # Strip whitespace
        df_normalized = df_normalized.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        return df_normalized
