"""
Table processing pipeline with filtering and clarification
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from core.llm_client import LLMClient
from core.table_parser import TableParser
from utils.similarity import calculate_similarity, select_top_k_items


class TableProcessor:
    """
    Processes tables with intelligent filtering and clarification
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize table processor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.llm_client = LLMClient(config_path)
        self.table_parser = TableParser()
        
        # Processing settings
        self.enable_filtering = self.config["table_processing"]["enable_filtering"]
        self.enable_clarification = self.config["table_processing"]["enable_clarification"]
        self.max_table_rows = self.config["table_processing"]["max_table_rows"]
        self.chunk_size = self.config["table_processing"]["chunk_size"]
    
    def process_table(self, 
                     table_data: str | pd.DataFrame,
                     table_name: str,
                     knowledge_explanation: str,
                     query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a table with filtering and clarification
        
        Args:
            table_data: Table data (file path, string, or DataFrame)
            table_name: Name/identifier for the table
            knowledge_explanation: Explanation of how to use this table
            query: Optional query for targeted filtering
            
        Returns:
            List of processed table chunks
        """
        # Load and parse table
        if isinstance(table_data, str):
            if os.path.exists(table_data):
                df = self.table_parser.load_table(table_data)
            else:
                df = self.table_parser.parse_table_from_string(table_data)
        elif isinstance(table_data, pd.DataFrame):
            df = table_data.copy()
        else:
            raise ValueError("table_data must be a file path, string, or DataFrame")
        
        # Normalize the table
        df = self.table_parser.normalize_table_data(df)
        
        # Apply filtering if enabled
        if self.enable_filtering and query:
            df = self._filter_table(df, query)
        
        # Apply clarification if enabled
        if self.enable_clarification:
            df = self._clarify_table(df, knowledge_explanation)
        
        # Chunk the table
        chunks = self._create_table_chunks(df, table_name, knowledge_explanation)
        
        return chunks
    
    def _filter_table(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Filter table based on query relevance
        
        Args:
            df: Input DataFrame
            query: Query for filtering
            
        Returns:
            Filtered DataFrame
        """
        # If table is small enough, don't filter
        if len(df) <= self.max_table_rows:
            return df
        
        # Use semantic filtering for large tables
        filtered_df = self._semantic_filter(df, query)
        
        # If still too large, use LLM filtering
        if len(filtered_df) > self.max_table_rows:
            filtered_df = self._llm_filter(filtered_df, query)
        
        return filtered_df
    
    def _semantic_filter(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Apply semantic filtering using embeddings
        
        Args:
            df: Input DataFrame
            query: Query for filtering
            
        Returns:
            Semantically filtered DataFrame
        """
        # Generate query embedding
        query_embedding = self.llm_client.generate_embedding(query)
        
        # Create text representations of rows
        row_texts = []
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            row_texts.append(row_text)
        
        # Generate embeddings for rows (in batches)
        row_embeddings = self.llm_client.generate_embeddings_batch(row_texts)
        
        # Calculate similarities
        similarities = [
            calculate_similarity(query_embedding, row_emb) 
            for row_emb in row_embeddings
        ]
        
        # Select top-k most similar rows
        target_rows = min(self.max_table_rows, len(df))
        top_indices = select_top_k_items(similarities, target_rows)
        
        # Keep header and top rows
        filtered_df = df.iloc[top_indices].copy()
        
        return filtered_df
    
    def _llm_filter(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Apply LLM-based intelligent filtering
        
        Args:
            df: Input DataFrame
            query: Query for filtering
            
        Returns:
            LLM-filtered DataFrame
        """
        # Convert table to string
        table_string = self.table_parser.table_to_string(df, "markdown")
        
        # Use LLM to filter
        filtered_table_string = self.llm_client.filter_table_content(
            table_string, query, self.max_table_rows
        )
        
        try:
            # Parse the filtered table back
            filtered_df = self.table_parser.parse_table_from_string(
                filtered_table_string, "markdown"
            )
            return filtered_df
        except Exception as e:
            print(f"Warning: LLM filtering failed ({e}). Returning truncated table.")
            return df.head(self.max_table_rows)
    
    def _clarify_table(self, df: pd.DataFrame, knowledge_explanation: str) -> pd.DataFrame:
        """
        Apply clarification to make table content more understandable
        
        Args:
            df: Input DataFrame
            knowledge_explanation: Context about the table
            
        Returns:
            Clarified DataFrame
        """
        # Convert table to string
        table_string = self.table_parser.table_to_string(df, "markdown")
        
        # Add context to the clarification prompt
        contextualized_table = f"""
Context: {knowledge_explanation}

Table:
{table_string}
"""
        
        # Use LLM to clarify
        clarified_table_string = self.llm_client.clarify_table_terms(contextualized_table)
        
        try:
            # Parse the clarified table back
            clarified_df = self.table_parser.parse_table_from_string(
                clarified_table_string, "markdown"
            )
            return clarified_df
        except Exception as e:
            print(f"Warning: Table clarification failed ({e}). Returning original table.")
            return df
    
    def _create_table_chunks(self, 
                           df: pd.DataFrame, 
                           table_name: str, 
                           knowledge_explanation: str) -> List[Dict[str, Any]]:
        """
        Create chunks from processed table
        
        Args:
            df: Processed DataFrame
            table_name: Name of the table
            knowledge_explanation: Knowledge explanation
            
        Returns:
            List of table chunks
        """
        chunks = []
        
        # Use chunk_size directly as number of rows per chunk
        chunk_size_rows = self.chunk_size
        
        # Create chunks
        df_chunks = self.table_parser.chunk_table(df, chunk_size_rows)
        
        for i, chunk_df in enumerate(df_chunks):
            # Calculate actual row range
            start_row = i * chunk_size_rows
            end_row = min(start_row + len(chunk_df) - 1, len(df) - 1)
            
            # Create chunk metadata
            chunk_info = {
                "chunk_index": i,
                "total_chunks": len(df_chunks),
                "chunk_rows": len(chunk_df),
                "row_range": f"{start_row}-{end_row}",
                "summary": self._generate_chunk_summary(chunk_df)
            }
            
            # Convert chunk to different formats
            chunk_data = {
                "table_name": table_name,
                "explanation": knowledge_explanation,
                "chunk_info": chunk_info,
                "table_data": chunk_df,
                "markdown_data": self.table_parser.table_to_string(chunk_df, "markdown"),
                "csv_data": self.table_parser.table_to_string(chunk_df, "csv"),
                "json_data": self.table_parser.table_to_string(chunk_df, "json")
            }
            
            chunks.append(chunk_data)
        
        return chunks
    
    def _generate_chunk_summary(self, chunk_df: pd.DataFrame) -> str:
        """
        Generate a summary for a table chunk
        
        Args:
            chunk_df: DataFrame chunk
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"{len(chunk_df)} rows, {len(chunk_df.columns)} columns")
        
        # Column names
        summary_parts.append(f"Columns: {', '.join(chunk_df.columns[:5])}")
        if len(chunk_df.columns) > 5:
            summary_parts[-1] += f" and {len(chunk_df.columns) - 5} more"
        
        # Sample values for key columns
        for col in chunk_df.columns[:3]:
            unique_values = chunk_df[col].unique()[:3]
            if len(unique_values) > 0:
                values_str = ", ".join([str(v) for v in unique_values])
                summary_parts.append(f"{col}: {values_str}")
                if len(chunk_df[col].unique()) > 3:
                    summary_parts[-1] += "..."
        
        return "; ".join(summary_parts)
    
    def process_multiple_tables(self, 
                              table_configs: List[Dict[str, Any]],
                              global_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple tables for an agent
        
        Args:
            table_configs: List of table configurations with data, name, and explanation
            global_query: Optional global query for filtering all tables
            
        Returns:
            List of all processed chunks from all tables
        """
        all_chunks = []
        
        for config in table_configs:
            table_data = config["table_data"]
            table_name = config["table_name"]
            knowledge_explanation = config["knowledge_explanation"]
            
            # Process individual table
            chunks = self.process_table(
                table_data=table_data,
                table_name=table_name,
                knowledge_explanation=knowledge_explanation,
                query=global_query
            )
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def update_table_processing(self, 
                              chunks: List[Dict[str, Any]], 
                              new_query: str) -> List[Dict[str, Any]]:
        """
        Re-process existing chunks with a new query
        
        Args:
            chunks: Existing table chunks
            new_query: New query for re-filtering
            
        Returns:
            Re-processed chunks
        """
        updated_chunks = []
        
        for chunk in chunks:
            # Extract original data
            df = chunk["table_data"]
            table_name = chunk["table_name"]
            explanation = chunk["explanation"]
            
            # Re-process with new query
            new_chunks = self.process_table(
                table_data=df,
                table_name=table_name,
                knowledge_explanation=explanation,
                query=new_query
            )
            
            updated_chunks.extend(new_chunks)
        
        return updated_chunks
