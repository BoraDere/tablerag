"""
Main API interface for Agent TableRAG
Simple and powerful table-based retrieval for AI chatbots
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime

from core.llm_client import LLMClient
from core.table_processor import TableProcessor
from core.faiss_retriever import FAISSRetriever
from core.table_parser import TableParser
from utils.text_processing import format_response_text, highlight_matches


class AgentTableRAG:
    """
    Main interface for Agent TableRAG system
    
    This class provides a simple API for AI agents to:
    1. Add table knowledge with explanations
    2. Query the table knowledge intelligently
    3. Get enhanced answers using advanced retrieval
    """
    
    def __init__(self, 
                 agent_explanation: str,
                 config_path: str = "config.json",
                 data_dir: str = "data"):
        """
        Initialize the Agent TableRAG system
        
        Args:
            agent_explanation: Explanation of what the agent does and how it should use tables
            config_path: Path to configuration file
            data_dir: Directory for storing processed data and indices
        """
        self.agent_explanation = agent_explanation
        self.config_path = config_path
        self.data_dir = data_dir
        
        # Initialize components
        self.llm_client = LLMClient(config_path)
        self.table_processor = TableProcessor(config_path)
        self.retriever = FAISSRetriever(config_path, os.path.join(data_dir, "indices"))
        self.table_parser = TableParser()
        
        # Storage for table knowledge
        self.table_knowledge = []
        self.index_name = self._generate_index_name()
        
        # Create data directories
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "indices"), exist_ok=True)
        
        print(f"AgentTableRAG initialized for: {agent_explanation}")
    
    def add_table_knowledge(self, 
                           table_data: Union[str, pd.DataFrame, Dict[str, Any]],
                           knowledge_explanation: str,
                           table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Add table knowledge to the agent
        
        Args:
            table_data: Table data (file path, DataFrame, or dict)
            knowledge_explanation: Explanation of how the agent should use this table
            table_name: Optional name for the table (auto-generated if not provided)
            
        Returns:
            Dictionary with processing results and statistics
        """
        # Generate table name if not provided
        if table_name is None:
            table_name = f"table_{len(self.table_knowledge) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Processing table knowledge: {table_name}")
        
        # Process the table
        try:
            chunks = self.table_processor.process_table(
                table_data=table_data,
                table_name=table_name,
                knowledge_explanation=knowledge_explanation
            )
            
            # Store knowledge metadata
            knowledge_entry = {
                "table_name": table_name,
                "knowledge_explanation": knowledge_explanation,
                "chunks": chunks,
                "added_at": datetime.now().isoformat(),
                "num_chunks": len(chunks)
            }
            
            self.table_knowledge.append(knowledge_entry)
            
            # Update the retrieval index
            self._update_index(chunks)
            
            # Generate statistics
            stats = self._generate_processing_stats(chunks)
            
            print(f"Successfully added table knowledge: {table_name}")
            print(f"Generated {len(chunks)} chunks for retrieval")
            
            return {
                "status": "success",
                "table_name": table_name,
                "num_chunks": len(chunks),
                "stats": stats
            }
            
        except Exception as e:
            error_msg = f"Failed to process table {table_name}: {str(e)}"
            print(error_msg)
            return {
                "status": "error",
                "table_name": table_name,
                "error": error_msg
            }
    
    def query(self, 
              user_question: str,
              top_k: Optional[int] = None,
              include_sources: bool = True,
              highlight_matches: bool = True) -> Dict[str, Any]:
        """
        Query the agent's table knowledge
        
        Args:
            user_question: The user's question
            top_k: Number of top results to retrieve (uses config default if None)
            include_sources: Whether to include source information
            highlight_matches: Whether to highlight query matches in results
            
        Returns:
            Dictionary with answer and optional source information
        """
        if not self.table_knowledge:
            return {
                "answer": "I don't have any table knowledge yet. Please add some table data first.",
                "confidence": 0.0,
                "sources": []
            }
        
        print(f"Processing query: {user_question}")
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retriever.retrieve(
                query=user_question,
                index_name=self.index_name,
                top_k=top_k
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information in my table knowledge to answer your question.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Generate answer using LLM
            answer = self._generate_answer(user_question, relevant_chunks)
            
            # Calculate confidence based on retrieval scores
            confidence = self._calculate_confidence(relevant_chunks)
            
            # Prepare response
            response = {
                "answer": answer,
                "confidence": confidence,
                "num_sources": len(relevant_chunks)
            }
            
            # Add source information if requested
            if include_sources:
                sources = self._format_sources(relevant_chunks, user_question, highlight_matches)
                response["sources"] = sources
            
            print(f"Generated answer with {len(relevant_chunks)} sources (confidence: {confidence:.2f})")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "answer": "I encountered an error while processing your question. Please try rephrasing it.",
                "confidence": 0.0,
                "sources": [],
                "error": error_msg
            }
    
    def get_table_summary(self) -> Dict[str, Any]:
        """
        Get summary of all table knowledge in the system
        
        Returns:
            Dictionary with summary information
        """
        if not self.table_knowledge:
            return {
                "num_tables": 0,
                "total_chunks": 0,
                "tables": []
            }
        
        total_chunks = sum(entry["num_chunks"] for entry in self.table_knowledge)
        
        table_summaries = []
        for entry in self.table_knowledge:
            table_summaries.append({
                "table_name": entry["table_name"],
                "knowledge_explanation": entry["knowledge_explanation"],
                "num_chunks": entry["num_chunks"],
                "added_at": entry["added_at"]
            })
        
        return {
            "num_tables": len(self.table_knowledge),
            "total_chunks": total_chunks,
            "agent_explanation": self.agent_explanation,
            "tables": table_summaries
        }
    
    def update_agent_explanation(self, new_explanation: str):
        """
        Update the agent explanation
        
        Args:
            new_explanation: New explanation for the agent
        """
        self.agent_explanation = new_explanation
        print(f"Updated agent explanation: {new_explanation}")
    
    def remove_table_knowledge(self, table_name: str) -> bool:
        """
        Remove specific table knowledge
        
        Args:
            table_name: Name of the table to remove
            
        Returns:
            True if removed successfully, False if not found
        """
        for i, entry in enumerate(self.table_knowledge):
            if entry["table_name"] == table_name:
                del self.table_knowledge[i]
                print(f"Removed table knowledge: {table_name}")
                
                # Rebuild index without this table
                self._rebuild_index()
                return True
        
        print(f"Table knowledge not found: {table_name}")
        return False
    
    def clear_all_knowledge(self):
        """
        Clear all table knowledge
        """
        self.table_knowledge = []
        
        # Delete the index
        try:
            self.retriever.delete_index(self.index_name)
        except Exception as e:
            print(f"Warning: Could not delete index {self.index_name}: {e}")
        
        print("Cleared all table knowledge")
    
    def _generate_index_name(self) -> str:
        """Generate unique index name for this agent"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"agent_index_{timestamp}"
    
    def _update_index(self, new_chunks: List[Dict[str, Any]]):
        """Update the retrieval index with new chunks"""
        try:
            # Check if index exists
            existing_indices = self.retriever.list_indices()
            
            if self.index_name in existing_indices:
                # Update existing index
                self.retriever.update_index(new_chunks, self.index_name)
            else:
                # Create new index
                self.retriever.create_index(new_chunks, self.index_name)
                
        except Exception as e:
            print(f"Warning: Failed to update index: {e}")
    
    def _rebuild_index(self):
        """Rebuild the entire index from current knowledge"""
        if not self.table_knowledge:
            return
        
        # Collect all chunks
        all_chunks = []
        for entry in self.table_knowledge:
            all_chunks.extend(entry["chunks"])
        
        # Recreate index
        try:
            self.retriever.delete_index(self.index_name)
        except:
            pass
        
        self.retriever.create_index(all_chunks, self.index_name, force_rebuild=True)
    
    def _generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM and retrieved chunks"""
        
        # Prepare context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk.get("metadata", {})
            table_name = metadata.get("table_name", f"Table {i+1}")
            explanation = metadata.get("explanation", "")
            
            # Use markdown format for better LLM understanding
            chunk_content = chunk.get("content", "")
            
            context_part = f"**{table_name}**\n"
            if explanation:
                context_part += f"Context: {explanation}\n"
            context_part += f"{chunk_content}\n"
            
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        # Create system prompt
        system_prompt = f"""You are an AI assistant with the following role:
{self.agent_explanation}

You have access to table data that you should use to answer questions accurately and helpfully.
When answering:
1. Use the provided table data to give accurate, specific answers
2. If you find relevant information, be precise and cite specific details
3. If the information isn't in the tables, clearly state that
4. Format your response in a clear, conversational manner
5. Focus on being helpful and accurate"""
        
        # Create user prompt
        user_prompt = f"""Based on the following table data, please answer this question:

Question: {question}

Table Data:
{context}

Please provide a clear, accurate answer based on the table information provided."""
        
        # Generate response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        answer = self.llm_client.generate_text(messages, temperature=0.1)
        
        # Format the response
        formatted_answer = format_response_text(answer)
        
        return formatted_answer
    
    def _calculate_confidence(self, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not relevant_chunks:
            return 0.0
        
        # Use the average score of top chunks, weighted by position
        scores = [chunk.get("score", 0.0) for chunk in relevant_chunks]
        
        if not scores:
            return 0.5  # Default confidence
        
        # Weight scores by position (first results are more important)
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)
        
        confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Normalize to 0-1 range and apply some smoothing
        confidence = min(max(confidence, 0.0), 1.0)
        
        return confidence
    
    def _format_sources(self, 
                       relevant_chunks: List[Dict[str, Any]], 
                       query: str,
                       highlight: bool = True) -> List[Dict[str, Any]]:
        """Format source information for the response"""
        sources = []
        
        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")
            
            # Apply highlighting if requested
            if highlight:
                content = highlight_matches(content, query)
            
            source = {
                "source_id": i + 1,
                "table_name": metadata.get("table_name", f"Table {i+1}"),
                "explanation": metadata.get("explanation", ""),
                "relevance_score": chunk.get("score", 0.0),
                "content_preview": content[:300] + "..." if len(content) > 300 else content
            }
            
            sources.append(source)
        
        return sources
    
    def _generate_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about processed chunks"""
        if not chunks:
            return {}
        
        total_rows = sum(
            chunk.get("chunk_info", {}).get("chunk_rows", 0) 
            for chunk in chunks
        )
        
        # Get sample of column names
        sample_chunk = chunks[0] if chunks else {}
        sample_df = sample_chunk.get("table_data")
        columns = list(sample_df.columns) if isinstance(sample_df, pd.DataFrame) else []
        
        return {
            "total_chunks": len(chunks),
            "total_rows_processed": total_rows,
            "avg_rows_per_chunk": total_rows / len(chunks) if chunks else 0,
            "sample_columns": columns[:10],  # First 10 columns
            "num_columns": len(columns)
        }
