"""
FAISS-based retrieval system for enhanced table retrieval
Alternative to ColBERT that uses OpenAI embeddings with FAISS
"""
import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import pandas as pd
from core.table_parser import TableParser
from core.llm_client import LLMClient
import hashlib
from datetime import datetime


class FAISSRetriever:
    """
    FAISS-based retrieval system using OpenAI embeddings
    """
    
    def __init__(self, 
                 config_path: str = "config.json",
                 index_root: str = "data/indices"):
        """
        Initialize FAISS retriever
        
        Args:
            config_path: Path to configuration file
            index_root: Root directory for storing indices
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config
        self.index_root = index_root
        self.top_k = config["retrieval"]["top_k"]
        
        # Initialize LLM client for embeddings
        self.llm_client = LLMClient(config_path)
        
        # Initialize table parser
        self.table_parser = TableParser()
        
        # Storage for indices and metadata
        self.indices = {}
        self.document_metadata = {}
        
        # Ensure indices directory exists
        os.makedirs(self.index_root, exist_ok=True)
    
    def create_index(self, 
                     table_chunks: List[Dict[str, Any]], 
                     index_name: str,
                     force_rebuild: bool = False) -> str:
        """
        Create FAISS index from table chunks
        
        Args:
            table_chunks: List of table chunk dictionaries
            index_name: Name for the index
            force_rebuild: Whether to rebuild existing index
            
        Returns:
            Path to created index
        """
        index_path = os.path.join(self.index_root, f"{index_name}.faiss")
        metadata_path = os.path.join(self.index_root, f"{index_name}.metadata")
        
        # Check if index already exists
        if os.path.exists(index_path) and not force_rebuild:
            print(f"Index {index_name} already exists. Set force_rebuild=True to recreate.")
            self._load_index(index_name)
            return index_path
        
        print(f"Creating FAISS index: {index_name}")
        
        # Prepare documents for indexing
        documents = []
        metadata = {}
        
        for i, chunk in enumerate(table_chunks):
            doc_id = f"chunk_{i}"
            
            # Create searchable text from table chunk
            searchable_text = self._create_searchable_text(chunk)
            documents.append(searchable_text)
            
            # Store metadata
            metadata[doc_id] = {
                "chunk_id": i,
                "table_name": chunk.get("table_name", "unknown"),
                "explanation": chunk.get("explanation", ""),
                "chunk_info": chunk.get("chunk_info", {}),
                "original_data": chunk.get("table_data", ""),
                "searchable_text": searchable_text
            }
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.llm_client.generate_embeddings_batch(documents)
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        # Save index and metadata
        faiss.write_index(index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Store in memory
        self.indices[index_name] = index
        self.document_metadata[index_name] = metadata
        
        print(f"Created index with {len(documents)} documents")
        
        return index_path
    
    def _create_searchable_text(self, chunk: Dict[str, Any]) -> str:
        """
        Create searchable text representation from table chunk
        
        Args:
            chunk: Table chunk dictionary
            
        Returns:
            Searchable text string
        """
        parts = []
        
        # Add table name if available
        if chunk.get("table_name"):
            parts.append(f"Table: {chunk['table_name']}")
        
        # Add explanation if available
        if chunk.get("explanation"):
            parts.append(f"Context: {chunk['explanation']}")
        
        # Add table data
        if chunk.get("markdown_data"):
            parts.append(chunk["markdown_data"])
        elif chunk.get("table_data"):
            if isinstance(chunk["table_data"], str):
                parts.append(chunk["table_data"])
            elif isinstance(chunk["table_data"], pd.DataFrame):
                # Convert DataFrame to searchable format
                table_text = self.table_parser.table_to_string(chunk["table_data"], "markdown")
                parts.append(table_text)
        
        # Add chunk-specific context
        if chunk.get("chunk_info"):
            chunk_info = chunk["chunk_info"]
            if chunk_info.get("summary"):
                parts.append(f"Summary: {chunk_info['summary']}")
        
        return "\n\n".join(parts)
    
    def retrieve(self, 
                 query: str, 
                 index_name: str,
                 top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant table chunks for a query
        
        Args:
            query: Search query
            index_name: Name of the index to search
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Load index if not in memory
        if index_name not in self.indices:
            self._load_index(index_name)
        
        if index_name not in self.indices:
            print(f"Index {index_name} not found")
            return []
        
        # Generate query embedding
        query_embedding = self.llm_client.generate_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search index
        index = self.indices[index_name]
        scores, indices = index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        metadata = self.document_metadata.get(index_name, {})
        
        for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
            if doc_idx == -1:  # No more results
                break
                
            doc_id = f"chunk_{doc_idx}"
            doc_metadata = metadata.get(doc_id, {})
            
            result = {
                "content": doc_metadata.get("searchable_text", ""),
                "score": float(score),
                "metadata": {
                    "table_name": doc_metadata.get("table_name", "unknown"),
                    "explanation": doc_metadata.get("explanation", ""),
                    "chunk_info": doc_metadata.get("chunk_info", {}),
                    "chunk_id": doc_metadata.get("chunk_id", doc_idx)
                },
                "doc_id": doc_id
            }
            
            results.append(result)
        
        return results
    
    def _load_index(self, index_name: str):
        """
        Load index and metadata from disk
        
        Args:
            index_name: Name of the index to load
        """
        index_path = os.path.join(self.index_root, f"{index_name}.faiss")
        metadata_path = os.path.join(self.index_root, f"{index_name}.metadata")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load index
            index = faiss.read_index(index_path)
            self.indices[index_name] = index
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.document_metadata[index_name] = pickle.load(f)
            
            print(f"Loaded index: {index_name}")
        else:
            print(f"Index files not found for: {index_name}")
    
    def update_index(self, 
                     new_chunks: List[Dict[str, Any]], 
                     index_name: str):
        """
        Update existing index with new chunks
        
        Args:
            new_chunks: New table chunks to add
            index_name: Name of the index to update
        """
        # Load existing index
        if index_name not in self.indices:
            self._load_index(index_name)
        
        if index_name not in self.indices:
            print(f"Index {index_name} not found. Creating new index.")
            self.create_index(new_chunks, index_name)
            return
        
        existing_metadata = self.document_metadata.get(index_name, {})
        
        # Determine starting chunk ID
        existing_chunk_ids = [
            meta.get("chunk_id", 0) 
            for meta in existing_metadata.values() 
            if isinstance(meta.get("chunk_id"), int)
        ]
        next_chunk_id = max(existing_chunk_ids, default=-1) + 1
        
        # Prepare new documents
        new_documents = []
        new_metadata = {}
        
        for i, chunk in enumerate(new_chunks):
            chunk_id = next_chunk_id + i
            doc_id = f"chunk_{chunk_id}"
            
            searchable_text = self._create_searchable_text(chunk)
            new_documents.append(searchable_text)
            
            new_metadata[doc_id] = {
                "chunk_id": chunk_id,
                "table_name": chunk.get("table_name", "unknown"),
                "explanation": chunk.get("explanation", ""),
                "chunk_info": chunk.get("chunk_info", {}),
                "original_data": chunk.get("table_data", ""),
                "searchable_text": searchable_text
            }
        
        # Generate embeddings for new documents
        new_embeddings = self.llm_client.generate_embeddings_batch(new_documents)
        embeddings_array = np.array(new_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Add to existing index
        index = self.indices[index_name]
        index.add(embeddings_array)
        
        # Update metadata
        existing_metadata.update(new_metadata)
        self.document_metadata[index_name] = existing_metadata
        
        # Save updated index and metadata
        index_path = os.path.join(self.index_root, f"{index_name}.faiss")
        metadata_path = os.path.join(self.index_root, f"{index_name}.metadata")
        
        faiss.write_index(index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(existing_metadata, f)
        
        print(f"Updated index {index_name} with {len(new_chunks)} new chunks")
    
    def delete_index(self, index_name: str):
        """
        Delete an index and its metadata
        
        Args:
            index_name: Name of the index to delete
        """
        index_path = os.path.join(self.index_root, f"{index_name}.faiss")
        metadata_path = os.path.join(self.index_root, f"{index_name}.metadata")
        
        # Remove files
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove from memory
        if index_name in self.indices:
            del self.indices[index_name]
        if index_name in self.document_metadata:
            del self.document_metadata[index_name]
        
        print(f"Deleted index: {index_name}")
    
    def list_indices(self) -> List[str]:
        """
        List all available indices
        
        Returns:
            List of index names
        """
        if not os.path.exists(self.index_root):
            return []
        
        faiss_files = [f for f in os.listdir(self.index_root) if f.endswith('.faiss')]
        return [os.path.splitext(f)[0] for f in faiss_files]
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about an index
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary with index statistics
        """
        if index_name not in self.indices:
            self._load_index(index_name)
        
        if index_name not in self.indices:
            return {"error": "Index not found"}
        
        index = self.indices[index_name]
        metadata = self.document_metadata.get(index_name, {})
        
        # Count tables
        table_names = set()
        for meta in metadata.values():
            table_names.add(meta.get("table_name", "unknown"))
        
        return {
            "index_name": index_name,
            "total_documents": index.ntotal,
            "dimension": index.d,
            "num_tables": len(table_names),
            "table_names": list(table_names),
            "created_at": datetime.now().isoformat()
        }
