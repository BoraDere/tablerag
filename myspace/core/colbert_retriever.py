"""
ColBERT-based retrieval system for enhanced table retrieval
"""
import os
import json
import pickle
from typing import List, Dict, Any, Tuple
from ragatouille import RAGPretrainedModel
import pandas as pd
from .table_parser import TableParser
import hashlib
import warnings
warnings.filterwarnings("ignore")


class ColBERTRetriever:
    """
    Enhanced retrieval system using ColBERT for fine-grained token-level matching
    """
    
    def __init__(self, 
                 config_path: str = "config.json",
                 index_root: str = "data/indices"):
        """
        Initialize ColBERT retriever
        
        Args:
            config_path: Path to configuration file
            index_root: Root directory for storing indices
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config
        self.index_root = index_root
        self.model_name = config["retrieval"]["colbert_model"]
        self.top_k = config["retrieval"]["top_k"]
        self.rerank = config["retrieval"]["rerank"]
        self.rerank_top_k = config["retrieval"]["rerank_top_k"]
        
        # Initialize ColBERT model
        self.rag_model = RAGPretrainedModel.from_pretrained(self.model_name)
        
        # Initialize table parser
        self.table_parser = TableParser()
        
        # Storage for document metadata
        self.document_metadata = {}
        
        # Ensure indices directory exists
        os.makedirs(self.index_root, exist_ok=True)
    
    def create_index(self, 
                     table_chunks: List[Dict[str, Any]], 
                     index_name: str,
                     force_rebuild: bool = False) -> str:
        """
        Create ColBERT index from table chunks
        
        Args:
            table_chunks: List of table chunk dictionaries
            index_name: Name for the index
            force_rebuild: Whether to rebuild existing index
            
        Returns:
            Path to created index
        """
        index_path = os.path.join(self.index_root, index_name)
        
        # Check if index already exists
        if os.path.exists(index_path) and not force_rebuild:
            print(f"Index {index_name} already exists. Set force_rebuild=True to recreate.")
            return index_path
        
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
                "original_data": chunk.get("table_data", "")
            }
        
        # Create index
        self.rag_model.index(
            collection=documents,
            index_name=index_name,
            max_document_length=512,
            split_documents=True
        )
        
        # Save metadata
        metadata_path = os.path.join(index_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.document_metadata[index_name] = metadata
        
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
        if chunk.get("table_data"):
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
                 top_k: int = None,
                 rerank: bool = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant table chunks for a query
        
        Args:
            query: Search query
            index_name: Name of the index to search
            top_k: Number of results to retrieve
            rerank: Whether to apply reranking
            
        Returns:
            List of relevant chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        if rerank is None:
            rerank = self.rerank
        
        # Load metadata if not in memory
        if index_name not in self.document_metadata:
            self._load_metadata(index_name)
        
        # Search using ColBERT
        search_results = self.rag_model.search(
            query=query,
            index_name=index_name,
            k=top_k
        )
        
        # Apply reranking if enabled
        if rerank and len(search_results) > 1:
            documents = [result['content'] for result in search_results]
            reranked_results = self.rag_model.rerank(
                query=query,
                documents=documents,
                k=min(self.rerank_top_k, len(documents))
            )
            
            # Map reranked results back to original format
            reranked_search_results = []
            for reranked in reranked_results:
                # Find corresponding original result
                for original in search_results:
                    if original['content'] == reranked['content']:
                        reranked_search_results.append({
                            **original,
                            'score': reranked['score']
                        })
                        break
            search_results = reranked_search_results
        
        # Enhance results with metadata
        enhanced_results = []
        metadata = self.document_metadata.get(index_name, {})
        
        for result in search_results:
            # Extract document ID from result
            doc_id = self._extract_doc_id(result, metadata)
            
            enhanced_result = {
                "content": result["content"],
                "score": result["score"],
                "metadata": metadata.get(doc_id, {}),
                "doc_id": doc_id
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _extract_doc_id(self, result: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Extract document ID from search result
        
        Args:
            result: Search result dictionary
            metadata: Metadata dictionary
            
        Returns:
            Document ID
        """
        # Try to find matching content in metadata
        result_content = result["content"]
        
        for doc_id, meta in metadata.items():
            if result_content in meta.get("original_data", ""):
                return doc_id
        
        # Fallback: use hash of content as ID
        content_hash = hashlib.md5(result_content.encode()).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def _load_metadata(self, index_name: str):
        """
        Load metadata for an index
        
        Args:
            index_name: Name of the index
        """
        metadata_path = os.path.join(self.index_root, index_name, "metadata.pkl")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.document_metadata[index_name] = pickle.load(f)
        else:
            self.document_metadata[index_name] = {}
    
    def update_index(self, 
                     new_chunks: List[Dict[str, Any]], 
                     index_name: str):
        """
        Update existing index with new chunks
        
        Args:
            new_chunks: New table chunks to add
            index_name: Name of the index to update
        """
        # Load existing metadata
        if index_name not in self.document_metadata:
            self._load_metadata(index_name)
        
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
                "original_data": chunk.get("table_data", "")
            }
        
        # Add to existing index
        self.rag_model.add_to_index(
            new_collection=new_documents,
            index_name=index_name
        )
        
        # Update metadata
        existing_metadata.update(new_metadata)
        self.document_metadata[index_name] = existing_metadata
        
        # Save updated metadata
        metadata_path = os.path.join(self.index_root, index_name, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(existing_metadata, f)
    
    def delete_index(self, index_name: str):
        """
        Delete an index and its metadata
        
        Args:
            index_name: Name of the index to delete
        """
        index_path = os.path.join(self.index_root, index_name)
        
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)
        
        if index_name in self.document_metadata:
            del self.document_metadata[index_name]
    
    def list_indices(self) -> List[str]:
        """
        List all available indices
        
        Returns:
            List of index names
        """
        if not os.path.exists(self.index_root):
            return []
        
        return [d for d in os.listdir(self.index_root) 
                if os.path.isdir(os.path.join(self.index_root, d))]
