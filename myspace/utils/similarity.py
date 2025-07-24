"""
Similarity calculation utilities
"""
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    # Convert to numpy arrays
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    return float(similarity)


def calculate_similarity_matrix(embeddings1: List[List[float]], 
                              embeddings2: List[List[float]]) -> np.ndarray:
    """
    Calculate similarity matrix between two sets of embeddings
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Similarity matrix
    """
    emb1_array = np.array(embeddings1)
    emb2_array = np.array(embeddings2)
    
    return cosine_similarity(emb1_array, emb2_array)


def select_top_k_items(scores: List[float], k: int) -> List[int]:
    """
    Select indices of top-k items based on scores
    
    Args:
        scores: List of scores
        k: Number of top items to select
        
    Returns:
        List of indices of top-k items
    """
    # Convert to numpy array for easier manipulation
    scores_array = np.array(scores)
    
    # Get indices of top-k scores
    top_k_indices = np.argsort(scores_array)[-k:][::-1]
    
    return top_k_indices.tolist()


def rank_by_similarity(query_embedding: List[float], 
                      candidate_embeddings: List[List[float]],
                      top_k: int = None) -> List[Tuple[int, float]]:
    """
    Rank candidates by similarity to query
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embeddings
        top_k: Number of top results to return (None for all)
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    
    for i, candidate_emb in enumerate(candidate_embeddings):
        similarity = calculate_similarity(query_embedding, candidate_emb)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k if specified
    if top_k is not None:
        similarities = similarities[:top_k]
    
    return similarities


def calculate_centroid(embeddings: List[List[float]]) -> List[float]:
    """
    Calculate centroid of a set of embeddings
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Centroid embedding vector
    """
    if not embeddings:
        return []
    
    embeddings_array = np.array(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    
    return centroid.tolist()


def find_diverse_subset(embeddings: List[List[float]], 
                       k: int, 
                       diversity_threshold: float = 0.8) -> List[int]:
    """
    Find a diverse subset of embeddings using similarity-based filtering
    
    Args:
        embeddings: List of embedding vectors
        k: Number of embeddings to select
        diversity_threshold: Minimum similarity threshold for diversity
        
    Returns:
        List of indices of diverse embeddings
    """
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    
    selected_indices = []
    remaining_indices = list(range(len(embeddings)))
    
    # Start with a random embedding
    selected_indices.append(remaining_indices.pop(0))
    
    while len(selected_indices) < k and remaining_indices:
        best_idx = None
        best_min_similarity = -1
        
        # Find the embedding that is most dissimilar to all selected ones
        for idx in remaining_indices:
            min_similarity = float('inf')
            
            for selected_idx in selected_indices:
                similarity = calculate_similarity(embeddings[idx], embeddings[selected_idx])
                min_similarity = min(min_similarity, similarity)
            
            if min_similarity > best_min_similarity:
                best_min_similarity = min_similarity
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    return selected_indices


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


def weighted_similarity(query_embedding: List[float],
                       candidate_embeddings: List[List[float]],
                       weights: List[float]) -> List[float]:
    """
    Calculate weighted similarity scores
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embeddings
        weights: Weights for each candidate
        
    Returns:
        List of weighted similarity scores
    """
    if len(candidate_embeddings) != len(weights):
        raise ValueError("Number of embeddings must match number of weights")
    
    similarities = []
    for i, candidate_emb in enumerate(candidate_embeddings):
        similarity = calculate_similarity(query_embedding, candidate_emb)
        weighted_sim = similarity * weights[i]
        similarities.append(weighted_sim)
    
    return similarities
