"""
Hybrid retrieval module combining dense and sparse retrievers using Reciprocal Rank Fusion
"""
from typing import List, Dict, Any
import numpy as np
import json

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.embedding.code_embedder import CodeEmbeddingGenerator

class HybridRetriever:
    """
    Hybrid retrieval system that combines dense and sparse retrievers
    using Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self, 
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        code_embedder: CodeEmbeddingGenerator,
        k1: float = 60.0  # RRF constant
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            dense_retriever: The dense retriever instance
            sparse_retriever: The sparse retriever instance
            code_embedder: The code embedder instance
            k1: The RRF constant (default: 60.0)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.code_embedder = code_embedder
        self.k1 = k1
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code snippets using both dense and sparse retrievers,
        and combine results using Reciprocal Rank Fusion
        
        Args:
            query: The query string
            k: The number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing the code snippets, file paths, and scores
        """
        # Generate query embedding for dense retrieval
        query_embedding = self.code_embedder.generate_embedding(query)
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query_embedding, k=k*2)
        sparse_results = self.sparse_retriever.search(query, k=k*2)
        
        # Combine results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, k)
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combine results from dense and sparse retrievers using Reciprocal Rank Fusion
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            k: The number of results to return
            
        Returns:
            List of dictionaries containing the code snippets, file paths, and scores
        """
        # Create a dictionary to store RRF scores for each document
        rrf_scores = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = f"{result['file_path']}|{result['code']}"
            rrf_score = 1.0 / (rank + self.k1)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "code": result["code"],
                    "file_path": result["file_path"],
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                    "rrf_score": rrf_score,
                    "source": "dense"
                }
            else:
                rrf_scores[doc_id]["dense_score"] = result["score"]
                rrf_scores[doc_id]["rrf_score"] += rrf_score
                rrf_scores[doc_id]["source"] = "both"
                
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = f"{result['file_path']}|{result['code']}"
            rrf_score = 1.0 / (rank + self.k1)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "code": result["code"],
                    "file_path": result["file_path"],
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                    "rrf_score": rrf_score,
                    "source": "sparse"
                }
            else:
                rrf_scores[doc_id]["sparse_score"] = result["score"]
                rrf_scores[doc_id]["rrf_score"] += rrf_score
                rrf_scores[doc_id]["source"] = "both"
                
        # Sort results by RRF score
        sorted_results = sorted(
            rrf_scores.values(), 
            key=lambda x: x["rrf_score"], 
            reverse=True
        )
        
        # Return top k results
        return sorted_results[:k]
    
    def add_code_snippets(self, code_snippets: List[str], file_paths: List[str] = None):
        """
        Add code snippets to both dense and sparse retrievers
        
        Args:
            code_snippets: List of code snippets to add
            file_paths: List of file paths corresponding to the snippets (optional)
        """
        # Generate embeddings for dense retriever
        embeddings = self.code_embedder.generate_embeddings(code_snippets)
        
        # Add to dense retriever
        self.dense_retriever.add_embeddings(embeddings, code_snippets, file_paths)
        
        # Add to sparse retriever
        self.sparse_retriever.add_documents(code_snippets, file_paths)
        
        print(f"Added {len(code_snippets)} code snippets to both retrievers")