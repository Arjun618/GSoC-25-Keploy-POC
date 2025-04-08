"""
Dense retrieval module using FAISS for efficient vector similarity search
"""
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

class DenseRetriever:
    """
    Retrieves code snippets using dense vector embeddings with FAISS
    """
    
    def __init__(self, dimension: int = 768):
        """
        Initialize the dense retriever with FAISS index
        
        Args:
            dimension: Dimensionality of the embeddings (default: 768 for CodeBERT)
        """
        self.dimension = dimension
        # Use L2 distance for similarity search
        self.index = faiss.IndexFlatL2(dimension)
        # For larger datasets, use IndexIVFFlat for better performance
        # self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100)
        self.code_snippets = []
        self.file_paths = []
        
    def add_embeddings(self, embeddings: np.ndarray, code_snippets: List[str], file_paths: List[str] = None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: The embeddings to add to the index
            code_snippets: The corresponding code snippets
            file_paths: The file paths corresponding to the snippets (optional)
        """
        if embeddings.shape[0] != len(code_snippets):
            raise ValueError("Number of embeddings must match number of code snippets")
            
        if file_paths and len(file_paths) != len(code_snippets):
            raise ValueError("Number of file paths must match number of code snippets")
            
        # Add embeddings to the index
        self.index.add(embeddings)
        
        # Store the corresponding code snippets and file paths
        self.code_snippets.extend(code_snippets)
        if file_paths:
            self.file_paths.extend(file_paths)
        else:
            self.file_paths.extend([""] * len(code_snippets))
            
        print(f"Added {len(code_snippets)} code snippets to the index")
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for the k nearest neighbors of the query embedding
        
        Args:
            query_embedding: The query embedding
            k: The number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing the code snippets, file paths, and distances
        """
        if len(self.code_snippets) == 0:
            return []
            
        # Ensure the query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search for the k nearest neighbors
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding code snippets and file paths
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.code_snippets):
                results.append({
                    "code": self.code_snippets[idx],
                    "file_path": self.file_paths[idx],
                    "distance": distances[0][i],
                    "score": 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                })
                
        return results
        
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save the index and metadata to files
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata (code snippets and file paths)
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save the metadata
        metadata = {
            "code_snippets": self.code_snippets,
            "file_paths": self.file_paths
        }
        np.save(metadata_path, metadata)
        
        print(f"Saved index to {index_path} and metadata to {metadata_path}")
        
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load the index and metadata from files
        
        Args:
            index_path: Path to load the FAISS index from
            metadata_path: Path to load the metadata from
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the metadata
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.code_snippets = metadata["code_snippets"]
        self.file_paths = metadata["file_paths"]
        
        print(f"Loaded index from {index_path} and metadata from {metadata_path}")
        print(f"Index contains {len(self.code_snippets)} code snippets")