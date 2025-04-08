"""
Code embedding generation module using a lightweight language model
"""
import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Union
import numpy as np

class CodeEmbeddingGenerator:
    """
    Generates embeddings for code snippets using a lightweight model
    with quantization for performance optimization.
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize the code embedding generator with a specified model.
        
        Args:
            model_name: The name of the pre-trained model to use (default: codebert-base)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with quantization for reduced memory footprint
        self.model = AutoModel.from_pretrained(model_name)
        
        # Apply dynamic quantization to the model
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        self.model.to(self.device)
        print(f"Model loaded and quantized. Using device: {self.device}")
    
    def generate_embedding(self, code_snippet: str) -> np.ndarray:
        """
        Generate an embedding for a single code snippet.
        
        Args:
            code_snippet: The code snippet to generate an embedding for
            
        Returns:
            numpy array: The embedding vector
        """
        # Tokenize the code snippet
        inputs = self.tokenizer(
            code_snippet, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token embedding as the code snippet embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding[0]  # Return the embedding as a 1D array
    
    def generate_embeddings(self, code_snippets: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets.
        
        Args:
            code_snippets: List of code snippets to generate embeddings for
            
        Returns:
            numpy array: Matrix of embedding vectors
        """
        embeddings = []
        for snippet in code_snippets:
            embedding = self.generate_embedding(snippet)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        """
        Save embeddings to a file.
        
        Args:
            embeddings: The embeddings to save
            file_path: The path to save the embeddings to
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f"Saved embeddings to {file_path}")
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        Load embeddings from a file.
        
        Args:
            file_path: The path to load the embeddings from
            
        Returns:
            numpy array: Matrix of embedding vectors
        """
        embeddings = np.load(file_path)
        print(f"Loaded embeddings from {file_path}")
        return embeddings