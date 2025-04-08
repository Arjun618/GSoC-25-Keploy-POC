"""
Sample implementation of machine learning models for code retrieval system testing.
"""
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin

class TextVectorizer(BaseEstimator, TransformerMixin):
    """
    A custom vectorizer for text data that implements the sklearn transformer interface.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: tuple = (1, 1)):
        """
        Initialize the text vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: The range of n-gram values to consider
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.document_count_ = 0
        self.idf_ = None
    
    def fit(self, X: List[str], y=None):
        """
        Fit the vectorizer to the data.
        
        Args:
            X: List of documents (strings)
            y: Ignored (for compatibility with sklearn)
            
        Returns:
            self
        """
        self.document_count_ = len(X)
        word_counts = {}
        doc_counts = {}
        
        # Count word occurrences across all documents
        for doc in X:
            seen_words = set()
            for word in self._tokenize(doc):
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
                
                # Count document frequency
                if word not in seen_words:
                    if word not in doc_counts:
                        doc_counts[word] = 0
                    doc_counts[word] += 1
                    seen_words.add(word)
        
        # Select top max_features based on frequency
        self.vocabulary_ = {
            word: idx for idx, (word, _) in enumerate(
                sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
            )
        }
        
        # Calculate IDF values
        self.idf_ = np.zeros(len(self.vocabulary_))
        for word, idx in self.vocabulary_.items():
            self.idf_[idx] = np.log((1 + self.document_count_) / (1 + doc_counts.get(word, 0))) + 1.0
        
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Transform documents to a document-term matrix.
        
        Args:
            X: List of documents (strings)
            
        Returns:
            Document-term matrix (scipy sparse matrix)
        """
        n_samples = len(X)
        n_features = len(self.vocabulary_)
        result = np.zeros((n_samples, n_features))
        
        for i, doc in enumerate(X):
            for word in self._tokenize(doc):
                if word in self.vocabulary_:
                    idx = self.vocabulary_[word]
                    result[i, idx] += 1
        
        # Apply TF-IDF transformation
        for i in range(n_samples):
            # L2 normalize term frequencies
            row_sum = np.sum(result[i, :] ** 2)
            if row_sum > 0:
                result[i, :] = result[i, :] / np.sqrt(row_sum)
            
            # Apply IDF weights
            result[i, :] = result[i, :] * self.idf_
        
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization by whitespace and lowercase
        tokens = text.lower().split()
        
        # Generate n-grams if needed
        if self.ngram_range[1] > 1:
            all_tokens = tokens.copy()
            for n in range(2, self.ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    all_tokens.append('_'.join(tokens[i:i+n]))
            return all_tokens
        
        return tokens


class CodeEmbeddingModel:
    """
    A model for generating embeddings from code snippets.
    """
    
    def __init__(self, embedding_size: int = 100, vocabulary_size: int = 5000):
        """
        Initialize the code embedding model.
        
        Args:
            embedding_size: Size of the embedding vector
            vocabulary_size: Size of the vocabulary
        """
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.embedding_matrix = np.random.randn(vocabulary_size, embedding_size) / 10
        self.word_to_index = {}
        self.index_to_word = {}
    
    def build_vocabulary(self, code_snippets: List[str]):
        """
        Build vocabulary from code snippets.
        
        Args:
            code_snippets: List of code snippets
        """
        word_counts = {}
        
        # Count all words in the code snippets
        for snippet in code_snippets:
            for word in self._tokenize_code(snippet):
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        # Select top words by frequency
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocabulary_size-1]
        
        # Build word-to-index and index-to-word mappings
        self.word_to_index = {"<UNK>": 0}
        self.index_to_word = {0: "<UNK>"}
        
        for i, (word, _) in enumerate(top_words):
            self.word_to_index[word] = i + 1
            self.index_to_word[i + 1] = word
    
    def generate_embedding(self, code: str) -> np.ndarray:
        """
        Generate embedding for a code snippet.
        
        Args:
            code: Code snippet
            
        Returns:
            Embedding vector
        """
        tokens = self._tokenize_code(code)
        indices = [self.word_to_index.get(token, 0) for token in tokens]
        
        # Average the embeddings of all tokens
        if not indices:
            return np.zeros(self.embedding_size)
        
        return np.mean([self.embedding_matrix[idx] for idx in indices], axis=0)
    
    def train(self, code_snippets: List[str], epochs: int = 10, learning_rate: float = 0.01):
        """
        Train the embedding model using skip-gram negative sampling.
        This is a simplified placeholder implementation.
        
        Args:
            code_snippets: List of code snippets
            epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        print(f"Training code embedding model on {len(code_snippets)} snippets...")
        # In a real implementation, this would train the model
        # For this prototype, we'll just use random embeddings
        pass
    
    def save_model(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        model_data = {
            "embedding_matrix": self.embedding_matrix,
            "word_to_index": self.word_to_index,
            "index_to_word": self.index_to_word,
            "embedding_size": self.embedding_size,
            "vocabulary_size": self.vocabulary_size
        }
        np.save(path, model_data)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        model_data = np.load(path, allow_pickle=True).item()
        self.embedding_matrix = model_data["embedding_matrix"]
        self.word_to_index = model_data["word_to_index"]
        self.index_to_word = model_data["index_to_word"]
        self.embedding_size = model_data["embedding_size"]
        self.vocabulary_size = model_data["vocabulary_size"]
        print(f"Model loaded from {path}")
    
    def _tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code snippet.
        
        Args:
            code: Code snippet
            
        Returns:
            List of tokens
        """
        # Simple tokenization by splitting on whitespace and punctuation
        # In a real implementation, this would use a more sophisticated code tokenizer
        code = code.replace('\n', ' ').replace('\t', ' ')
        for char in '()[]{};:,.\/=+-*&^%$#@!~`|<>?':
            code = code.replace(char, ' ' + char + ' ')
        return [token.strip().lower() for token in code.split() if token.strip()]