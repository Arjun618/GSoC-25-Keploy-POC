"""
Sparse retrieval module using Whoosh for BM25-based search
"""
import os
import shutil
from typing import List, Dict, Any
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
from whoosh.writing import AsyncWriter

class SparseRetriever:
    """
    Retrieves code snippets using BM25-based sparse retrieval with Whoosh
    """
    
    def __init__(self, index_dir: str = "index"):
        """
        Initialize the sparse retriever with Whoosh
        
        Args:
            index_dir: Directory to store the Whoosh index
        """
        self.index_dir = index_dir
        
        # Create schema for the index
        self.schema = Schema(
            id=ID(stored=True),
            code=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            file_path=STORED
        )
        
        # Create index directory if it doesn't exist
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            self.ix = index.create_in(index_dir, self.schema)
            print(f"Created new Whoosh index in {index_dir}")
        else:
            # Open existing index
            try:
                self.ix = index.open_dir(index_dir)
                print(f"Opened existing Whoosh index in {index_dir}")
            except:
                # If index is corrupted, create a new one
                shutil.rmtree(index_dir)
                os.makedirs(index_dir)
                self.ix = index.create_in(index_dir, self.schema)
                print(f"Created new Whoosh index in {index_dir} (replaced corrupted index)")
    
    def add_documents(self, code_snippets: List[str], file_paths: List[str] = None):
        """
        Add documents to the index
        
        Args:
            code_snippets: List of code snippets to add
            file_paths: List of file paths corresponding to the snippets (optional)
        """
        if file_paths and len(file_paths) != len(code_snippets):
            raise ValueError("Number of file paths must match number of code snippets")
            
        # Use AsyncWriter for better performance
        writer = AsyncWriter(self.ix)
        
        for i, snippet in enumerate(code_snippets):
            file_path = file_paths[i] if file_paths else ""
            
            # Add document to the index
            writer.add_document(
                id=str(i),
                code=snippet,
                file_path=file_path
            )
            
        # Commit changes
        writer.commit()
        print(f"Added {len(code_snippets)} documents to the Whoosh index")
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query
        
        Args:
            query: The query string
            k: The number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing the code snippets, file paths, and scores
        """
        # Create a query parser for the "code" field
        parser = QueryParser("code", self.ix.schema, group=OrGroup)
        q = parser.parse(query)
        
        # Search for documents
        with self.ix.searcher() as searcher:
            results = searcher.search(q, limit=k)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "code": result["code"],
                    "file_path": result["file_path"],
                    "score": result.score
                })
                
            return formatted_results