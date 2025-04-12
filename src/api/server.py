"""
API server for the code retrieval system
"""
import os
import sys
import json

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Any, Optional
import numpy as np

from src.embedding.code_embedder import CodeEmbeddingGenerator
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.graph.code_graph import CodeGraphBuilder
from src.voice.voice_processor import VoiceInputProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Code Retrieval System",
    description="An open-source code retrieval system with dense and sparse search capabilities",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Initialize static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration
CONFIG = {
    "data_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
    "models_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"),
    "config_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"),
    "index_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "index"),
    "embedding_model": "microsoft/codebert-base",
    "voice_model": "base",
    "max_results": 10
}

# Ensure directories exist
for dir_path in [CONFIG["data_dir"], CONFIG["models_dir"], CONFIG["config_dir"], CONFIG["index_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize components
code_embedder = None
dense_retriever = None
sparse_retriever = None
hybrid_retriever = None
code_graph_builder = None
voice_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global code_embedder, dense_retriever, sparse_retriever, hybrid_retriever, code_graph_builder, voice_processor
    
    # Initialize embedding generator
    print("Initializing code embedding generator...")
    code_embedder = CodeEmbeddingGenerator(model_name=CONFIG["embedding_model"])
    
    # Initialize dense retriever
    print("Initializing dense retriever...")
    dense_retriever = DenseRetriever()
    
    # Initialize sparse retriever
    print("Initializing sparse retriever...")
    sparse_retriever = SparseRetriever(index_dir=CONFIG["index_dir"])
    
    # Initialize hybrid retriever
    print("Initializing hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        code_embedder=code_embedder
    )
    
    # Initialize code graph builder
    print("Initializing code graph builder...")
    code_graph_builder = CodeGraphBuilder()
    
    # Initialize voice processor (optional)
    try:
        print("Initializing voice processor...")
        voice_processor = VoiceInputProcessor(model_name=CONFIG["voice_model"])
    except Exception as e:
        print(f"Warning: Failed to initialize voice processor: {e}")
        voice_processor = None
    
    # Load sample code if available
    sample_code_dir = os.path.join(CONFIG["data_dir"], "sample_code")
    if os.path.exists(sample_code_dir):
        print(f"Loading sample code from {sample_code_dir}...")
        load_sample_code(sample_code_dir)

def load_sample_code(sample_code_dir: str):
    """Load sample code from the given directory"""
    code_snippets = []
    file_paths = []
    
    # Walk through the directory and find Python files
    for root, _, files in os.walk(sample_code_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    
                    code_snippets.append(code)
                    file_paths.append(file_path)
                    
                    # Parse code into graph
                    code_graph_builder.parse_code(code, file_path)
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
    
    if code_snippets:
        print(f"Loaded {len(code_snippets)} code snippets")
        
        # Add code snippets to hybrid retriever
        hybrid_retriever.add_code_snippets(code_snippets, file_paths)
    else:
        print("No sample code found")

@app.get("/")
async def read_root(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def search(
    query: str = Query(..., description="The query string"),
    mode: str = Query("hybrid", description="The search mode (dense, sparse, hybrid)"),
    max_results: int = Query(10, description="The maximum number of results to return"),
    use_graph: bool = Query(False, description="Whether to use graph-based reranking")
) -> Dict[str, Any]:
    """
    Search for code snippets using the specified mode
    """
    if not hybrid_retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
        
    results = []
    
    # Search using the specified mode
    if mode == "dense":
        query_embedding = code_embedder.generate_embedding(query)
        results = dense_retriever.search(query_embedding, k=max_results)
    elif mode == "sparse":
        results = sparse_retriever.search(query, k=max_results)
    else:  # hybrid
        results = hybrid_retriever.search(query, k=max_results)
    
    # Apply graph-based reranking if requested
    if use_graph and code_graph_builder and results:
        results = apply_graph_reranking(results)
    
    return {
        "query": query,
        "mode": mode,
        "results": results
    }

def apply_graph_reranking(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply graph-based reranking to the search results
    """
    # Extract code relations from the graph
    relations = code_graph_builder.get_relations()
    
    # Add relation information to the results
    for result in results:
        code = result["code"]
        file_path = result.get("file_path", "")
        
        # Parse the code to extract relations
        if file_path:
            try:
                # Find function and class dependencies
                # This is a simple implementation - could be extended for more complex graph analysis
                for func_call in relations["function_calls"]:
                    if func_call["caller"] in code or func_call["callee"] in code:
                        if "related_functions" not in result:
                            result["related_functions"] = []
                        result["related_functions"].append(func_call)
                
                for class_info in relations["class_members"]:
                    if class_info["class"] in code:
                        result["class_info"] = class_info
            except Exception as e:
                print(f"Warning: Failed to analyze code relations: {e}")
    
    # Rerank results based on relation density
    # (More relations = higher rank)
    for result in results:
        relation_score = 0
        if "related_functions" in result:
            relation_score += len(result["related_functions"]) * 0.1
        if "class_info" in result:
            relation_score += (len(result["class_info"].get("functions", [])) + 
                               len(result["class_info"].get("attributes", []))) * 0.05
        
        # Adjust score based on relations
        result["score"] = result["score"] * (1 + relation_score)
    
    # Sort by adjusted score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

@app.post("/api/voice-search")
async def voice_search(
    duration: int = Query(5, description="The duration to record in seconds"),
    mode: str = Query("hybrid", description="The search mode (dense, sparse, hybrid)"),
    max_results: int = Query(10, description="The maximum number of results to return"),
    use_graph: bool = Query(False, description="Whether to use graph-based reranking")
) -> Dict[str, Any]:
    """
    Record voice input and search for code snippets
    """
    if not voice_processor:
        raise HTTPException(status_code=500, detail="Voice processor not initialized")
        
    # Get voice query
    query = voice_processor.get_voice_query(duration)
    
    # Search using the query
    return await search(query=query, mode=mode, max_results=max_results, use_graph=use_graph)

@app.post("/api/add-code")
async def add_code(
    code: str = Query(..., description="The code snippet to add"),
    file_path: str = Query(None, description="The file path of the code (optional)")
) -> Dict[str, Any]:
    """
    Add a code snippet to the system
    """
    if not hybrid_retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
        
    # Add the code snippet
    hybrid_retriever.add_code_snippets([code], [file_path] if file_path else None)
    
    # Parse code into graph
    if code_graph_builder:
        code_graph_builder.parse_code(code, file_path or "")
    
    return {"status": "success", "message": "Code snippet added successfully"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)