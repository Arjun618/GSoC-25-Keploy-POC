# Code Retrieval System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

An advanced open-source code retrieval system built on modern NLP and information retrieval techniques. This system enables efficient searching and discovery of code snippets through natural language queries, supporting both semantic understanding and structural code awareness.

## Overview

Modern software development requires quick access to relevant code snippets. This system addresses this need by providing:

- **Semantic Code Search**: Find code based on natural language descriptions
- **Hybrid Retrieval**: Combines dense vector embeddings (semantic) and sparse retrieval (keyword) for superior results
- **Graph-Aware Reranking**: Understands code relationships and dependencies
- **Voice Input Support**: Query code through voice commands (experimental)

## Key Features

- **Compressed Code Embedding Pipeline**: Efficient representation of code using quantized transformer models
- **Hybrid Retrieval System**: Combines best-of-both-worlds approaches:
  - Dense retrieval (CodeBERT embeddings)
  - Sparse retrieval (BM25 algorithm)
- **Graph-Aware Retrieval**: Analyzes code structure, function calls, and class relationships
- **Multi-Modal Query Input**: Support for text and voice-based queries

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Arjun618/GSoC-25-Keploy-POC.git
   cd GSoC-25-Keploy-POC
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download pre-trained models:
   ```bash
   # Note: This feature is coming soon
   # Models will be downloaded automatically on first run
   ```

## Usage

### Running the Server

Start the API server with:

```bash
python src/api/server.py
```

This will launch the server at `http://localhost:8000` with the interactive UI available at the root URL.

### System Requirements

- **CPU**: Minimum dual-core processor (quad-core recommended)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk Space**: ~2GB for models and indexes
- **Operating Systems**: Windows, macOS, and Linux

### Web Interface

Access the web interface by opening `http://localhost:8000` in your browser. The interface provides:

- Text-based search with mode selection (dense, sparse, hybrid)
- Optional graph-based reranking
- Voice search capabilities
- Code submission form

### API Endpoints

The system exposes the following REST API endpoints:

- `GET /api/search?query=<query>&mode=<mode>&max_results=<max_results>&use_graph=<use_graph>`
  - Returns: `{"query": "...", "mode": "...", "results": [{"code": "...", "file_path": "...", "score": 0.95}]}`
- `POST /api/voice-search`
  - Request body: `{"duration": 5, "mode": "hybrid", "max_results": 10, "use_graph": false}`
  - Returns: Same format as search endpoint with additional transcribed query
- `POST /api/add-code`
  - Request body: `{"code": "...", "file_path": "..."}`
  - Returns: `{"status": "success", "message": "Code snippet added successfully"}`

## Examples

### Text Search Example

To search for code that "reads data from a CSV file":

1. Enter the query in the search box
2. Select "Hybrid" mode for best results
3. Enable "Use Graph Reranking" for structure-aware results
4. Click "Search"

Results will include relevant code snippets with their file paths and similarity scores.

### Voice Search Example

To search using voice:

1. Click the "Voice Search" button
2. Speak your query clearly (e.g., "find function to normalize data")
3. Wait for transcription and search results

### Adding Code Example

To add custom code to the system:

1. Paste your code in the "Code Snippet" textarea
2. (Optional) Provide a file path 
3. Click "Add Code"

Your code will be indexed and available for future searches.

## Testing

Run the test suite with the following command:

```bash
python -m unittest discover tests
```

## Troubleshooting

### Common Issues

1. **Models fail to download**
   - Check your internet connection
   - Try running with administrator/sudo privileges
   - Models will be downloaded on first run

2. **Voice search not working**
   - Ensure your microphone is properly connected
   - Install required system dependencies: `apt-get install libportaudio2` (Linux)
   - Voice search requires additional packages which must be properly installed

3. **High memory usage**
   - Lower model quality by editing `config/settings.json`
   - Disable graph-based reranking for lower memory footprint

## Project Structure

- `src/`: Source code for the application
  - `embedding/`: Code embedding generation
  - `retrieval/`: Retrieval mechanisms (dense and sparse)
  - `api/`: API server and endpoints
  - `graph/`: Graph-based code analysis
  - `voice/`: Voice input processing
- `data/`: Sample code repositories and indexed data
- `tests/`: Test scripts
- `config/`: Configuration files
- `models/`: Saved model files

## Development

### Setting Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black isort
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. A copy of the license is included in the root directory of this repository.
