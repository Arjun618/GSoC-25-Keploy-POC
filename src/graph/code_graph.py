"""
Graph-aware code analysis module for improved code retrieval
"""
import ast
import networkx as nx
from typing import Dict, List, Any, Tuple, Set
import matplotlib.pyplot as plt
import os

class CodeGraphBuilder:
    """
    Builds a graph representation of Python code using AST parsing
    """
    
    def __init__(self):
        """
        Initialize the code graph builder
        """
        self.graph = nx.DiGraph()
    
    def parse_code(self, code: str, file_path: str = "") -> nx.DiGraph:
        """
        Parse Python code into a graph representation
        
        Args:
            code: The Python code to parse
            file_path: The file path of the code (optional)
            
        Returns:
            The graph representation of the code
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Build the graph
            self._build_graph(tree, file_path)
            
            return self.graph
        except SyntaxError as e:
            print(f"Failed to parse code: {e}")
            return self.graph
    
    def _build_graph(self, tree: ast.AST, file_path: str):
        """
        Build a graph representation of the AST
        
        Args:
            tree: The AST to build a graph from
            file_path: The file path of the code
        """
        # Reset the graph
        self.graph = nx.DiGraph()
        
        # Add the file node
        file_node_id = f"file:{file_path}"
        self.graph.add_node(file_node_id, type="file", name=file_path)
        
        # Visit all nodes in the AST and add them to the graph
        for node in ast.walk(tree):
            # Skip simple literals and constants
            if isinstance(node, (ast.Constant, ast.Str, ast.Num, ast.Bytes)):
                continue
                
            # Get node information
            node_info = self._get_node_info(node)
            
            if node_info:
                node_id, node_attrs = node_info
                
                # Add the node to the graph
                self.graph.add_node(node_id, **node_attrs)
                
                # Connect to the file node
                self.graph.add_edge(file_node_id, node_id)
                
                # Add edges based on node type
                self._add_node_edges(node, node_id)
    
    def _get_node_info(self, node: ast.AST) -> Tuple[str, Dict[str, Any]]:
        """
        Get information about an AST node
        
        Args:
            node: The AST node
            
        Returns:
            Tuple of node ID and node attributes
        """
        node_id = f"{id(node)}"
        
        if isinstance(node, ast.FunctionDef):
            return node_id, {
                "type": "function",
                "name": node.name,
                "line": node.lineno
            }
        elif isinstance(node, ast.ClassDef):
            return node_id, {
                "type": "class",
                "name": node.name,
                "line": node.lineno
            }
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return node_id, {
                "type": "import",
                "names": names,
                "line": node.lineno
            }
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            return node_id, {
                "type": "importfrom",
                "module": node.module,
                "names": names,
                "line": node.lineno
            }
        elif isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                
            return node_id, {
                "type": "call",
                "name": func_name,
                "line": getattr(node, "lineno", 0)
            }
        elif isinstance(node, ast.Assign):
            targets = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                    
            return node_id, {
                "type": "assign",
                "targets": targets,
                "line": node.lineno
            }
        else:
            # Skip other node types
            return None
    
    def _add_node_edges(self, node: ast.AST, node_id: str):
        """
        Add edges to the graph based on the node type
        
        Args:
            node: The AST node
            node_id: The ID of the node in the graph
        """
        # Function calls
        if isinstance(node, ast.Call):
            # Add edges to arguments
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    arg_id = self._find_node_by_name(arg.id)
                    if arg_id:
                        self.graph.add_edge(node_id, arg_id, type="calls")
        
        # Function definitions
        elif isinstance(node, ast.FunctionDef):
            # Add edges to function calls within the function
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and hasattr(child, "func"):
                    if isinstance(child.func, ast.Name):
                        call_id = self._find_node_by_name(child.func.id)
                        if call_id:
                            self.graph.add_edge(node_id, call_id, type="calls")
    
    def _find_node_by_name(self, name: str) -> str:
        """
        Find a node in the graph by name
        
        Args:
            name: The name to search for
            
        Returns:
            The ID of the node if found, otherwise None
        """
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("name") == name:
                return node_id
        return None
    
    def get_function_dependencies(self, function_name: str) -> Set[str]:
        """
        Get the dependencies of a function
        
        Args:
            function_name: The name of the function
            
        Returns:
            Set of function names that are called by the function
        """
        dependencies = set()
        
        # Find the function node
        function_id = None
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "function" and attrs.get("name") == function_name:
                function_id = node_id
                break
                
        if function_id:
            # Find all outgoing edges of type "calls"
            for _, target, edge_attrs in self.graph.out_edges(function_id, data=True):
                if edge_attrs.get("type") == "calls":
                    target_attrs = self.graph.nodes[target]
                    if "name" in target_attrs:
                        dependencies.add(target_attrs["name"])
                        
        return dependencies
    
    def get_class_members(self, class_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the members of a class
        
        Args:
            class_name: The name of the class
            
        Returns:
            Dictionary of class members by type
        """
        members = {
            "functions": [],
            "attributes": []
        }
        
        # Find the class node
        class_id = None
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "class" and attrs.get("name") == class_name:
                class_id = node_id
                break
                
        if class_id:
            # Find all functions and attributes in the class
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("type") == "function" and self.graph.has_edge(class_id, node_id):
                    members["functions"].append({
                        "name": attrs.get("name"),
                        "line": attrs.get("line")
                    })
                elif attrs.get("type") == "assign" and self.graph.has_edge(class_id, node_id):
                    for target in attrs.get("targets", []):
                        members["attributes"].append({
                            "name": target,
                            "line": attrs.get("line")
                        })
                        
        return members
    
    def visualize_graph(self, output_file: str = "code_graph.png"):
        """
        Visualize the graph using matplotlib
        
        Args:
            output_file: The file to save the visualization to
        """
        plt.figure(figsize=(12, 8))
        
        # Define node colors based on type
        node_colors = []
        for _, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "function":
                node_colors.append("blue")
            elif attrs.get("type") == "class":
                node_colors.append("red")
            elif attrs.get("type") == "import" or attrs.get("type") == "importfrom":
                node_colors.append("green")
            elif attrs.get("type") == "call":
                node_colors.append("orange")
            elif attrs.get("type") == "file":
                node_colors.append("purple")
            else:
                node_colors.append("gray")
                
        # Set node labels
        node_labels = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "function" or attrs.get("type") == "class":
                node_labels[node_id] = attrs.get("name", "")
            elif attrs.get("type") == "call":
                node_labels[node_id] = f"call:{attrs.get('name', '')}"
            elif attrs.get("type") == "file":
                node_labels[node_id] = os.path.basename(attrs.get("name", ""))
            else:
                node_labels[node_id] = attrs.get("type", "")
                
        # Draw the graph
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=8)
        
        plt.title("Code Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
        
    def get_relations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relations in the graph
        
        Returns:
            Dictionary of relations by type
        """
        relations = {
            "function_calls": [],
            "class_members": [],
            "imports": []
        }
        
        # Function calls
        for source, target, attrs in self.graph.edges(data=True):
            if attrs.get("type") == "calls":
                source_attrs = self.graph.nodes[source]
                target_attrs = self.graph.nodes[target]
                
                if source_attrs.get("type") == "function" and target_attrs.get("type") == "function":
                    relations["function_calls"].append({
                        "caller": source_attrs.get("name"),
                        "callee": target_attrs.get("name")
                    })
        
        # Class members
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "class":
                class_name = attrs.get("name")
                members = self.get_class_members(class_name)
                
                relations["class_members"].append({
                    "class": class_name,
                    "functions": [func["name"] for func in members["functions"]],
                    "attributes": [attr["name"] for attr in members["attributes"]]
                })
        
        # Imports
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "import":
                relations["imports"].extend(attrs.get("names", []))
            elif attrs.get("type") == "importfrom":
                module = attrs.get("module", "")
                names = attrs.get("names", [])
                for name in names:
                    relations["imports"].append(f"{module}.{name}")
        
        return relations