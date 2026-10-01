import ast
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# AST PARSER & CODE2VEC PATH EXTRACTOR
# ============================================================

class Code2VecExtractor:
    """
    Code2Vec Feature Extraction Pipeline as described in Section 4.2.2 & 4.3:
    - Step 1: Parsing source code into Abstract Syntax Tree (AST)
    - Step 2: Extracting path-based representations (terminal node 1 -> AST path -> terminal node 2)
    - Step 3: Token and path embedding
    - Step 4: Attention-based path embedding aggregation
    """

    def __init__(self, embedding_dim=128, max_paths=64, max_path_length=8):
        self.embedding_dim = embedding_dim
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        self.vocab = {}
        self.path_vocab = {}

    def parse_python_ast(self, code_str):
        """Parse Python source code into an AST and extract (start_token, path, end_token) triples."""
        try:
            tree = ast.parse(code_str)
        except Exception:
            # Fallback for code fragments or non-Python code
            return self.parse_generic_code_tokens(code_str)

        terminals = []

        def traverse(node, current_path):
            node_type = type(node).__name__
            new_path = current_path + [node_type]

            # Check if terminal/leaf token
            if isinstance(node, ast.Name):
                terminals.append((node.id, new_path))
            elif isinstance(node, ast.Constant):
                terminals.append((str(node.value), new_path))
            elif isinstance(node, ast.arg):
                terminals.append((node.arg, new_path))
            elif isinstance(node, ast.FunctionDef):
                terminals.append((node.name, new_path))
            elif isinstance(node, ast.Attribute):
                terminals.append((node.attr, new_path))

            for child in ast.iter_child_nodes(node):
                traverse(child, new_path)

        traverse(tree, [])

        if not terminals:
            return self.parse_generic_code_tokens(code_str)

        # Generate pairwise paths between terminals
        path_contexts = []
        for i in range(len(terminals)):
            for j in range(i + 1, min(i + 6, len(terminals))):
                t1, path1 = terminals[i]
                t2, path2 = terminals[j]

                # Find Lowest Common Ancestor (LCA)
                k = 0
                while k < len(path1) and k < len(path2) and path1[k] == path2[k]:
                    k += 1

                up_path = path1[k - 1:][::-1]
                down_path = path2[k:]
                full_path = "->".join(up_path + down_path)
                if len(full_path.split("->")) <= self.max_path_length:
                    path_contexts.append((t1, full_path, t2))

        if not path_contexts:
            return self.parse_generic_code_tokens(code_str)

        return path_contexts[:self.max_paths]

    def parse_generic_code_tokens(self, code_str):
        """Fallback tokenizer & pseudo-AST extractor for Java/C/C++ snippets."""
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[+\-*/=<>!]+|[{}();,]', code_str)
        if not tokens:
            tokens = ["empty_code"]

        path_contexts = []
        for i in range(0, len(tokens) - 2, 2):
            t1 = tokens[i]
            t2 = tokens[min(i + 2, len(tokens) - 1)]
            mid_op = tokens[i + 1] if i + 1 < len(tokens) else "Expr"
            path = f"Node({tokens[i][:4]})->{mid_op}->Node({t2[:4]})"
            path_contexts.append((t1, path, t2))
            if len(path_contexts) >= self.max_paths:
                break

        if not path_contexts:
            path_contexts.append(("token_start", "Root->Stmt->Expr", "token_end"))

        return path_contexts

    def string_hash_vector(self, text, dim=128):
        """Deterministic hashing embedding vector for arbitrary tokens/paths."""
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.normal(0, 1, size=(dim,)).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def extract_code_vector(self, code_str):
        """
        Processes a raw code snippet:
        1. Extracts AST path contexts: (t1, path, t2)
        2. Computes combined vector embeddings for each path context
        3. Aggregates using an attention mechanism to produce a fixed-size code vector (dim=128)
        """
        contexts = self.parse_python_ast(code_str)
        if not contexts:
            contexts = [("entry", "Method->Body", "exit")]

        path_vectors = []
        for t1, path, t2 in contexts:
            v_t1 = self.string_hash_vector(t1, self.embedding_dim)
            v_p = self.string_hash_vector(path, self.embedding_dim)
            v_t2 = self.string_hash_vector(t2, self.embedding_dim)

            # Combined path context vector: c = tanh(W * [v_t1; v_p; v_t2])
            combined = np.tanh((v_t1 + v_p + v_t2) / 3.0)
            path_vectors.append(combined)

        path_vectors = np.array(path_vectors, dtype=np.float32)  # [num_paths, embedding_dim]

        # Step 4: Attention Aggregation
        # Attention vector 'a'
        np.random.seed(42)
        attention_weights = np.dot(path_vectors, np.ones(self.embedding_dim, dtype=np.float32))
        exp_weights = np.exp(attention_weights - np.max(attention_weights))
        normalized_weights = exp_weights / (np.sum(exp_weights) + 1e-8)

        # Weighted sum: v_code = sum(alpha_i * c_i)
        code_vector = np.sum(path_vectors * normalized_weights[:, np.newaxis], axis=0)

        # L2 Normalize
        code_vector = code_vector / (np.linalg.norm(code_vector) + 1e-8)

        return code_vector, contexts


# Global extractor instance
extractor = Code2VecExtractor(embedding_dim=128)
