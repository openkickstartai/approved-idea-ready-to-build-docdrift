"""DocDrift — Semantic drift detection between documentation and code."""
import ast
import re
import math
from collections import Counter


def tokenize(text):
    """Split text into lowercase alphabetic tokens."""
    return re.findall(r'[a-z][a-z0-9]*', text.lower())


def split_identifier(name):
    """Split camelCase and snake_case identifiers into word tokens."""
    expanded = re.sub(r'([A-Z])', r' \1', name).replace('_', ' ')
    return re.findall(r'[a-z][a-z0-9]*', expanded.lower())


def cosine_similarity(tokens_a, tokens_b):
    """Cosine similarity between two token frequency vectors."""
    if not tokens_a or not tokens_b:
        return 0.0
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    keys = set(ca) | set(cb)
    dot = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v * v for v in ca.values()))
    mag_b = math.sqrt(sum(v * v for v in cb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def extract_code_elements(source):
    """Extract functions and classes with docstrings and code tokens."""
    tree = ast.parse(source)
    elements = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        docstring = ast.get_docstring(node) or ""
        code_tokens = list(split_identifier(node.name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                code_tokens.extend(split_identifier(arg.arg))
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                code_tokens.extend(split_identifier(child.id))
            elif isinstance(child, ast.Attribute):
                code_tokens.extend(split_identifier(child.attr))
        elements.append({
            "name": node.name,
            "type": "class" if isinstance(node, ast.ClassDef) else "function",
            "docstring": docstring,
            "code_tokens": code_tokens,
            "args": [arg.arg for arg in node.args.args] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [],
        })
    return elements


def extract_doc_sections(docstring):
    """Extract semantic tokens from a docstring."""
    if not docstring:
        return []
    return tokenize(docstring)


def compute_drift(code_tokens, doc_tokens):
    """Compute drift score between code and doc tokens.

    Returns a float in [0.0, 1.0] where 0.0 means fully aligned
    and 1.0 means completely drifted.
    """
    sim = cosine_similarity(code_tokens, doc_tokens)
    return round(1.0 - sim, 4)


def analyze(source, threshold=0.7):
    """Analyze a Python source string for doc-code drift.

    Returns a list of report dicts with keys:
        name, type, status, drift, message.
    """
    elements = extract_code_elements(source)
    reports = []
    for elem in elements:
        name = elem["name"]
        docstring = elem["docstring"]
        code_tokens = elem["code_tokens"]
        if not docstring:
            reports.append({
                "name": name,
                "type": elem["type"],
                "status": "missing_docs",
                "drift": 1.0,
                "message": f"No docstring for {elem['type']} '{name}'",
            })
            continue
        doc_tokens = extract_doc_sections(docstring)
        drift = compute_drift(code_tokens, doc_tokens)
        if drift >= threshold:
            status = "drifted"
        else:
            status = "aligned"
        reports.append({
            "name": name,
            "type": elem["type"],
            "status": status,
            "drift": drift,
            "message": f"{elem['type']} '{name}': drift={drift:.1%}",
        })
    return reports
