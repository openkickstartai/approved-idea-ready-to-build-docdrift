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
            "name": node.name, "type": "class" if isinstance(node, ast.ClassDef) else "function",
            "lineno": node.lineno, "docstring": docstring, "code_tokens": code_tokens,
        })
    return elements


def extract_doc_sections(markdown):
    """Extract heading-content pairs from markdown text."""
    sections, heading, lines = [], None, []
    for line in markdown.split('\n'):
        m = re.match(r'^(#{1,6})\s+(.+)', line)
        if m:
            if heading is not None:
                sections.append({"heading": heading, "content": '\n'.join(lines).strip()})
            heading, lines = m.group(2).strip(), []
        else:
            lines.append(line)
    if heading is not None:
        sections.append({"heading": heading, "content": '\n'.join(lines).strip()})
    return sections


def compute_drift(docstring_text, code_tokens):
    """Compute drift score: 0.0=perfectly aligned, 1.0=fully drifted."""
    doc_tokens = tokenize(docstring_text)
    similarity = cosine_similarity(doc_tokens, code_tokens)
    return round(1.0 - similarity, 4)


def analyze(source, threshold=0.7):
    """Analyze Python source code for doc-code cognitive drift."""
    reports = []
    for elem in extract_code_elements(source):
        if not elem["docstring"]:
            reports.append({
                "name": elem["name"], "type": elem["type"], "line": elem["lineno"],
                "drift": 1.0, "status": "missing_docs",
                "message": f"No docstring for {elem['type']} '{elem['name']}'",
            })
        else:
            drift = compute_drift(elem["docstring"], elem["code_tokens"])
            status = "drifted" if drift >= threshold else "aligned"
            reports.append({
                "name": elem["name"], "type": elem["type"], "line": elem["lineno"],
                "drift": drift, "status": status,
                "message": f"{elem['type']} '{elem['name']}': drift={drift:.1%}",
            })
    return reports
