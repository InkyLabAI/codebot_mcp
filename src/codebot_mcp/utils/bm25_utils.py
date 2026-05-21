import re
from typing import List
import logging
logger = logging.getLogger(__name__)


def build_search_text(function_data: dict) -> str:
    """
    Build BM25-optimized search text.
    Focus: Extract and repeat important keywords.
    """
    keywords = []
    
    # 1. Function name (concatenated + split, repeat 2x for importance)
    name = function_data['name']
    func_words = split_identifier(name)
    # Concatenated form (no underscores) survives PG tokenizer + stopword removal
    # e.g. is_on_next_page → isonextpage (single token that PG keeps intact)
    name_concat = name.replace('_', '')
    keywords.extend([name_concat] * 2)
    keywords.extend(func_words * 2)   # split form for partial word matching

    # 2. Class name (concatenated + split, repeat 2x)
    if function_data.get('class_name'):
        cls = function_data['class_name']
        class_words = split_identifier(cls)
        cls_concat = cls.replace('_', '')
        keywords.extend([cls_concat] * 2)
        keywords.extend(class_words * 2)
    
    # 3. File path components (split path, repeat 1x)
    file_path = function_data['file_path']
    # ultralytics/models/yolo/v11/train.py → [ultralytics, models, yolo, v11, train]
    path_parts = [p for p in file_path.replace('.py', '').split('/') if p]
    keywords.extend(path_parts)
    
    # 4. Parameter names (useful for "with image parameter" queries)
    if function_data.get('parameters'):
        param_words = []
        for p in function_data['parameters']:
            if p.get('name'):
                param_words.extend(split_identifier(p['name']))
            if p.get('type'):
                param_words.extend(split_identifier(p['type']))
        keywords.extend(param_words)
    
    # 5. Return type
    if function_data.get('return_type'):
        keywords.extend(split_identifier(function_data['return_type']))
    
    # 6. Docstring (most important - full text)
    docstring = ""
    if function_data.get('docstring'):
        docstring = function_data['docstring'].strip()
        # Remove docstring quotes
        for quote in ['"""', "'''"]:
            docstring = docstring.replace(quote, '')

    # 7. Class docstring (class-level context)
    if function_data.get('class_docstring'):
        class_doc = function_data['class_docstring'].strip()
        for quote in ['"""', "'''"]:
            class_doc = class_doc.replace(quote, '')
        docstring = class_doc + " " + docstring if docstring else class_doc

    # 8. Module docstring (file-level context)
    if function_data.get('module_docstring'):
        module_doc = function_data['module_docstring'].strip()
        for quote in ['"""', "'''"]:
            module_doc = module_doc.replace(quote, '')
        docstring = docstring + " " + module_doc if docstring else module_doc

    # Combine: keywords first (weighted), then docstring
    return " ".join(keywords) + "\n" + docstring


def split_identifier(name: str) -> list:
    """
    Split identifiers on:
    - Underscores: train_model → train model
    - CamelCase: YOLODetector → YOLO Detector  
    - Letter-number boundaries: v11 → v 11, ResNet50 → ResNet 50
    
    Keeps original case (PostgreSQL to_tsvector handles normalization).
    """    
    # Split letter→number and number→letter boundaries
    # v11 → v 11, ResNet50 → ResNet 50, 3Plus → 3 Plus
    name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', name)  # letter BEFORE number
    name = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', name)  # number BEFORE letter
    
    # Split on underscores
    words = name.split('_')
    
    # Split camelCase in each word
    result = []
    for word in words:
        # Insert space before uppercase letters
        # YOLODetector → YOLO Detector
        # HTTPSConnection → HTTPS Connection  
        spaced = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word))
        result.extend([w for w in spaced.split() if w])
    
    return result


def calculate_rrf_score(
    semantic_rank: int,
    bm25_rank: int,
    semantic_weight: float = 3.0,
    bm25_weight: float = 1.0,
    k: int = 60
) -> float:
    """
    Calculate weighted Reciprocal Rank Fusion (RRF) score.
    
    RRF formula: score = w1 * (1 / (k + rank1)) + w2 * (1 / (k + rank2))
    
    Args:
        semantic_rank: Position in semantic search results (1-based)
        bm25_rank: Position in BM25 search results (1-based)
        semantic_weight: Weight for semantic search (default 3.0)
        bm25_weight: Weight for BM25 search (default 1.0)
        k: RRF constant (default 60, standard in literature)
        
    Returns:
        Combined RRF score (higher is better)
    """
    # Handle missing ranks (not in that result set)
    semantic_score = semantic_weight / (k + semantic_rank) if semantic_rank > 0 else 0
    bm25_score = bm25_weight / (k + bm25_rank) if bm25_rank > 0 else 0
    
    return semantic_score + bm25_score


def merge_results_with_rrf(
    semantic_results: List,
    bm25_results: List,
    semantic_weight: float = 3.0,
    bm25_weight: float = 1.0,
    limit: int = 20
) -> List:
    """
    Merge semantic and BM25 results using weighted RRF.
    
    Args:
        semantic_results: List of FunctionSearchResult from semantic search (ordered)
        bm25_results: List of FunctionSearchResult from BM25 search (ordered)
        semantic_weight: Weight for semantic results (default 3.0)
        bm25_weight: Weight for BM25 results (default 1.0)
        limit: Maximum results to return
        
    Returns:
        Merged and re-ranked FunctionSearchResult list
    """
    # Build rank maps (function_id -> rank)
    semantic_ranks = {
        result.function.function_id: idx + 1 
        for idx, result in enumerate(semantic_results)
    }
    
    bm25_ranks = {
        result.function.function_id: idx + 1 
        for idx, result in enumerate(bm25_results)
    }
    
    # Get all unique function IDs
    all_function_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
    
    # Calculate RRF scores for all functions
    scored_results = []
    
    # Build function map for quick lookup
    function_map = {}
    for result in semantic_results:
        function_map[result.function.function_id] = result
    for result in bm25_results:
        if result.function.function_id not in function_map:
            function_map[result.function.function_id] = result
    
    for function_id in all_function_ids:
        semantic_rank = semantic_ranks.get(function_id, 0)
        bm25_rank = bm25_ranks.get(function_id, 0)
        
        # Calculate RRF score
        rrf_score = calculate_rrf_score(
            semantic_rank,
            bm25_rank,
            semantic_weight,
            bm25_weight
        )
        
        # Get function data
        result = function_map.get(function_id)
        if result:
            # Add RRF score and rank info
            result.rrf_score = rrf_score
            result.semantic_rank = semantic_rank if semantic_rank > 0 else None
            result.bm25_rank = bm25_rank if bm25_rank > 0 else None
            
            scored_results.append(result)
    
    # Sort by RRF score (descending)
    scored_results.sort(key=lambda x: x.rrf_score, reverse=True)
    
    # Return top results
    return scored_results[:limit]