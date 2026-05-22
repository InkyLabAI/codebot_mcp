"""
Call Graph Ordering Utility

Orders functions based on their call relationships to show execution flow.
Supports two modes:
1. Local-only: Uses only search results (fast, no DB access)
2. PCST: Uses full repository graph to find bridge nodes (requires session)

Each entry point creates its own chain, and functions can appear in multiple chains.
"""
import math
import re
import uuid
from typing import List, Dict, Set, Deque, Optional, Tuple
from collections import defaultdict, deque

import networkx as nx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codebot_mcp.schemas.function import FunctionSearchResult, FunctionResponse
from codebot_mcp.db import Function, Repository
import logging
logger = logging.getLogger(__name__)

_TEST_FILE_RE = re.compile(
    r'(^|/)tests?/'          # directory named test/ or tests/
    r'|(^|/)test_[^/]+/'     # directory whose name starts with test_ (e.g. test_unstructured/)
    r'|/test_[^/]*\.py$'     # filename starting with test_ (e.g. test_base.py)
    r'|_test\.py$'           # filename ending with _test.py
    r'|(^|/)conftest\.py$',  # pytest conftest files
    re.IGNORECASE,
)
_TEST_ID_RE = re.compile(r'^test\.|\.test_|\.Test[A-Z]|^test_', re.IGNORECASE)


def _is_test_function(file_path: str, function_id: str) -> bool:
    return bool(_TEST_FILE_RE.search(file_path) or _TEST_ID_RE.search(function_id))

# =============================================================================
# PCST-based Call Graph (uses full repository graph)
# =============================================================================

async def build_weighted_call_graph(
    session: AsyncSession,
    repository_id: uuid.UUID
) -> Tuple[nx.DiGraph, Dict[str, Function], str]:
    """
    Build a weighted directed graph from all functions in the repository.

    Edge weight for u → v:
        base  = ln(in_degree(v) + 1) + 1   (penalize utility functions)
        discount:
          × 0.5  if caller and callee are in the same class
          × 0.7  if same module (file) but different class
          × 1.0  otherwise

    Structural discounts make Steiner/Dijkstra prefer paths through
    code that is structurally related (same class > same file > other).

    Args:
        session: Async database session
        repository_id: Repository UUID

    Returns:
        Tuple of (graph, func_map, repo_name) where:
        - graph: NetworkX DiGraph with weighted edges
        - func_map: Mapping from function_id to Function DB object
        - repo_name: Repository name for creating FunctionSearchResult
    """
    repo_id_str = str(repository_id)

    # Fetch repository name
    repo_stmt = select(Repository.name).where(Repository.id == repo_id_str)
    repo_result = await session.execute(repo_stmt)
    repo_name = repo_result.scalar_one_or_none() or "Unknown"

    # Fetch all functions
    stmt = select(Function).where(Function.repository_id == repo_id_str)
    result = await session.execute(stmt)
    all_functions = result.scalars().all()

    # Build mapping
    func_map: Dict[str, Function] = {f.function_id: f for f in all_functions}
    func_ids = set(func_map.keys())

    # First pass: count in-degrees (only for internal calls)
    in_degree: Dict[str, int] = defaultdict(int)
    for func in all_functions:
        for called_id in (func.calls or []):
            if called_id in func_ids:  # Only count internal calls
                in_degree[called_id] += 1

    # Build weighted graph
    G = nx.DiGraph()

    # Add all nodes
    for func_id in func_ids:
        G.add_node(func_id)

    # Add weighted edges with structural discounts
    for func in all_functions:
        caller_id = func.function_id
        for called_id in (func.calls or []):
            if called_id in func_ids:
                callee = func_map[called_id]

                # Base weight: penalize high-traffic utility functions
                base_weight = math.log(in_degree[called_id] + 1) + 1

                # Structural discount
                if (func.class_name and callee.class_name
                        and func.class_name == callee.class_name
                        and func.file_path == callee.file_path):
                    structural_factor = 0.5   # same class
                elif func.file_path == callee.file_path:
                    structural_factor = 0.7   # same module
                else:
                    structural_factor = 1.0   # cross-module

                G.add_edge(caller_id, called_id,
                           weight=base_weight * structural_factor)

    return G, func_map, repo_name


def _build_module_entries(
    func_ids: Set[str],
    dominant_file: str,
    func_map: Dict[str, Function],
    G: nx.DiGraph,
) -> Dict:
    """Build a community entry dict for a set of function IDs grouped under one module."""
    module_doc = ""
    for fid in func_ids:
        f = func_map.get(fid)
        if f and f.file_path == dominant_file and f.module_docstring:
            first_line = f.module_docstring.strip().split("\n")[0].strip()
            if first_line:
                module_doc = first_line
            break

    in_deg_local: Dict[str, int] = {fid: 0 for fid in func_ids if fid in G}
    for fid in func_ids:
        if fid not in G:
            continue
        for succ in G.successors(fid):
            if succ in func_ids and succ in in_deg_local:
                in_deg_local[succ] += 1

    entry_points = [fid for fid, deg in in_deg_local.items() if deg == 0]
    entry_points.sort(key=lambda fid: -G.out_degree(fid))

    functions = []
    for fid in entry_points[:5]:
        f = func_map.get(fid)
        if not f:
            continue
        name = f"{f.class_name}.{f.name}" if f.class_name else f.name
        functions.append({"name": name})

    return {"module": dominant_file, "module_docstring": module_doc, "functions": functions}


def _fallback_by_module(func_map: Dict[str, Function], G: nx.DiGraph) -> List[Dict]:
    """Group functions by file path when Leiden finds no non-trivial communities."""
    by_module: Dict[str, Set[str]] = defaultdict(set)
    for fid, f in func_map.items():
        if not _is_test_function(f.file_path, fid):
            by_module[f.file_path].add(fid)

    sorted_modules = sorted(by_module.items(), key=lambda x: -len(x[1]))
    result = []
    for module, fids in sorted_modules[:30]:
        result.append(_build_module_entries(fids, module, func_map, G))
    logger.info("[communities] fallback: grouped %d modules by file path", len(result))
    return result


def detect_communities(
    G: nx.DiGraph,
    func_map: Dict[str, Function],
) -> List[Dict]:
    """
    Detect graph communities via Louvain and return structured data per community.

    Groups communities that share the same dominant module. Falls back to
    grouping by file path when Louvain finds no non-trivial communities
    (common in codebases that call mostly external libraries).

    Returns:
        List of dicts with keys: module, module_docstring, functions.
    """
    # Exclude test functions before community detection
    func_map = {
        fid: f for fid, f in func_map.items()
        if not _is_test_function(f.file_path, fid)
    }

    if len(G) == 0 or not func_map:
        return _fallback_by_module(func_map, G) if func_map else []

    import igraph as ig
    import leidenalg

    undirected = G.to_undirected()
    nodes = list(undirected.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in undirected.edges()]
    weights = [undirected[u][v].get("weight", 1.0) for u, v in undirected.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=42,
    )
    communities = [{nodes[i] for i in community} for community in partition]
    communities = sorted(communities, key=len, reverse=True)

    non_trivial = sum(1 for c in communities if len(c) > 2)
    logger.info(
        "[communities] Leiden found %d clusters (%d with >2 nodes) in a graph of %d nodes",
        len(communities), non_trivial, len(G),
    )

    # Group communities by dominant module
    module_communities: Dict[str, List[Set[str]]] = defaultdict(list)
    for community in communities:
        if len(community) <= 2:
            continue
        members = [fid for fid in community if fid in func_map]
        if not members:
            continue
        file_counts: Dict[str, int] = defaultdict(int)
        for fid in members:
            file_counts[func_map[fid].file_path] += 1
        dominant_file = max(file_counts, key=file_counts.get)
        module_communities[dominant_file].append(set(community))

    if not module_communities:
        logger.info("[communities] no non-trivial Louvain clusters — falling back to module grouping")
        return _fallback_by_module(func_map, G)

    sorted_modules = sorted(
        module_communities.items(),
        key=lambda item: sum(len(c) for c in item[1]),
        reverse=True,
    )

    result: List[Dict] = []
    for dominant_file, community_groups in sorted_modules:
        merged: Set[str] = set()
        for c in community_groups:
            merged.update(c)
        result.append(_build_module_entries(merged, dominant_file, func_map, G))

    logger.info("[communities] merged into %d components (by dominant module)", len(result))
    return result


def find_steiner_bridges(
    G: nx.DiGraph,
    terminals: Set[str],
    local_chains: List[List[FunctionSearchResult]],
    cutoff: float = 5.0
) -> Set[str]:
    """
    Find bridge nodes that connect *different chains* of terminals.

    First assigns each terminal to a chain ID. Then only runs Dijkstra
    from boundary nodes (entry points / leaves of each chain) and only
    collects bridges from paths that reach a terminal in a *different* chain.

    This avoids:
    - Bridging between nodes already connected within the same chain
    - Running Dijkstra from every terminal (only boundary nodes)

    Args:
        G: Weighted directed graph
        terminals: Set of terminal node IDs (search results)
        local_chains: Pre-computed chains from order_functions_local
        cutoff: Maximum path weight to consider

    Returns:
        Set of bridge node IDs (intermediate nodes on shortest paths)
    """
    # Map each terminal to its chain index
    node_to_chain: Dict[str, int] = {}
    for chain_idx, chain in enumerate(local_chains):
        for f in chain:
            node_to_chain[f.function.function_id] = chain_idx

    # Find boundary nodes per chain: entry points (in-degree 0) and
    # leaves (out-degree 0) within the local chain subgraph
    boundary_nodes: Set[str] = set()
    for chain in local_chains:
        if len(chain) <= 1:
            # Single-node chain: the node itself is the boundary
            boundary_nodes.add(chain[0].function.function_id)
            continue

        chain_ids = {f.function.function_id for f in chain}
        in_deg = {fid: 0 for fid in chain_ids}
        out_deg = {fid: 0 for fid in chain_ids}
        for f in chain:
            fid = f.function.function_id
            for callee in (f.function.calls or []):
                if callee in chain_ids:
                    in_deg[callee] += 1
                    out_deg[fid] += 1

        for fid in chain_ids:
            if in_deg[fid] == 0 or out_deg[fid] == 0:
                boundary_nodes.add(fid)

    logger.info(
        "[bridges] %d chains, %d boundary nodes (of %d terminals)",
        len(local_chains), len(boundary_nodes), len(terminals),
    )

    # Run Dijkstra only from boundary nodes, collect bridges only for
    # paths that reach a terminal in a different chain
    bridges: Set[str] = set()

    for source in boundary_nodes:
        if source not in G:
            continue
        source_chain = node_to_chain.get(source)

        try:
            distances, paths = nx.single_source_dijkstra(
                G, source, cutoff=cutoff, weight='weight'
            )
        except nx.NetworkXError:
            continue

        for target in terminals:
            if target not in paths:
                continue
            # Only bridge across different chains
            if node_to_chain.get(target) == source_chain:
                continue
            path = paths[target]
            for node in path[1:-1]:
                bridges.add(node)

    logger.info("[bridges] found %d bridge nodes", len(bridges))
    return bridges


def _create_bridge_result(
    func: Function,
    repo_name: str
) -> FunctionSearchResult:
    """
    Convert a database Function to FunctionSearchResult for bridge nodes.

    Bridge nodes have similarity=0.0 and is_bridge=True.
    """
    function_response = FunctionResponse(
        id=func.id,
        repository_id=func.repository_id,
        function_id=func.function_id,
        name=func.name,
        file_path=func.file_path,
        class_name=func.class_name,
        nested=func.nested,
        code=func.code,
        docstring=func.docstring,
        start_line=func.start_line,
        end_line=func.end_line,
        parameters=func.parameters,
        decorators=func.decorators,
        return_type=func.return_type,
        calls=func.calls,
        created_at=func.created_at,
        has_embedding=False  # Bridge nodes may not have embeddings
    )

    return FunctionSearchResult(
        function=function_response,
        similarity=0.0,
        repository_name=repo_name,
        is_bridge=True
    )


async def order_functions_by_pcst(
    functions: List[FunctionSearchResult],
    session: AsyncSession,
    repository_id: uuid.UUID,
    cutoff: float = 5.0
) -> List[List[FunctionSearchResult]]:
    """
    Order functions using PCST-based Steiner Tree approach.

    Algorithm:
    1. Find local chains among search results (no DB access)
    2. Build weighted graph from full repository
    3. Find bridge nodes connecting *different* local chains
    4. Create induced subgraph with terminals + bridges
    5. Apply entry-point based chaining to the subgraph

    Args:
        functions: Search results (terminals)
        session: Async database session
        repository_id: Repository UUID
        cutoff: Maximum path weight for bridging (default 5.0)

    Returns:
        List of chains, each chain is a list of FunctionSearchResult
    """
    if len(functions) <= 1:
        return [functions] if functions else []

    # 1. Find local chains first (uses only edges between search results)
    local_chains = order_functions_local(functions)

    # If everything is already in one chain, no bridging needed
    if len(local_chains) <= 1:
        logger.info("[pcst] all %d results in a single local chain, skipping bridging", len(functions))
        return local_chains

    # 2. Build full weighted graph from repository
    G, func_map, repo_name = await build_weighted_call_graph(session, repository_id)

    # Terminal nodes are our search results
    terminals = {f.function.function_id for f in functions}
    terminal_results = {f.function.function_id: f for f in functions}

    # 3. Find bridge nodes between different chains
    bridges = find_steiner_bridges(G, terminals, local_chains, cutoff)

    # Create FunctionSearchResult objects for bridge nodes
    bridge_results: Dict[str, FunctionSearchResult] = {}
    for bridge_id in bridges:
        if bridge_id in func_map and bridge_id not in terminals:
            bridge_results[bridge_id] = _create_bridge_result(
                func_map[bridge_id], repo_name
            )

    # Combine terminals and bridges
    all_results = {**terminal_results, **bridge_results}
    all_func_ids = set(all_results.keys())

    # 4. Build subgraph edges (only between included nodes)
    subgraph: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = {fid: 0 for fid in all_func_ids}

    for func_id in all_func_ids:
        if func_id in func_map:
            calls = func_map[func_id].calls or []
        else:
            # For terminal nodes, use the calls from search result
            calls = all_results[func_id].function.calls or []

        for called_id in calls:
            if called_id in all_func_ids:
                subgraph[func_id].append(called_id)
                in_degree[called_id] += 1

    # 5. Apply entry-point based chaining to the subgraph
    return _build_chains_from_graph(subgraph, in_degree, all_results, all_func_ids)


def _build_chains_from_graph(
    graph: Dict[str, List[str]],
    in_degree: Dict[str, int],
    result_map: Dict[str, FunctionSearchResult],
    func_ids: Set[str]
) -> List[List[FunctionSearchResult]]:
    """
    Build chains from a call graph using entry-point based chaining.

    Each entry point (in-degree 0) creates its own chain.
    Functions can appear in multiple chains.

    Args:
        graph: Adjacency list (caller -> [callees])
        in_degree: In-degree count for each node
        result_map: Mapping from function_id to FunctionSearchResult
        func_ids: Set of all function IDs in the graph

    Returns:
        List of chains sorted by length (longest first)
    """
    # Find entry points (in-degree 0)
    entry_points = [fid for fid in func_ids if in_degree[fid] == 0]

    # Sort entry points by original order (non-bridges first, then by position)
    def entry_sort_key(fid: str) -> Tuple[int, float]:
        result = result_map.get(fid)
        if result and not result.is_bridge:
            return (0, -result.similarity)  # Non-bridges first, by similarity
        return (1, 0)  # Bridges last

    entry_points.sort(key=entry_sort_key)

    # Build chains
    chains: List[List[FunctionSearchResult]] = []
    covered: Set[str] = set()

    for entry_id in entry_points:
        reachable = _find_reachable(entry_id, graph, func_ids)
        covered.update(reachable)

        if len(reachable) > 1:
            sorted_chain = _dfs_preorder_sort(entry_id, reachable, graph, result_map)
            chains.append(sorted_chain)
        else:
            chains.append([result_map[entry_id]])

    # Handle cycles with no entry points
    uncovered = func_ids - covered
    while uncovered:
        # Pick first uncovered (prefer non-bridges)
        start_id = min(uncovered, key=lambda x: (result_map[x].is_bridge, -result_map[x].similarity))

        reachable = _find_reachable(start_id, graph, func_ids)
        reachable = reachable & uncovered
        covered.update(reachable)
        uncovered -= reachable

        if len(reachable) > 1:
            sorted_chain = _dfs_preorder_sort(start_id, reachable, graph, result_map)
            chains.append(sorted_chain)
        else:
            chains.append([result_map[start_id]])

    # Sort chains by score
    chains.sort(key=lambda c: -_chain_score(c))

    return chains


def _chain_score(
    chain: List[FunctionSearchResult],
    graph: Optional[Dict[str, List[str]]] = None,
    mu: float = 1.0,
    lam: float = 0.3
) -> float:
    """
    Score a chain using a "Net Profit" graph scoring approach.

    Formula: Sum(NodeScores) + mu * Avg(EdgeCohesion) - lam * (N - 1)

    - NodeScores: similarity of each function node
    - EdgeCohesion: for each edge A->B in the chain, Score(A) * Score(B),
      averaged over the number of edges to avoid biasing toward longer chains
    - N: number of nodes
    - mu (1.0): rewards connected relevance
    - lam (0.3): complexity tax penalizing length

    If graph is not provided, edges are derived from each node's calls list.
    """
    n = len(chain)
    if n == 0:
        return 0.0

    # Build a quick lookup: function_id -> similarity
    sim = {r.function.function_id: r.similarity for r in chain}
    chain_ids = set(sim.keys())

    # Sum of node scores
    node_sum = sum(sim.values())

    # Average edge cohesion (only edges within the chain)
    edge_sum = 0.0
    edge_count = 0
    for r in chain:
        caller_id = r.function.function_id
        # Use provided graph if available, otherwise fall back to node's calls
        callees = graph.get(caller_id, []) if graph else (r.function.calls or [])
        for callee_id in callees:
            if callee_id in chain_ids:
                edge_sum += sim[caller_id] * sim[callee_id]
                edge_count += 1

    avg_edge = edge_sum / edge_count if edge_count > 0 else 0.0

    return node_sum + mu * avg_edge - lam * (n - 1)


# =============================================================================
# Local-only Call Graph (no DB access, uses only search results)
# =============================================================================

def order_functions_local(
    functions: List[FunctionSearchResult]
) -> List[List[FunctionSearchResult]]:
    """
    Order functions based on call graph relationships using only search results.

    This is the fallback when no repository_id is provided.

    Key behaviors:
    - Each entry point (function with no callers in result set) creates its own chain
    - Functions can appear in multiple chains if called from different entry points
    - Chains are sorted by length (longest first)

    Args:
        functions: List of search results to order

    Returns:
        List of chains, where each chain is a list of functions in execution order.
    """
    if len(functions) <= 1:
        return [functions] if functions else []

    # Create mapping: function_id -> FunctionSearchResult
    func_map = {f.function.function_id: f for f in functions}
    func_ids = set(func_map.keys())

    # Build call graph with DIRECT edges only
    graph: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = {fid: 0 for fid in func_ids}

    for result in functions:
        caller_id = result.function.function_id
        calls = result.function.calls or []

        for called_id in calls:
            if called_id in func_ids:
                graph[caller_id].append(called_id)
                in_degree[called_id] += 1

    return _build_chains_from_graph(graph, in_degree, func_map, func_ids)


# =============================================================================
# Shared Helper Functions
# =============================================================================

def _find_reachable(
    start_id: str,
    graph: Dict[str, List[str]],
    func_ids: Set[str]
) -> Set[str]:
    """
    Find all functions reachable from a starting point (forward edges only).

    Args:
        start_id: Starting function ID
        graph: Call graph (caller -> [callees])
        func_ids: Set of valid function IDs

    Returns:
        Set of all reachable function IDs (including start_id)
    """
    reachable: Set[str] = set()
    queue: Deque[str] = deque([start_id])

    while queue:
        node_id = queue.popleft()

        if node_id in reachable:
            continue

        reachable.add(node_id)

        for callee_id in graph[node_id]:
            if callee_id in func_ids and callee_id not in reachable:
                queue.append(callee_id)

    return reachable


def _dfs_preorder_sort(
    entry_id: str,
    reachable: Set[str],
    graph: Dict[str, List[str]],
    result_map: Dict[str, FunctionSearchResult]
) -> List[FunctionSearchResult]:
    """
    Sort reachable functions using DFS pre-order traversal from entry point.

    This ensures execution flow order: caller before callees, depth-first.

    Args:
        entry_id: Entry point function ID
        reachable: Set of reachable function IDs
        graph: Call graph
        result_map: Mapping from function_id to FunctionSearchResult

    Returns:
        List of FunctionSearchResult in DFS pre-order
    """
    visited: Set[str] = set()
    result: List[str] = []

    def dfs(func_id: str):
        if func_id in visited or func_id not in reachable:
            return

        visited.add(func_id)
        result.append(func_id)

        # Visit callees in original call order (execution flow)
        callees = [c for c in graph[func_id] if c in reachable and c not in visited]

        for callee_id in callees:
            dfs(callee_id)

    # Start DFS from entry point
    dfs(entry_id)

    # Handle remaining nodes (cycles)
    remaining = reachable - visited
    if remaining:
        remaining_sorted = sorted(remaining, key=lambda fid: (
            result_map[fid].is_bridge, -result_map[fid].similarity
        ))
        for func_id in remaining_sorted:
            dfs(func_id)

    return [result_map[func_id] for func_id in result]


def _topological_sort_chain(
    nodes: Set[str],
    graph: Dict[str, List[str]],
    result_map: Dict[str, FunctionSearchResult]
) -> List[FunctionSearchResult]:
    """
    Sort nodes using Kahn's algorithm (topological BFS).

    Unlike DFS preorder, this guarantees ALL callers appear before their callees
    even when multiple entry points share downstream nodes. Tie-breaking among
    ready nodes uses call position (the order callees appear in their parent's
    call list) to preserve execution flow, falling back to (is_bridge, similarity)
    for entry points with no parent.

    Falls back to appending remaining nodes (cycles) sorted by priority.

    Args:
        nodes: Set of node IDs to sort
        graph: Call graph adjacency list
        result_map: Mapping from function_id to FunctionSearchResult

    Returns:
        List of FunctionSearchResult in topological order
    """
    # Compute in-degree within the node set
    in_deg = {nid: 0 for nid in nodes}
    for nid in nodes:
        for callee in graph.get(nid, []):
            if callee in nodes:
                in_deg[callee] += 1

    # Track call position: node -> (parent_emit_order, index_in_parent_calls)
    # Used to preserve execution order when breaking ties among ready nodes.
    call_position: Dict[str, Tuple[int, int]] = {}
    emit_order = 0

    def sort_key(nid: str):
        r = result_map[nid]
        # Primary: call position (preserves execution order within a caller)
        # Secondary: bridge status and similarity (for entry points without a parent)
        pos = call_position.get(nid, (float('inf'), float('inf')))
        return (pos, r.is_bridge, -r.similarity)

    # Seed with all zero-in-degree nodes
    ready = sorted(
        [nid for nid in nodes if in_deg[nid] == 0],
        key=sort_key
    )

    result: List[str] = []
    visited: Set[str] = set()

    while ready:
        nid = ready.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        result.append(nid)

        # Decrease in-degree of callees, promote newly ready ones
        newly_ready = []
        for i, callee in enumerate(graph.get(nid, [])):
            if callee in nodes and callee not in visited:
                # Track earliest call position for tie-breaking
                new_pos = (emit_order, i)
                if callee not in call_position or new_pos < call_position[callee]:
                    call_position[callee] = new_pos
                in_deg[callee] -= 1
                if in_deg[callee] == 0:
                    newly_ready.append(callee)

        emit_order += 1

        if newly_ready:
            # Merge into ready list maintaining sort order
            merged = sorted(ready + newly_ready, key=sort_key)
            ready = merged

    # Handle remaining nodes (cycles)
    remaining = nodes - visited
    if remaining:
        for nid in sorted(remaining, key=sort_key):
            if nid not in visited:
                visited.add(nid)
                result.append(nid)

    return [result_map[nid] for nid in result]


def _merge_overlapping_chains(
    chains: List[List[FunctionSearchResult]],
    graph: Dict[str, List[str]],
    result_map: Dict[str, FunctionSearchResult],
    overlap_threshold: float = 0.6
) -> List[List[FunctionSearchResult]]:
    """
    Merge chains that share a significant portion of their nodes.

    Uses containment ratio: |A ∩ B| / min(|A|, |B|) >= threshold.
    This catches small chains fully contained in larger ones (unlike Jaccard).

    Overlapping chains are grouped via Union-Find (transitive closure),
    then the common core (nodes in 2+ chains) is extracted and sorted
    via DFS preorder. The merged chain is only accepted if its score
    >= the best original.

    Args:
        chains: List of chains to potentially merge
        graph: Call graph adjacency list
        result_map: Mapping from function_id to FunctionSearchResult
        overlap_threshold: Minimum containment ratio to trigger merge (default 0.6)

    Returns:
        Merged and re-sorted list of chains
    """
    if len(chains) <= 1:
        return chains

    # Build node sets per chain; track which are mergeable (size > 1)
    chain_node_sets: List[Set[str]] = []
    for chain in chains:
        chain_node_sets.append({f.function.function_id for f in chain})

    n = len(chains)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise compare mergeable chains (skip single-node chains)
    for i in range(n):
        if len(chain_node_sets[i]) <= 1:
            continue
        for j in range(i + 1, n):
            if len(chain_node_sets[j]) <= 1:
                continue
            intersection = len(chain_node_sets[i] & chain_node_sets[j])
            min_size = min(len(chain_node_sets[i]), len(chain_node_sets[j]))
            if intersection / max(min_size, 1) >= overlap_threshold:
                union(i, j)

    # Group chains by their root
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Build merged chains
    merged_chains: List[List[FunctionSearchResult]] = []

    for group_indices in groups.values():
        if len(group_indices) == 1:
            # No merge needed, keep original chain
            merged_chains.append(chains[group_indices[0]])
            continue

        # Extract common core: nodes appearing in 2+ chains of the group
        node_counts: Dict[str, int] = defaultdict(int)
        for idx in group_indices:
            for nid in chain_node_sets[idx]:
                node_counts[nid] += 1

        core_nodes = {nid for nid, count in node_counts.items() if count >= 2}

        # Fallback: if no shared core (shouldn't happen), keep originals
        if not core_nodes:
            merged_chains.extend(chains[idx] for idx in group_indices)
            continue

        # Use DFS preorder to preserve execution flow order
        if len(core_nodes) > 1:
            # Find entry point: in-degree 0 within core, prefer non-bridge
            core_in_deg = {nid: 0 for nid in core_nodes}
            for nid in core_nodes:
                for callee in graph.get(nid, []):
                    if callee in core_nodes:
                        core_in_deg[callee] += 1

            entry = min(
                core_nodes,
                key=lambda nid: (
                    core_in_deg[nid] > 0,         # prefer in-degree 0
                    result_map[nid].is_bridge,     # prefer non-bridge
                    -result_map[nid].similarity,   # prefer high score
                ),
            )
            merged_chain = _dfs_preorder_sort(entry, core_nodes, graph, result_map)
        else:
            merged_chain = [result_map[next(iter(core_nodes))]]

        # Only merge if the merged chain scores higher than the originals
        original_chains = [chains[idx] for idx in group_indices]
        best_original_score = max(_chain_score(c) for c in original_chains)
        merged_score = _chain_score(merged_chain)

        if merged_score >= best_original_score:
            merged_chains.append(merged_chain)
            logger.info(
                "[merge] merged %d chains (%s nodes) -> %d nodes "
                "(score %.3f >= best original %.3f)",
                len(group_indices),
                [len(c) for c in original_chains],
                len(merged_chain),
                merged_score,
                best_original_score,
            )
        else:
            # Keep originals — merge would degrade quality
            merged_chains.extend(original_chains)
            logger.info(
                "[merge] skipped merge of %d chains "
                "(merged score %.3f < best original %.3f)",
                len(group_indices),
                merged_score,
                best_original_score,
            )

    # Re-sort all chains by chain score (no graph arg so scoring is
    # consistent with downstream callers that don't have the graph)
    merged_chains.sort(key=lambda c: -_chain_score(c))

    return merged_chains


# =============================================================================
# 1-Hop Neighbor Expansion (for GraphRAG v2)
# =============================================================================


def ensure_topological_order(
    chains: List[List[FunctionSearchResult]],
) -> List[List[FunctionSearchResult]]:
    """
    Re-sort each chain using Kahn's algorithm (topological BFS).

    DFS preorder does NOT guarantee all callers appear before their callees
    when multiple paths converge on the same node. This function re-sorts
    each chain so that every caller precedes its callees.

    Args:
        chains: List of chains (possibly in DFS preorder)

    Returns:
        Same chains with each one in topological order
    """
    sorted_chains: List[List[FunctionSearchResult]] = []

    for chain in chains:
        if len(chain) <= 2:
            sorted_chains.append(chain)
            continue

        chain_map = {f.function.function_id: f for f in chain}
        chain_ids = set(chain_map.keys())

        # Build call graph within the chain
        graph: Dict[str, List[str]] = defaultdict(list)
        for f in chain:
            fid = f.function.function_id
            for callee in (f.function.calls or []):
                if callee in chain_ids:
                    graph[fid].append(callee)

        sorted_chains.append(
            _topological_sort_chain(chain_ids, graph, chain_map)
        )

    return sorted_chains


def prune_low_score_nodes(
    chains: List[List[FunctionSearchResult]],
    threshold: float = 0.3,
) -> List[List[FunctionSearchResult]]:
    """
    Iteratively remove low-scoring leaf nodes from chains.

    A leaf is a node with no callees within the current chain (out-degree 0).
    Only leaves below *threshold* are removed. Nodes with callees are never
    pruned — they provide structural connectivity regardless of their score.
    Pruning repeats until no more prunable leaves remain, since removing a
    leaf may expose another.

    Args:
        chains: List of chains to prune
        threshold: Leaves with similarity below this are removed

    Returns:
        Pruned list of chains (empty chains dropped)
    """
    pruned_chains: List[List[FunctionSearchResult]] = []

    for chain in chains:
        if len(chain) <= 1:
            pruned_chains.append(chain)
            continue

        result_map = {f.function.function_id: f for f in chain}
        kept_ids: Set[str] = set(result_map.keys())

        # Iteratively prune low-score leaves
        while True:
            out_deg: Dict[str, int] = {fid: 0 for fid in kept_ids}
            for fid in kept_ids:
                for callee in (result_map[fid].function.calls or []):
                    if callee in kept_ids:
                        out_deg[fid] += 1

            to_remove = {
                fid for fid in kept_ids
                if out_deg[fid] == 0
                and result_map[fid].similarity < threshold
            }
            if not to_remove:
                break
            kept_ids -= to_remove

        if not kept_ids:
            continue

        kept = [f for f in chain if f.function.function_id in kept_ids]
        removed_count = len(chain) - len(kept)
        if removed_count > 0:
            logger.info(
                "[prune] chain %d -> %d kept (%d leaf nodes removed)",
                len(chain), len(kept), removed_count,
            )
        pruned_chains.append(kept)

    return pruned_chains


async def expand_adaptive_neighbors(
    seed_results: List[FunctionSearchResult],
    session: AsyncSession,
    repository_id: uuid.UUID,
    scored_results: Optional[List[FunctionSearchResult]] = None,
    prebuilt_graph: Optional[Tuple[nx.DiGraph, Dict[str, Function], str]] = None,
) -> Tuple[List[FunctionSearchResult], Dict[str, FunctionSearchResult],
           nx.DiGraph, Dict[str, Function], str]:
    """
    Expand seed results with adaptive hop depth based on structural proximity.

    - All seeds get 1-hop expansion (direct callers + callees)
    - 1-hop neighbors that share a class or module with any seed get a 2nd hop
      (their callers + callees are also added)

    Same-class = same file_path AND same class_name (non-None).
    Same-module = same file_path.

    Neighbors that appear in `scored_results` keep their original score.
    All other neighbors become bridges (is_bridge=True, similarity=0.0).

    Args:
        seed_results: Search results to expand (the top-N seed)
        session: Async database session
        repository_id: Repository UUID
        scored_results: Full set of reranked results (superset of seed).
            Neighbors found here reuse their scored FunctionSearchResult.
        prebuilt_graph: Optional (G, func_map, repo_name) tuple to skip
            rebuilding the graph. If provided, session/repository_id are
            not used for graph construction.

    Returns:
        Tuple of (expanded_results, result_map, G, func_map, repo_name):
        - expanded_results: seed + neighbor FunctionSearchResults
        - result_map: mapping from function_id to FunctionSearchResult
        - G: weighted DiGraph (for Steiner bridging)
        - func_map: function_id to Function DB object
        - repo_name: repository name
    """
    if not seed_results:
        return [], {}

    # Build lookup of already-scored functions (from the full reranked set)
    scored_map: Dict[str, FunctionSearchResult] = {}
    if scored_results:
        scored_map = {f.function.function_id: f for f in scored_results}

    # Use pre-built graph if provided, otherwise build from DB
    if prebuilt_graph:
        G, func_map, repo_name = prebuilt_graph
    else:
        G, func_map, repo_name = await build_weighted_call_graph(session, repository_id)

    # Build reverse caller index once: callee_id -> {caller_ids}
    reverse_callers: Dict[str, Set[str]] = defaultdict(set)
    for fid, func in func_map.items():
        for called_id in (func.calls or []):
            if called_id in func_map:
                reverse_callers[called_id].add(fid)

    # Collect seed structural context for proximity checks
    seed_classes: Set[Tuple[str, str]] = set()  # (file_path, class_name)
    seed_modules: Set[str] = set()              # file_path
    for r in seed_results:
        seed_modules.add(r.function.file_path)
        if r.function.class_name:
            seed_classes.add((r.function.file_path, r.function.class_name))

    seed_ids = {f.function.function_id for f in seed_results}
    result_map: Dict[str, FunctionSearchResult] = {
        f.function.function_id: f for f in seed_results
    }

    def _is_structurally_close_to_seed(fid: str) -> bool:
        """Check if a function shares a class or module with any seed."""
        func = func_map.get(fid)
        if not func:
            return False
        # Same class (strongest signal)
        if func.class_name and (func.file_path, func.class_name) in seed_classes:
            return True
        # Same module / file
        return func.file_path in seed_modules

    # --- Phase 1: 1-hop neighbors of seeds ---
    hop1_ids: Set[str] = set()

    # Callees of seeds
    for r in seed_results:
        for callee_id in (r.function.calls or []):
            if callee_id in func_map and callee_id not in seed_ids:
                hop1_ids.add(callee_id)

    # Callers of seeds (via reverse index)
    for sid in seed_ids:
        for caller_id in reverse_callers.get(sid, set()):
            if caller_id not in seed_ids:
                hop1_ids.add(caller_id)

    # --- Phase 2: 2nd hop from structurally close 1-hop neighbors ---
    hop2_ids: Set[str] = set()
    already_included = seed_ids | hop1_ids
    structurally_close_count = 0

    for nid in hop1_ids:
        if not _is_structurally_close_to_seed(nid):
            continue

        structurally_close_count += 1
        neighbor_func = func_map[nid]

        # Callees of this structurally close neighbor
        for callee_id in (neighbor_func.calls or []):
            if callee_id in func_map and callee_id not in already_included:
                hop2_ids.add(callee_id)

        # Callers of this structurally close neighbor
        for caller_id in reverse_callers.get(nid, set()):
            if caller_id not in already_included:
                hop2_ids.add(caller_id)

    # --- Add all neighbors to result_map ---
    all_neighbor_ids = hop1_ids | hop2_ids
    scored_reused = 0
    bridge_added = []
    for nid in all_neighbor_ids:
        if nid in scored_map:
            result_map[nid] = scored_map[nid]
            scored_reused += 1
        elif nid in func_map:
            result_map[nid] = _create_bridge_result(func_map[nid], repo_name)
            hop_type = "hop2" if nid in hop2_ids and nid not in hop1_ids else "hop1"
            bridge_added.append((nid, hop_type))

    expanded = list(result_map.values())

    logger.info(
        "[adaptive] %d seeds + %d hop1 (%d same-class/module -> %d hop2) "
        "+ %d scored reused = %d expanded",
        len(seed_ids), len(hop1_ids), structurally_close_count, len(hop2_ids),
        scored_reused, len(expanded),
    )
    if bridge_added:
        for bid, hop_type in bridge_added:
            logger.info("[adaptive] bridge added (%s): %s", hop_type, bid)

    return expanded, result_map, G, func_map, repo_name


# =============================================================================
# Main Entry Point
# =============================================================================

async def order_functions_by_call_graph(
    functions: List[FunctionSearchResult],
    session: Optional[AsyncSession] = None,
    repository_id: Optional[uuid.UUID] = None,
    max_hops: int = 1,  # Kept for backward compatibility
    cutoff: float = 5.0
) -> List[List[FunctionSearchResult]]:
    """
    Order functions by call graph relationships.

    If repository_id is provided with a session, uses PCST algorithm to find
    bridge nodes from the full repository graph. Otherwise uses local-only
    algorithm with just the search results.

    Args:
        functions: List of search results to order
        session: Optional async database session (required for PCST)
        repository_id: Optional repository UUID (required for PCST)
        max_hops: Unused, kept for backward compatibility
        cutoff: Maximum path weight for PCST bridging (default 5.0)

    Returns:
        List of chains, where each chain is a list of functions in execution order.
        Functions may have is_bridge=True if they are discovered bridge nodes.
    """
    if repository_id and session:
        # Use PCST algorithm with full repository graph
        return await order_functions_by_pcst(functions, session, repository_id, cutoff)
    else:
        # Fall back to local-only algorithm
        return order_functions_local(functions)


# =============================================================================
# Statistics
# =============================================================================

def get_call_chain_stats(functions: List[FunctionSearchResult]) -> Dict:
    """
    Get statistics about call chains in the result set.

    Useful for debugging and understanding the call structure.

    Returns:
        Dictionary with:
        - num_chains: Number of chains (one per entry point)
        - chain_sizes: List of sizes for each chain
        - total_internal_calls: Number of calls between functions in result
        - entry_points: Number of entry points (in-degree 0)
        - bridge_count: Number of bridge nodes
    """
    if not functions:
        return {
            'num_chains': 0,
            'chain_sizes': [],
            'total_internal_calls': 0,
            'entry_points': 0,
            'bridge_count': 0
        }

    func_ids = {f.function.function_id for f in functions}

    # Build graph and count in-degrees
    graph = defaultdict(list)
    in_degree = {fid: 0 for fid in func_ids}
    internal_calls = 0

    for result in functions:
        caller_id = result.function.function_id
        calls = result.function.calls or []

        for called_id in calls:
            if called_id in func_ids:
                graph[caller_id].append(called_id)
                in_degree[called_id] += 1
                internal_calls += 1

    # Count entry points
    entry_points = [fid for fid in func_ids if in_degree[fid] == 0]

    # Calculate chain sizes
    chain_sizes = []
    for entry_id in entry_points:
        reachable = _find_reachable(entry_id, graph, func_ids)
        chain_sizes.append(len(reachable))

    # Count bridge nodes
    bridge_count = sum(1 for f in functions if f.is_bridge)

    return {
        'num_chains': len(entry_points),
        'chain_sizes': sorted(chain_sizes, reverse=True),
        'total_internal_calls': internal_calls,
        'entry_points': len(entry_points),
        'bridge_count': bridge_count
    }
