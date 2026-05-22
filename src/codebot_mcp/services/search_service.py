"""Search service using numpy for vector search and rank_bm25 for BM25."""

import re
import uuid
import logging
from collections import defaultdict
from typing import List, Optional, Tuple, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codebot_mcp.db import Function, FunctionEmbedding, Repository
from codebot_mcp.services.embedding_service import embedding_service
from codebot_mcp.schemas.function import FunctionSearchResult, FunctionResponse
from codebot_mcp.utils.call_graph import (
    order_functions_by_call_graph, get_call_chain_stats, _chain_score,
    expand_adaptive_neighbors, order_functions_local, prune_low_score_nodes,
    find_steiner_bridges, _create_bridge_result, build_weighted_call_graph,
    _merge_overlapping_chains, detect_communities,
)
from codebot_mcp.utils.bm25_utils import merge_results_with_rrf, split_identifier

logger = logging.getLogger(__name__)


def _summarize_communities(community_data: list[dict]) -> str:
    """Build a plain-text codebase summary from detect_communities() output.

    No LLM required — formats the structured community data directly into
    the same kind of component list the reranker uses as context.
    """
    if not community_data:
        return ""
    lines = []
    for c in community_data:
        module = c["module"]
        if c["module_docstring"]:
            lines.append(f"{module}: {c['module_docstring']}")
        else:
            lines.append(module)
        for fn in c["functions"]:
            lines.append(f"  - {fn['name']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test-exclusion patterns (same as backend)
# ---------------------------------------------------------------------------
_TEST_FILE_RE = re.compile(
    r'(^|/)tests?/'
    r'|(^|/)test_[^/]+/'
    r'|/test_[^/]*\.py$'
    r'|_test\.py$'
    r'|(^|/)conftest\.py$',
    re.IGNORECASE,
)
_TEST_FUNC_RE = re.compile(r'^test_|\.test_|\.Test[A-Z]', re.IGNORECASE)


def _is_test_function(func: Function) -> bool:
    """Check if a function is a test function based on file path and function_id."""
    if _TEST_FILE_RE.search(func.file_path):
        return True
    if _TEST_FUNC_RE.search(func.function_id):
        return True
    return False


# ---------------------------------------------------------------------------
# Reranker document builder
# ---------------------------------------------------------------------------

def _normalize_id(name: str) -> str:
    """Split a code identifier into natural language words for the reranker."""
    return " ".join(split_identifier(name))


def _rerank_document(f) -> str:
    """Build an enriched reranker document with structural context."""
    func = f.function
    parts = []

    parts.append(f"# {func.file_path}")
    if func.module_docstring:
        parts.append(f'"""{func.module_docstring}"""')

    if func.class_name:
        parts.append(f"class {_normalize_id(func.class_name)}:")
        if func.class_docstring:
            parts.append(f'    """{func.class_docstring}"""')

    params_str = ""
    if func.parameters:
        param_parts = []
        for p in func.parameters:
            name = _normalize_id(p["name"]) if p.get("name") else ""
            if p.get("type"):
                name += f": {_normalize_id(p['type'])}"
            if name:
                param_parts.append(name)
        params_str = ", ".join(param_parts)
    ret = f" -> {_normalize_id(func.return_type)}" if func.return_type else ""
    parts.append(f"def {_normalize_id(func.name)}({params_str}){ret}:")

    if func.docstring:
        summary = func.docstring.strip().split('\n')[0].strip()
        parts.append(f'    """{summary}"""')
    else:
        parts.append(func.code)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: DB Function -> FunctionResponse
# ---------------------------------------------------------------------------

def _func_to_response(func: Function, has_embedding: bool = False) -> FunctionResponse:
    """Convert a DB Function model to a FunctionResponse schema."""
    return FunctionResponse(
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
        module_docstring=func.module_docstring,
        class_docstring=func.class_docstring,
        created_at=func.created_at,
        has_embedding=has_embedding,
    )


# ---------------------------------------------------------------------------
# Search Service
# ---------------------------------------------------------------------------

class SearchService:
    """Search service using numpy vectors and rank_bm25."""

    async def search_functions(
        self,
        session: AsyncSession,
        query: str,
        repository_id: Optional[uuid.UUID] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        exclude_tests: bool = False,
    ) -> List[FunctionSearchResult]:
        """Semantic search using numpy cosine similarity."""
        # Generate query embedding
        query_embedding = np.array(embedding_service.embed_query(query), dtype=np.float32)

        # Load all functions + embeddings from DB
        stmt = (
            select(Function, FunctionEmbedding.embedding_blob, Repository.name.label("repo_name"))
            .join(FunctionEmbedding, Function.id == FunctionEmbedding.function_id)
            .join(Repository, Function.repository_id == Repository.id)
        )
        if repository_id:
            stmt = stmt.where(Function.repository_id == str(repository_id))

        result = await session.execute(stmt)
        rows = result.all()

        if not rows:
            return []

        # Filter test functions and compute cosine similarity
        candidates = []
        for func, emb_blob, repo_name in rows:
            if exclude_tests and _is_test_function(func):
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            # Cosine similarity
            dot = np.dot(query_embedding, emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                similarity = 0.0
            else:
                similarity = float(dot / (norm_q * norm_e))

            if similarity >= similarity_threshold:
                candidates.append((func, repo_name, similarity))

        # Sort by similarity descending, take top_k
        candidates.sort(key=lambda x: -x[2])
        candidates = candidates[:top_k]

        search_results = []
        for func, repo_name, similarity in candidates:
            func_response = _func_to_response(func, has_embedding=True)
            search_results.append(
                FunctionSearchResult(
                    function=func_response,
                    similarity=round(similarity, 4),
                    repository_name=repo_name,
                )
            )

        return search_results

    async def bm25_search_functions(
        self,
        session: AsyncSession,
        query: str,
        repository_id: Optional[uuid.UUID] = None,
        top_k: Optional[int] = None,
        exclude_tests: bool = False,
    ) -> List[FunctionSearchResult]:
        """BM25 search using rank_bm25 library."""
        # Load all functions with search_vector text
        stmt = (
            select(Function, Repository.name.label("repo_name"))
            .join(Repository, Function.repository_id == Repository.id)
            .where(Function.search_vector.isnot(None))
        )
        if repository_id:
            stmt = stmt.where(Function.repository_id == str(repository_id))

        # Check embedding existence with a subquery
        emb_stmt = select(FunctionEmbedding.function_id)
        emb_result = await session.execute(emb_stmt)
        embedded_ids = {row[0] for row in emb_result.all()}

        result = await session.execute(stmt)
        rows = result.all()

        if not rows:
            return []

        # Filter test functions
        filtered = []
        for func, repo_name in rows:
            if exclude_tests and _is_test_function(func):
                continue
            filtered.append((func, repo_name))

        if not filtered:
            return []

        # Tokenize documents and query (same splitting as indexing)
        corpus = []
        for func, _ in filtered:
            tokens = func.search_vector.lower().split()
            corpus.append(tokens)

        # Apply same identifier splitting as the original
        query_tokens = []
        for word in query.split():
            if '_' in word:
                query_tokens.append(word.replace('_', ''))
            query_tokens.extend(split_identifier(word))
        # Deduplicate while preserving order
        seen = set()
        unique_tokens = []
        for t in query_tokens:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique_tokens.append(low)

        # Build BM25 index and score
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(unique_tokens)

        # Pair scores with functions, filter zero scores
        scored = []
        for i, (func, repo_name) in enumerate(filtered):
            if scores[i] > 0:
                scored.append((func, repo_name, float(scores[i])))

        # Sort by score descending
        scored.sort(key=lambda x: -x[2])
        if top_k is not None:
            scored = scored[:top_k]

        results = []
        for func, repo_name, bm25_score in scored:
            func_response = _func_to_response(func, has_embedding=(func.id in embedded_ids))
            results.append(
                FunctionSearchResult(
                    function=func_response,
                    similarity=round(bm25_score, 4),
                    repository_name=repo_name,
                )
            )

        return results

    async def hybrid_search_functions(
        self,
        session: AsyncSession,
        query: str,
        repository_id: Optional[uuid.UUID] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        semantic_weight: float = 3.0,
        bm25_weight: float = 1.0,
        exclude_tests: bool = False,
    ) -> List[FunctionSearchResult]:
        """Hybrid search combining semantic and BM25 using weighted RRF."""
        # Run both searches
        semantic_results = []
        try:
            semantic_results = await self.search_functions(
                session=session,
                query=query,
                repository_id=repository_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                exclude_tests=exclude_tests,
            )
        except Exception as e:
            logger.warning("Semantic search failed, falling back to BM25 only: %s", e)

        bm25_results = await self.bm25_search_functions(
            session=session,
            query=query,
            repository_id=repository_id,
            exclude_tests=exclude_tests,
        )

        merged_results = merge_results_with_rrf(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            limit=top_k,
        )

        return merged_results

    async def graphrag_search(
        self,
        session: AsyncSession,
        query: str,
        repository_id: uuid.UUID,
        exclude_tests: bool = False,
    ) -> Tuple[List[FunctionSearchResult], List[List[FunctionSearchResult]]]:
        """GraphRAG pipeline dispatcher. Delegates to v2."""
        return await self.graphrag_search_v2(
            session=session,
            query=query,
            repository_id=repository_id,
            exclude_tests=exclude_tests,
        )

    async def graphrag_search_v2(
        self,
        session: AsyncSession,
        query: str,
        repository_id: uuid.UUID,
        exclude_tests: bool = False,
    ) -> Tuple[List[FunctionSearchResult], List[List[FunctionSearchResult]]]:
        """
        GraphRAG v2: rerank-first -> adaptive expand -> Steiner bridge -> chain.

        Steps:
        1. Hybrid search (100 candidates)
        2. Rerank all 100 in a single Voyage API call
        3. Select top-N seed by reranker score
        4. Adaptive expansion (1-hop + 2-hop for same-class/module)
        5. Local chaining to identify disconnected clusters
        6. Steiner bridging to connect clusters
        7. Rerank all bridge nodes in one batch
        8. Final local chaining on the full set
        9. Prune low-score bridges, topological sort, score
        """
        # 1. Hybrid search
        results = await self.hybrid_search_functions(
            session=session,
            query=query,
            repository_id=repository_id,
            top_k=100,
            similarity_threshold=0.3,
            semantic_weight=3.0,
            bm25_weight=1.0,
            exclude_tests=exclude_tests,
        )

        logger.info(
            "\n======== GRAPHRAG-V2 SEARCH ========\n"
            "QUERY:          %s\n"
            "HYBRID RESULTS: %d\n"
            "====================================",
            query, len(results),
        )

        if not results:
            return [], []

        if len(results) <= 1:
            return results, [results]

        # Build call graph (needed for expansion + bridging)
        G, func_map, repo_name = await build_weighted_call_graph(
            session, repository_id,
        )

        # Build community summary (no LLM — derived from graph structure)
        community_data = detect_communities(G, func_map)
        community_summary = _summarize_communities(community_data)

        rerank_instruction = embedding_service.RERANK_INSTRUCTION
        if community_summary:
            rerank_instruction += f"\nCodebase components:\n{community_summary}\n"

        # 2. Rerank all candidates in one batch (with community context)
        documents = [_rerank_document(f) for f in results]

        logger.info(
            "\n======== RERANKER INSTRUCTION ========\n%s"
            "Query: %s\n"
            "======================================",
            rerank_instruction, query,
        )
        if documents:
            logger.info(
                "[graphrag-v2] sample reranker document [0]:\n%s",
                documents[0],
            )

        rerank_results = embedding_service.rerank(
            query=query, documents=documents, instruction=rerank_instruction,
        )

        score_map = {r["index"]: r["relevance_score"] for r in rerank_results}
        for i, func in enumerate(results):
            if i in score_map:
                func.similarity = round(score_map[i], 4)

        # 3. Select top-N seed by reranker score
        results.sort(key=lambda f: -f.similarity)
        seed = results[:30]

        logger.debug(
            "[graphrag-v2] reranked %d -> top %d seed (scores %.3f..%.3f)",
            len(results), len(seed),
            seed[0].similarity if seed else 0,
            seed[-1].similarity if seed else 0,
        )

        # Build scored lookup for reusing scores on neighbors
        scored_map = {f.function.function_id: f for f in results}

        # 4. Adaptive expansion (pass pre-built graph to avoid rebuilding)
        expanded, result_map, G, func_map, repo_name = (
            await expand_adaptive_neighbors(
                seed, session, repository_id, scored_results=results,
                prebuilt_graph=(G, func_map, repo_name),
            )
        )

        # 5. Local chaining to identify disconnected clusters
        pre_chains = order_functions_local(expanded)

        # 6. Steiner bridging to connect disconnected clusters
        if len(pre_chains) > 1:
            terminal_ids = {f.function.function_id for f in expanded}
            steiner_bridges = find_steiner_bridges(
                G, terminal_ids, pre_chains, cutoff=5.0,
            )

            steiner_added = []
            for bridge_id in steiner_bridges:
                if bridge_id in result_map:
                    continue
                if bridge_id not in func_map:
                    continue

                if bridge_id in scored_map:
                    result_map[bridge_id] = scored_map[bridge_id]
                else:
                    result_map[bridge_id] = _create_bridge_result(
                        func_map[bridge_id], repo_name,
                    )
                steiner_added.append(bridge_id)

            if steiner_added:
                expanded = list(result_map.values())
                logger.debug(
                    "[graphrag-v2] Steiner added %d bridge nodes "
                    "(from %d candidates, %d pre-chains)",
                    len(steiner_added), len(steiner_bridges), len(pre_chains),
                )

        # 7. Rerank ALL bridge nodes in one batch (with community context)
        bridges = [f for f in expanded if f.is_bridge]
        if bridges:
            bridge_docs = [_rerank_document(f) for f in bridges]

            RERANK_BATCH_LIMIT = 1000
            bridge_score_map = {}
            for batch_start in range(0, len(bridge_docs), RERANK_BATCH_LIMIT):
                batch = bridge_docs[batch_start:batch_start + RERANK_BATCH_LIMIT]
                batch_rerank = embedding_service.rerank(
                    query=query, documents=batch,
                    instruction=rerank_instruction,
                )
                for r in batch_rerank:
                    bridge_score_map[batch_start + r["index"]] = r["relevance_score"]

            for i, func in enumerate(bridges):
                if i in bridge_score_map:
                    func.similarity = round(bridge_score_map[i], 4)

            logger.debug(
                "[graphrag-v2] reranked %d bridge nodes", len(bridges),
            )

        # 8. Final local chaining on the full set
        chains = order_functions_local(expanded)

        # 9. Prune low-score nodes (may split chains into smaller pieces)
        chains = prune_low_score_nodes(chains, threshold=0.4)

        # 9b. Merge overlapping chains
        if len(chains) > 1:
            pre_merge_count = len(chains)
            pruned_funcs = [f for chain in chains for f in chain]
            pruned_map = {f.function.function_id: f for f in pruned_funcs}
            pruned_graph: dict[str, list[str]] = defaultdict(list)
            pruned_ids = set(pruned_map.keys())
            for f in pruned_funcs:
                fid = f.function.function_id
                for callee in (f.function.calls or []):
                    if callee in pruned_ids:
                        pruned_graph[fid].append(callee)

            chains = _merge_overlapping_chains(
                chains, pruned_graph, pruned_map, overlap_threshold=0.6,
            )
            logger.info(
                "[graphrag-v2] merge step: %d chains -> %d chains",
                pre_merge_count, len(chains),
            )

        chains.sort(key=lambda c: -_chain_score(c))

        # Debug: log first 5 chains
        for ci, chain in enumerate(chains[:5]):
            funcs_info = [
                f"{'[B]' if f.is_bridge else ''}{f.function.function_id}"
                for f in chain
            ]
            logger.info("[graphrag-v2] chain[%d]: %s", ci, funcs_info)

        flattened = [func for chain in chains for func in chain]
        return flattened, chains


# Singleton instance
search_service = SearchService()
