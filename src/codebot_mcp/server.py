"""MCP server exposing CodeBot semantic search to Claude Code (stdio transport)."""

import asyncio
import json
import os
import sys
import time
import uuid
import logging
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from codebot_mcp.db import init_db, get_session_maker, get_db_path, Repository, Function, Class
from codebot_mcp.services.search_service import search_service
from sqlalchemy import select, and_, or_

logger = logging.getLogger(__name__)

LOG_DIR = os.environ.get("MCP_LOG_DIR", os.path.expanduser("~/.codebot/logs"))
os.makedirs(LOG_DIR, exist_ok=True)
QUERY_LOG = os.path.join(LOG_DIR, "queries.jsonl")


def _log_query(tool: str, params: dict, results: list[dict], duration_s: float):
    """Append a structured log entry for every tool call."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "params": params,
        "duration_s": round(duration_s, 3),
        "num_results": len(results),
        "results": results,
    }
    with open(QUERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        "[%s] query=%r  results=%d  duration=%.2fs",
        tool, params.get("query") or params.get("name"),
        len(results), duration_s,
    )


# ---------------------------------------------------------------------------
# Globals resolved at startup
# ---------------------------------------------------------------------------
repo_uuid: str | None = None
repo_path: str | None = None

mcp = FastMCP("codebot")

# Lock held by the incremental indexer while updating the DB.
# Search tools acquire-then-release it as a barrier so they always
# read a consistent, up-to-date index.
_indexing_lock: asyncio.Lock = asyncio.Lock()


async def _wait_for_index() -> None:
    """Block until any active indexing job has finished."""
    async with _indexing_lock:
        pass

# ---------------------------------------------------------------------------
# Cached chain from the last search (for expand_function).
# Persisted to disk so the cache survives server restarts (e.g. Vibe stdio).
# ---------------------------------------------------------------------------
_cached_chain: dict = {}  # function_id -> FunctionSearchResult
_CACHE_FILE: str | None = None  # set in run_server()


def _get_cache_path() -> str | None:
    """Return the path to the chain cache file."""
    return _CACHE_FILE


def _save_chain_cache():
    """Persist _cached_chain to disk as JSON."""
    path = _get_cache_path()
    if not path or not _cached_chain:
        return
    try:
        from codebot_mcp.schemas.function import FunctionSearchResult
        data = {fid: r.model_dump(mode="json") for fid, r in _cached_chain.items()}
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Failed to save chain cache: %s", e)


def _load_chain_cache():
    """Restore _cached_chain from disk if in-memory cache is empty."""
    global _cached_chain
    path = _get_cache_path()
    if not path or _cached_chain:
        return
    try:
        with open(path, "r") as f:
            data = json.load(f)
        from codebot_mcp.schemas.function import FunctionSearchResult
        _cached_chain = {fid: FunctionSearchResult.model_validate(obj) for fid, obj in data.items()}
        logger.info("Restored chain cache: %d entries", len(_cached_chain))
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Failed to load chain cache: %s", e)


def _count_descendants(func_id, chain_index, visited=None):
    """Count all unique descendants of a function within the cached chain."""
    if visited is None:
        visited = set()
    r = chain_index.get(func_id)
    if not r:
        return 0
    count = 0
    for call_id in (r.function.calls or []):
        if call_id in chain_index and call_id not in visited:
            visited.add(call_id)
            count += 1 + _count_descendants(call_id, chain_index, visited)
    return count


def _find_roots(chain_index):
    """Find entry points (in-degree 0) in the cached chain."""
    chain_ids = set(chain_index.keys())
    in_degree = {fid: 0 for fid in chain_ids}
    for r in chain_index.values():
        for call_id in (r.function.calls or []):
            if call_id in chain_ids:
                in_degree[call_id] = in_degree.get(call_id, 0) + 1

    roots = [fid for fid in chain_ids if in_degree[fid] == 0]
    if not roots:
        best = max(chain_index.values(), key=lambda r: r.similarity)
        roots = [best.function.function_id]
    return roots


def _render_node(func_id, chain_index):
    """Render a single node line: `function_id  [filename:line]  (+N more)`."""
    r = chain_index[func_id]
    f = r.function
    filename = os.path.basename(f.file_path)
    desc_count = _count_descendants(func_id, chain_index)
    suffix = f"  (+{desc_count} more)" if desc_count > 0 else ""
    return f"{f.function_id}  [{filename}:{f.start_line}]{suffix}"


def _render_shallow_tree(chain_index):
    """Render roots and their direct children (depth 1) as a nested markdown list."""
    roots = _find_roots(chain_index)
    chain_ids = set(chain_index.keys())
    placed = set()
    lines = []

    for root_id in roots:
        if root_id in placed:
            continue
        placed.add(root_id)
        lines.append(f"- {_render_node(root_id, chain_index)}")

        r = chain_index[root_id]
        children = [
            cid for cid in (r.function.calls or [])
            if cid in chain_ids and cid not in placed
        ]
        for child_id in children:
            placed.add(child_id)
            lines.append(f"  - {_render_node(child_id, chain_index)}")

    return "\n".join(lines)


def _render_full_tree(chain_index):
    """Render the complete call tree using box-drawing characters."""
    roots = _find_roots(chain_index)
    chain_ids = set(chain_index.keys())
    placed = set()
    lines = []

    def _render_node_tree(func_id, prefix, child_prefix):
        if func_id in placed:
            return
        placed.add(func_id)
        f = chain_index[func_id].function
        filename = os.path.basename(f.file_path)
        lines.append(f"{prefix}{f.function_id}  [{filename}:{f.start_line}]")

        children = [cid for cid in (f.calls or []) if cid in chain_ids and cid not in placed]
        for i, child_id in enumerate(children):
            last = i == len(children) - 1
            _render_node_tree(
                child_id,
                prefix=child_prefix + ("└── " if last else "├── "),
                child_prefix=child_prefix + ("    " if last else "│   "),
            )

    for root_id in roots:
        _render_node_tree(root_id, prefix="", child_prefix="")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_codebase_overview(refresh: bool = False) -> str:
    """Get a high-level map of the codebase organised by component.

    Returns the major modules, what each one does, and its key entry-point
    functions. Call this BEFORE writing search queries so you can use the
    correct module names, class names, and terminology that actually exist
    in the codebase.

    The overview is derived from the call graph (no LLM call) and cached
    after the first invocation.

    Args:
        refresh: Recompute even if a cached overview exists.
    """
    await _wait_for_index()

    session_maker = get_session_maker()
    async with session_maker() as session:
        # Return cached summary if available and not forcing refresh
        repo_stmt = select(Repository).where(Repository.id == repo_uuid)
        repo = (await session.execute(repo_stmt)).scalar_one_or_none()
        if not refresh and repo and repo.community_summary:
            return repo.community_summary

        from codebot_mcp.utils.call_graph import detect_communities, build_weighted_call_graph
        G, func_map, _ = await build_weighted_call_graph(session, uuid.UUID(repo_uuid))
        community_data = detect_communities(G, func_map)

    if not community_data:
        return "No community structure detected."

    lines = ["# Codebase Overview\n"]
    for c in community_data:
        fn_names = ", ".join(fn["name"] for fn in c["functions"])
        header = f"## {c['module']}"
        if c["module_docstring"]:
            header += f" — {c['module_docstring']}"
        lines.append(header)
        if fn_names:
            lines.append(fn_names)
        lines.append("")

    summary = "\n".join(lines)

    # Cache in DB so subsequent calls are instant
    async with session_maker() as session:
        await session.execute(
            Repository.__table__.update()
            .where(Repository.id == repo_uuid)
            .values(community_summary=summary)
        )
        await session.commit()

    return summary


@mcp.tool()
async def search_code(query: str, exclude_tests: bool = True) -> str:
    """Search the codebase by describing what the code does.

    Returns a shallow call tree: entry points and their direct callees.
    Nodes with deeper callees show (+N more) -- use expand_function to
    drill into them. Each node shows a function_id in dot-separated format
    (module.path.ClassName.function_name) and its file location.

    Use expand_function to explore deeper into a subtree.
    Use lookup_function with "ClassName.method" to read a function's code.

    Args:
        query: Natural language description of the code you're looking for.
        exclude_tests: Whether to filter out test functions (default True).
    """
    global _cached_chain

    await _wait_for_index()

    t0 = time.monotonic()

    session_maker = get_session_maker()
    async with session_maker() as session:
        results, raw_chains = await search_service.graphrag_search(
            session=session,
            query=query,
            repository_id=uuid.UUID(repo_uuid),
            exclude_tests=exclude_tests,
        )

    if not results:
        _cached_chain = {}
        return "No results found."

    best_chain = raw_chains[0] if raw_chains else results

    # MCP agents get the focused best-chain view; cache all results so that
    # expand_function can reach any node the agent discovers.
    _cached_chain = {r.function.function_id: r for r in results}
    _save_chain_cache()

    _log_query("search_code", {"query": query, "exclude_tests": exclude_tests}, [
        {
            "function_id": r.function.function_id,
            "file_path": r.function.file_path,
            "score": r.similarity,
            "is_bridge": r.is_bridge,
        }
        for r in best_chain
    ], time.monotonic() - t0)

    if len(best_chain) <= 1:
        r = best_chain[0]
        f = r.function
        filename = os.path.basename(f.file_path)
        return f"- {f.function_id}  [{filename}:{f.start_line}]"

    result = _render_shallow_tree(_cached_chain)
    # result += "\n\nUse expand_function to drill into nodes with (+N more)."
    # result += "\nUse lookup_function with \"ClassName.method\" to read a function's code."
    return result


@mcp.tool()
async def expand_function(function_id: str) -> str:
    """Expand a node from the last search_code result to see its callees.

    Shows the function and its direct children (one level deeper).
    Children with further descendants show (+N more) so you can
    keep expanding deeper as needed.

    Args:
        function_id: The function_id from a search_code or previous expand_function result.
    """
    await _wait_for_index()
    _load_chain_cache()

    if not _cached_chain:
        return "No cached search results. Run search_code first."

    r = _cached_chain.get(function_id)
    if not r:
        return f"Function '{function_id}' not found in the last search result."

    chain_ids = set(_cached_chain.keys())
    f = r.function
    filename = os.path.basename(f.file_path)

    lines = [f"- {f.function_id}  [{filename}:{f.start_line}]"]

    children = [
        cid for cid in (f.calls or [])
        if cid in chain_ids
    ]

    if not children:
        lines.append("  (no callees in this chain)")
    else:
        for child_id in children:
            lines.append(f"  - {_render_node(child_id, _cached_chain)}")

    return "\n".join(lines)


@mcp.tool()
async def lookup_function(
    names: str,
    file_path: str | None = None,
) -> str:
    """Look up one or more functions and return their full code.

    Function names use the format from search_code results
    (dot-separated: module.path.ClassName.function_name). Use
    "ClassName.method" for methods or just "function_name" for top-level.

    Args:
        names: Function names separated by commas. Use "ClassName.method" for methods.
               Examples: "chunk_by_title", "chunk_by_title, is_title", "ChunkingConfig.new"
        file_path: Optional file path substring to filter all lookups by.
    """
    await _wait_for_index()
    t0 = time.monotonic()

    # Split into function lookups (qualified: ClassName.method or full id)
    # and class lookups (bare name: ClassName)
    func_lookups = []  # list of (func_name, class_name)
    class_names = []   # bare names → query Class table

    for entry in names.split(","):
        entry = entry.strip()
        if not entry:
            continue
        dot_parts = entry.split(".")
        if len(dot_parts) >= 2:
            # Full function_id or ClassName.method — use last two parts
            func_lookups.append((dot_parts[-1], dot_parts[-2]))
        else:
            # Bare name — try as a function (no class qualifier) AND as a class
            func_lookups.append((entry, None))
            class_names.append(entry)

    if not func_lookups and not class_names:
        return "No function names provided."

    session_maker = get_session_maker()
    async with session_maker() as session:
        functions = []
        if func_lookups:
            per_func_conditions = []
            for func_name, class_name in func_lookups:
                conds = [Function.name == func_name]
                if class_name:
                    conds.append(Function.class_name == class_name)
                per_func_conditions.append(and_(*conds))

            base_conds = [Function.repository_id == repo_uuid]
            if file_path:
                base_conds.append(Function.file_path.contains(file_path))

            stmt = (
                select(Function)
                .where(and_(*base_conds, or_(*per_func_conditions)))
                .limit(50)
            )
            result = await session.execute(stmt)
            functions = result.scalars().all()

        classes = []
        if class_names:
            base_cls_conds = [Class.repository_id == repo_uuid, Class.name.in_(class_names)]
            if file_path:
                base_cls_conds.append(Class.file_path.contains(file_path))
            cls_stmt = select(Class).where(and_(*base_cls_conds)).limit(20)
            cls_result = await session.execute(cls_stmt)
            classes = cls_result.scalars().all()

    params = {"names": names, "file_path": file_path}

    if not functions and not classes:
        _log_query("lookup_function", params, [], time.monotonic() - t0)
        return f"No functions or classes found for: {names}"

    _log_query("lookup_function", params, [
        {"function_id": fn.function_id, "file_path": fn.file_path}
        for fn in functions
    ] + [
        {"class_id": cls.class_id, "file_path": cls.file_path}
        for cls in classes
    ], time.monotonic() - t0)

    parts = []
    for fn in functions:
        parts.append(f"# {fn.file_path}:{fn.start_line}")
        if fn.class_name:
            parts.append(f"# Class: {fn.class_name}")
        parts.append(fn.code)
        parts.append("")

    for cls in classes:
        parts.append(f"# {cls.file_path}:{cls.start_line}")
        parts.append(f"# Class: {cls.name}")
        parts.append(cls.code)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def _setup_globals(repository_path: str) -> str | None:
    """Populate module globals needed by tool functions. Returns an error string or None."""
    global repo_uuid, repo_path, _CACHE_FILE

    api_key = os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return (
            "Error: VOYAGE_API_KEY is not set.\n"
            "Add your Voyage AI key to .env in the project directory:\n"
            "  VOYAGE_API_KEY=pa-...\n"
            "Get a free key at https://www.voyageai.com"
        )

    repo_path = os.path.abspath(repository_path)
    db_path = get_db_path(repo_path)
    _CACHE_FILE = db_path.rsplit(".", 1)[0] + ".chain_cache.json"

    if not os.path.exists(db_path):
        return f"No index found at {db_path}. Run 'codebot setup {repo_path}' first."

    await init_db(db_path)

    session_maker = get_session_maker()
    async with session_maker() as session:
        result = await session.execute(select(Repository))
        repo = result.scalar_one_or_none()
        if not repo:
            return "No repository found in database."
        repo_uuid = repo.id

    return None


async def db_stats() -> str:
    """CLI-only: print index statistics for the current repository."""
    from sqlalchemy import func as sql_func
    session_maker = get_session_maker()
    async with session_maker() as session:
        total_functions = (await session.execute(
            select(sql_func.count()).select_from(Function).where(Function.repository_id == repo_uuid)
        )).scalar_one()
        embedded_functions = (await session.execute(
            select(sql_func.count()).select_from(FunctionEmbedding)
            .join(Function, Function.id == FunctionEmbedding.function_id)
            .where(Function.repository_id == repo_uuid)
        )).scalar_one()
        functions_with_calls = (await session.execute(
            select(sql_func.count()).select_from(Function)
            .where(Function.repository_id == repo_uuid)
            .where(Function.calls.isnot(None))
        )).scalar_one()

    lines = [
        f"Functions indexed:    {total_functions}",
        f"Functions embedded:   {embedded_functions}",
        f"Functions with calls: {functions_with_calls}",
        f"Coverage:             {embedded_functions/total_functions*100:.1f}%" if total_functions else "N/A",
    ]
    return "\n".join(lines)


async def search_tree(query: str, exclude_tests: bool = True) -> str:
    """CLI-only: run a search and return the full call tree at all depths."""
    global _cached_chain

    await _wait_for_index()

    t0 = time.monotonic()
    session_maker = get_session_maker()
    async with session_maker() as session:
        results, raw_chains = await search_service.graphrag_search(
            session=session,
            query=query,
            repository_id=uuid.UUID(repo_uuid),
            exclude_tests=exclude_tests,
        )

    if not results:
        _cached_chain = {}
        return "No results found."

    # Use all functions across all chains (not just the best chain) so the
    # full picture is visible. _cached_chain also covers expand after search-tree.
    _cached_chain = {r.function.function_id: r for r in results}
    _save_chain_cache()

    _log_query("search_tree", {"query": query, "exclude_tests": exclude_tests}, [
        {"function_id": r.function.function_id, "file_path": r.function.file_path, "score": r.similarity}
        for r in results
    ], time.monotonic() - t0)

    return _render_full_tree(_cached_chain)


async def run_cli_query(repository_path: str, tool: str, **kwargs) -> str:
    """Run a single tool call for CLI use (no MCP server, no watcher)."""
    err = await _setup_globals(repository_path)
    if err:
        return err

    if tool == "search":
        return await search_code(**kwargs)
    if tool == "search-tree":
        return await search_tree(**kwargs)
    if tool == "expand":
        return await expand_function(**kwargs)
    if tool == "lookup":
        return await lookup_function(**kwargs)
    if tool == "stats":
        return await db_stats()
    if tool == "overview":
        return await get_codebase_overview(**kwargs)
    return f"Unknown tool: {tool}"


async def run_server(repository_path: str):
    """Initialize DB and run MCP server with stdio transport."""
    global repo_uuid, repo_path, _CACHE_FILE

    abs_path = os.path.abspath(repository_path)
    db_path = get_db_path(abs_path)

    if not os.path.exists(db_path):
        logger.info("No index found — running setup for %s", abs_path)
        from codebot_mcp.setup import setup_repository
        await asyncio.to_thread(setup_repository, abs_path)

    err = await _setup_globals(repository_path)
    if err:
        sys.exit(err)

    # Warn if not yet embedded
    session_maker = get_session_maker()
    async with session_maker() as session:
        result = await session.execute(select(Repository))
        repo = result.scalar_one_or_none()
        if repo and not repo.is_embedded:
            logger.warning(
                "Repository %s is not yet embedded -- semantic search will be degraded.",
                repo.name,
            )
        logger.info("MCP server ready -- repo: %s (%s)", repo.name if repo else "?", repo_uuid)

    # Start startup catch-up and file watcher as background tasks.
    # Both run independently; search tools wait on _indexing_lock when needed.
    async def _run_catchup() -> None:
        from codebot_mcp.incremental import startup_catchup
        async with _indexing_lock:
            await startup_catchup(repo_uuid, repo_path)

    async def _run_watcher() -> None:
        from codebot_mcp.watcher import watch_repo
        await watch_repo(repo_path, repo_uuid, _indexing_lock)

    asyncio.create_task(_run_catchup())
    asyncio.create_task(_run_watcher())

    await mcp.run_stdio_async()
