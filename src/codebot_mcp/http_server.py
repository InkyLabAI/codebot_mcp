"""HTTP server: REST API (port 8080) + MCP over SSE (port 8081).

Exposes the same search capabilities as the stdio MCP server but over HTTP,
making codebot compatible with any agent that supports HTTP tool calling
(OpenAI function calling, Mistral tools, Gemini tools, etc.) as well as
MCP-compatible agents that prefer SSE over stdio.

Usage:
    codebot serve-http /path/to/repo
    codebot serve-http /path/to/repo --rest-port 8080 --sse-port 8081
"""

import asyncio
import logging
import os
import sys
import uuid as _uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, and_, or_

import codebot_mcp.server as _mcp_server
from codebot_mcp.db import (
    init_db, get_db_path, get_session_maker,
    Repository, Function, Class,
)
from codebot_mcp.services.search_service import search_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models (used by FastAPI for OpenAPI schema generation)
# ---------------------------------------------------------------------------

class Callee(BaseModel):
    function_id: str
    file_path: str
    start_line: int
    has_deeper_callees: bool


class SearchResultItem(BaseModel):
    function_id: str
    file_path: str
    start_line: int
    end_line: int
    class_name: Optional[str]
    similarity: float
    is_bridge: bool
    callees: list[str]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class FunctionItem(BaseModel):
    function_id: str
    file_path: str
    start_line: int
    end_line: int
    class_name: Optional[str]
    code: str
    docstring: Optional[str]


class ClassItem(BaseModel):
    class_id: str
    file_path: str
    start_line: int
    end_line: int
    name: str
    code: str
    docstring: Optional[str]


class LookupResponse(BaseModel):
    functions: list[FunctionItem]
    classes: list[ClassItem]


class ExpandResponse(BaseModel):
    function_id: str
    file_path: str
    start_line: int
    callees: list[Callee]


class HealthResponse(BaseModel):
    status: str
    repo_path: Optional[str]
    repo_name: Optional[str]
    indexing: bool


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="codebot",
    description=(
        "Semantic code search for any Python codebase. "
        "Use /search to find functions by what they do, "
        "/lookup to read function source, "
        "/expand to explore the call graph."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Server liveness check. Returns whether an indexing job is currently running."""
    return HealthResponse(
        status="ok",
        repo_path=_mcp_server.repo_path,
        repo_name=os.path.basename(_mcp_server.repo_path) if _mcp_server.repo_path else None,
        indexing=_mcp_server._indexing_lock.locked(),
    )


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Natural language description of what the code does"),
    exclude_tests: bool = Query(True, description="Exclude test functions from results"),
):
    """Find functions by describing what they do.

    Returns ranked results with similarity scores. Each result includes the
    function's resolved callees so agents can navigate the call graph.
    """
    if not _mcp_server.repo_uuid:
        raise HTTPException(status_code=503, detail="Server not ready")

    await _mcp_server._wait_for_index()

    session_maker = get_session_maker()
    async with session_maker() as session:
        results, _ = await search_service.graphrag_search(
            session=session,
            query=q,
            repository_id=_uuid.UUID(_mcp_server.repo_uuid),
            exclude_tests=exclude_tests,
        )

    if not results:
        return SearchResponse(query=q, results=[])

    items = [
        SearchResultItem(
            function_id=r.function.function_id,
            file_path=r.function.file_path,
            start_line=r.function.start_line,
            end_line=r.function.end_line,
            class_name=r.function.class_name,
            similarity=round(r.similarity, 4),
            is_bridge=r.is_bridge,
            callees=r.function.calls or [],
        )
        for r in results
    ]
    return SearchResponse(query=q, results=items)


@app.get("/lookup", response_model=LookupResponse)
async def lookup(
    names: str = Query(..., description="Comma-separated function names. Use 'ClassName.method' or 'function_name'"),
    file_path: Optional[str] = Query(None, description="Optional file path substring to filter results"),
):
    """Retrieve full source code for one or more functions or classes.

    Accepts the dot-separated function_id format returned by /search,
    or shorter 'ClassName.method' / 'function_name' forms.
    """
    if not _mcp_server.repo_uuid:
        raise HTTPException(status_code=503, detail="Server not ready")

    await _mcp_server._wait_for_index()

    func_lookups = []   # (func_name, class_name)
    class_names = []

    for entry in names.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(".")
        if len(parts) >= 2:
            func_lookups.append((parts[-1], parts[-2]))
        else:
            class_names.append(entry)

    if not func_lookups and not class_names:
        raise HTTPException(status_code=400, detail="No valid names provided")

    session_maker = get_session_maker()
    async with session_maker() as session:
        functions = []
        if func_lookups:
            per_func = []
            for func_name, class_name in func_lookups:
                conds = [Function.name == func_name]
                if class_name:
                    conds.append(Function.class_name == class_name)
                per_func.append(and_(*conds))

            base = [Function.repository_id == _mcp_server.repo_uuid]
            if file_path:
                base.append(Function.file_path.contains(file_path))

            stmt = select(Function).where(and_(*base, or_(*per_func))).limit(50)
            result = await session.execute(stmt)
            functions = result.scalars().all()

        classes = []
        if class_names:
            base_cls = [Class.repository_id == _mcp_server.repo_uuid, Class.name.in_(class_names)]
            if file_path:
                base_cls.append(Class.file_path.contains(file_path))
            cls_stmt = select(Class).where(and_(*base_cls)).limit(20)
            cls_result = await session.execute(cls_stmt)
            classes = cls_result.scalars().all()

    if not functions and not classes:
        raise HTTPException(status_code=404, detail=f"No functions or classes found for: {names}")

    return LookupResponse(
        functions=[
            FunctionItem(
                function_id=fn.function_id,
                file_path=fn.file_path,
                start_line=fn.start_line,
                end_line=fn.end_line,
                class_name=fn.class_name,
                code=fn.code,
                docstring=fn.docstring,
            )
            for fn in functions
        ],
        classes=[
            ClassItem(
                class_id=cls.class_id,
                file_path=cls.file_path,
                start_line=cls.start_line,
                end_line=cls.end_line,
                name=cls.name,
                code=cls.code,
                docstring=cls.docstring,
            )
            for cls in classes
        ],
    )


@app.get("/expand", response_model=ExpandResponse)
async def expand(
    function_id: str = Query(..., description="The function_id from a /search result"),
):
    """Explore direct callees of a function.

    Looks up the function by its dot-separated function_id and returns its
    resolved callees with their locations. Unlike the MCP expand_function tool
    this does not require a prior /search call.
    """
    if not _mcp_server.repo_uuid:
        raise HTTPException(status_code=503, detail="Server not ready")

    await _mcp_server._wait_for_index()

    session_maker = get_session_maker()
    async with session_maker() as session:
        stmt = select(Function).where(
            Function.repository_id == _mcp_server.repo_uuid,
            Function.function_id == function_id,
        ).limit(1)
        result = await session.execute(stmt)
        func = result.scalar_one_or_none()

        if not func:
            raise HTTPException(status_code=404, detail=f"Function not found: {function_id}")

        callees = []
        if func.calls:
            callee_stmt = select(Function).where(
                Function.repository_id == _mcp_server.repo_uuid,
                Function.function_id.in_(func.calls),
            )
            callee_result = await session.execute(callee_stmt)
            callee_funcs = {f.function_id: f for f in callee_result.scalars().all()}

            for callee_id in func.calls:
                if callee_id in callee_funcs:
                    cf = callee_funcs[callee_id]
                    callees.append(Callee(
                        function_id=cf.function_id,
                        file_path=cf.file_path,
                        start_line=cf.start_line,
                        has_deeper_callees=bool(cf.calls),
                    ))

    return ExpandResponse(
        function_id=func.function_id,
        file_path=func.file_path,
        start_line=func.start_line,
        callees=callees,
    )


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

async def run_http_server(
    repository_path: str,
    rest_port: int = 8080,
    sse_port: int = 8081,
):
    """Initialize the index and run both the REST server and the MCP SSE server."""
    repo_path = os.path.abspath(repository_path)
    db_path = get_db_path(repo_path)

    if not os.path.exists(db_path):
        logger.info("No index found — running setup for %s", repo_path)
        from codebot_mcp.setup import setup_repository
        await asyncio.to_thread(setup_repository, repo_path)

    await init_db(db_path)

    session_maker = get_session_maker()
    async with session_maker() as session:
        stmt = select(Repository)
        result = await session.execute(stmt)
        repo = result.scalar_one_or_none()
        if not repo:
            sys.exit("No repository found in database.")

    # Populate the shared globals that MCP tools (SSE) and REST handlers rely on
    _mcp_server.repo_uuid = repo.id
    _mcp_server.repo_path = repo_path
    _mcp_server._CACHE_FILE = db_path.rsplit(".", 1)[0] + ".chain_cache.json"

    logger.info("HTTP server ready — repo: %s (%s)", repo.name, repo.id)

    # Startup catchup and file watcher (same as stdio server)
    async def _run_catchup():
        from codebot_mcp.incremental import startup_catchup
        async with _mcp_server._indexing_lock:
            await startup_catchup(_mcp_server.repo_uuid, repo_path)

    async def _run_watcher():
        from codebot_mcp.watcher import watch_repo
        await watch_repo(repo_path, _mcp_server.repo_uuid, _mcp_server._indexing_lock)

    asyncio.create_task(_run_catchup())
    asyncio.create_task(_run_watcher())

    # REST server (FastAPI + uvicorn)
    rest_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=rest_port,
        log_level="warning",
    )
    rest_server = uvicorn.Server(rest_config)

    # MCP SSE server (reuses the same registered tools from server.py)
    async def _run_sse():
        try:
            await _mcp_server.mcp.run_sse_async(
                host="0.0.0.0",
                port=sse_port,
                log_level="warning",
            )
        except Exception as e:
            logger.error("SSE server error: %s", e, exc_info=True)

    logger.info("REST API listening on http://0.0.0.0:%d  (docs: /docs)", rest_port)
    logger.info("MCP SSE listening on http://0.0.0.0:%d/sse", sse_port)

    await asyncio.gather(
        rest_server.serve(),
        _run_sse(),
    )
