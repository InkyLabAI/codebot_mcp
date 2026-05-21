"""CLI entry point for codebot."""

import argparse
import asyncio
import logging
import os
import sys


def _load_env(repo_path: str) -> None:
    """Load .env from the repo root before any codebot modules are imported.

    pydantic-settings reads env vars at Settings() instantiation time (module
    import).  By calling load_dotenv here — before the lazy codebot imports
    below — the API keys in {repo_path}/.env are available when Settings() runs.
    """
    try:
        from dotenv import load_dotenv
        env_file = os.path.join(os.path.abspath(repo_path), ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
            logging.getLogger(__name__).debug("Loaded env from %s", env_file)
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment


def main():
    parser = argparse.ArgumentParser(
        prog="codebot",
        description="MCP server for semantic code search using GraphRAG",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup subcommand
    setup_parser = subparsers.add_parser(
        "setup",
        help="Parse a repository and generate embeddings",
    )
    setup_parser.add_argument(
        "repo_path",
        help="Path to the repository root directory",
    )

    # serve subcommand (stdio — for Claude Code and other MCP subprocess clients)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server (stdio transport); runs setup automatically if needed",
    )
    serve_parser.add_argument(
        "repo_path",
        help="Path to the repository root directory",
    )

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold .mcp.json, .env, and agent rules files for a project",
    )
    init_parser.add_argument(
        "repo_path",
        nargs="?",
        default=os.getcwd(),
        help="Path to the project root directory (default: current directory)",
    )
    init_parser.add_argument(
        "--agent",
        choices=["claude", "cursor", "windsurf", "cline", "copilot", "vibe", "all"],
        default="claude",
        help="Which agent to generate rules for (default: claude)",
    )
    init_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=None,
        help="MCP transport (default: stdio for most agents, none for copilot)",
    )
    init_parser.add_argument(
        "--sse-port", type=int, default=8081,
        help="SSE port to use when --transport=sse (default: 8081)",
    )
    init_parser.add_argument(
        "--voyage-api-key",
        default=None,
        help="Voyage AI API key — saved to .env and used to build the index immediately",
    )
    init_parser.add_argument(
        "--no-index", action="store_true",
        help="Skip building the index (useful if you want to set the API key first)",
    )

    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics for the current repository",
    )
    stats_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )

    # overview subcommand
    overview_parser = subparsers.add_parser(
        "overview",
        help="Print a high-level map of the codebase organised by component",
    )
    overview_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )
    overview_parser.add_argument(
        "--refresh", action="store_true",
        help="Recompute the overview even if a cached version exists",
    )

    # search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search the indexed codebase by describing what code does",
    )
    search_parser.add_argument("query", help="Natural language description of the code")
    search_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )
    search_parser.add_argument(
        "--include-tests", action="store_true",
        help="Include test functions in results (excluded by default)",
    )

    # search-tree subcommand
    search_tree_parser = subparsers.add_parser(
        "search-tree",
        help="Search the codebase and return the full call tree at all depths",
    )
    search_tree_parser.add_argument("query", help="Natural language description of the code")
    search_tree_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )
    search_tree_parser.add_argument(
        "--include-tests", action="store_true",
        help="Include test functions in results (excluded by default)",
    )

    # lookup subcommand
    lookup_parser = subparsers.add_parser(
        "lookup",
        help="Look up one or more functions or classes by name and print their code",
    )
    lookup_parser.add_argument(
        "names",
        help='Comma-separated names: "function_name", "ClassName.method", or "ClassName"',
    )
    lookup_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )
    lookup_parser.add_argument(
        "--file", default=None, metavar="SUBSTRING",
        help="Filter results to file paths containing this substring",
    )

    # expand subcommand
    expand_parser = subparsers.add_parser(
        "expand",
        help="Expand a node from the last search result to see its callees",
    )
    expand_parser.add_argument("function_id", help="Dot-separated function_id from a search result")
    expand_parser.add_argument(
        "--repo", default=os.getcwd(), metavar="PATH",
        help="Path to the repository root (default: current directory)",
    )

    # serve-http subcommand (REST + SSE — for any HTTP-capable agent)
    serve_http_parser = subparsers.add_parser(
        "serve-http",
        help="Start REST API + MCP SSE server; compatible with any HTTP tool-calling agent",
    )
    serve_http_parser.add_argument(
        "repo_path",
        help="Path to the repository root directory",
    )
    serve_http_parser.add_argument(
        "--rest-port", type=int, default=8080,
        help="Port for the REST API server (default: 8080)",
    )
    serve_http_parser.add_argument(
        "--sse-port", type=int, default=8081,
        help="Port for the MCP SSE server (default: 8081)",
    )

    args = parser.parse_args()

    _query_commands = {"search", "search-tree", "lookup", "expand"}
    if args.verbose:
        level = logging.DEBUG
    elif args.command in _query_commands:
        level = logging.WARNING  # keep CLI output clean
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Determine the repo path for .env loading — different subcommands store it
    # under different attribute names.
    _repo_for_env = getattr(args, "repo_path", None) or getattr(args, "repo", None) or os.getcwd()
    _load_env(_repo_for_env)

    if args.command == "init":
        from codebot_mcp.init_cmd import init_project

        init_project(
            args.repo_path,
            agent=args.agent,
            transport=args.transport,
            sse_port=args.sse_port,
            voyage_api_key=args.voyage_api_key,
            run_index=not args.no_index,
        )
        return

    if args.command == "setup":
        from codebot_mcp.setup import setup_repository

        setup_repository(args.repo_path)

    elif args.command == "serve":
        from codebot_mcp.server import run_server

        asyncio.run(run_server(args.repo_path))

    elif args.command == "serve-http":
        from codebot_mcp.http_server import run_http_server

        asyncio.run(run_http_server(args.repo_path, args.rest_port, args.sse_port))

    elif args.command == "stats":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(args.repo, "stats"))
        print(result)

    elif args.command == "overview":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(args.repo, "overview", refresh=args.refresh))
        print(result)

    elif args.command == "search":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(
            args.repo,
            "search",
            query=args.query,
            exclude_tests=not args.include_tests,
        ))
        print(result)

    elif args.command == "search-tree":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(
            args.repo,
            "search-tree",
            query=args.query,
            exclude_tests=not args.include_tests,
        ))
        print(result)

    elif args.command == "lookup":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(
            args.repo,
            "lookup",
            names=args.names,
            file_path=args.file,
        ))
        print(result)

    elif args.command == "expand":
        from codebot_mcp.server import run_cli_query

        result = asyncio.run(run_cli_query(
            args.repo,
            "expand",
            function_id=args.function_id,
        ))
        print(result)


if __name__ == "__main__":
    main()
