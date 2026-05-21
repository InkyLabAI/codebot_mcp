"""codebot init: scaffold project files for coding agent integration."""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Agent → rules file mapping
# ---------------------------------------------------------------------------

AGENTS = ["claude", "cursor", "windsurf", "cline", "copilot", "vibe"]

_RULES_FILES = {
    "claude":   "CLAUDE.md",
    "cursor":   ".cursor/rules",
    "windsurf": ".windsurfrules",
    "cline":    ".clinerules",
    "copilot":  ".github/copilot-instructions.md",
    "vibe":     ".vibe/AGENTS.md",
}

# Default MCP transport per agent.
# None means the agent does not support MCP (REST only — no .mcp.json generated).
_DEFAULT_TRANSPORT = {
    "claude":   "stdio",  # subprocess launched by the agent
    "cursor":   "stdio",  # subprocess launched by the agent
    "windsurf": "stdio",  # subprocess launched by the agent
    "cline":    "stdio",  # subprocess launched by the agent
    "copilot":  None,     # no MCP support — REST only
    "vibe":     "stdio",  # subprocess launched by the agent (config.toml format)
    "all":      "stdio",  # applied to all MCP-capable agents
}

# ---------------------------------------------------------------------------
# File content templates
# ---------------------------------------------------------------------------

_ENV_TEMPLATE = """\
VOYAGE_API_KEY=your_key_here
"""

_CODEBOT_RULES = """\
## codebot — Semantic Code Search

codebot indexes this codebase and exposes semantic search over its functions
and call graph. Use codebot tools for code navigation and understanding.

### Tool selection

| Task | Tool |
|---|---|
| Get a map of the codebase before searching | `get_codebase_overview` |
| Understand what a piece of code does | `search_code` |
| Find where a feature or behaviour is implemented | `search_code` |
| Read the full source of a known function | `lookup_function` |
| Explore what a function calls | `expand_function` |
| Find a file by path pattern | Glob (built-in) |
| Find an exact string, import, or symbol reference | Grep (built-in) |
| Read a known file path | Read (built-in) |

### Start of session

Call `get_codebase_overview` at the beginning of a session to learn the
codebase structure: which modules exist, what each one does, and the key
entry-point functions. Use the module names and terminology from the overview
when writing `search_code` queries — this produces significantly better results
than generic descriptions.

### Writing good `search_code` queries

Describe what the code **does**, using module and class names from the overview:

- GOOD: `"parse Python source and extract function definitions"`
- GOOD: `"hybrid search combining semantic vectors and keyword ranking"`
- GOOD: `"compute SHA-256 hash of a file for change detection"`
- BAD:  `"parse_functions"` — that's a name, not a description
- BAD:  `"the function that calls cosine_distance"` — describes structure, not behaviour

### Typical workflow

1. `get_codebase_overview` — understand the codebase map (once per session).
2. `search_code` — find candidate functions by natural language description.
   Results are returned as a shallow call tree; nodes with `(+N more)` have
   deeper subtrees.
3. `expand_function` — drill one level deeper into a node from the last search.
4. `lookup_function` — read full source once you know the exact function name.

### After editing code

The index updates automatically in the background (2-3 second debounce after
the last file change). If you search immediately after editing, the search
waits for the re-index to finish before returning results, so results are
always up to date.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_json_stdio(repo_path: Path) -> dict:
    return {
        "mcpServers": {
            "codebot": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "codebot_mcp", "serve", str(repo_path)],
            }
        }
    }


def _mcp_json_sse(sse_port: int) -> dict:
    return {
        "mcpServers": {
            "codebot": {
                "type": "sse",
                "url": f"http://localhost:{sse_port}/sse",
            }
        }
    }


def _write_mcp_json(
    path: Path,
    repo_path: Path,
    transport: str,
    sse_port: int,
    created: list,
    updated: list,
) -> None:
    new_server = (
        _mcp_json_stdio(repo_path) if transport == "stdio" else _mcp_json_sse(sse_port)
    )["mcpServers"]["codebot"]

    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

        servers = existing.setdefault("mcpServers", {})
        servers["codebot"] = new_server
        path.write_text(json.dumps(existing, indent=2) + "\n")
        updated.append(path.name)
    else:
        data = {"mcpServers": {"codebot": new_server}}
        path.write_text(json.dumps(data, indent=2) + "\n")
        created.append(path.name)


def _write_vibe_config(path: Path, repo_path: Path, created: list, updated: list, skipped: list) -> None:
    """Write or update .vibe/config.toml with the codebot MCP server entry."""
    block = (
        "\n[[mcp_servers]]\n"
        'name = "codebot"\n'
        'transport = "stdio"\n'
        f'command = "{sys.executable}"\n'
        f'args = ["-m", "codebot_mcp", "serve", "{repo_path}"]\n'
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        content = path.read_text()
        if 'name = "codebot"' in content:
            skipped.append(f"{path} (codebot entry already present)")
            return
        path.write_text(content.rstrip("\n") + block)
        updated.append(str(path.relative_to(path.parents[len(path.parents) - 2])))
    else:
        path.write_text(block.lstrip("\n"))
        created.append(str(path))


def _write_env(path: Path, voyage_api_key: str | None, created: list, updated: list, skipped: list) -> None:
    import re

    if path.exists():
        if voyage_api_key:
            content = path.read_text()
            if re.search(r"^VOYAGE_API_KEY=", content, re.MULTILINE):
                content = re.sub(
                    r"^VOYAGE_API_KEY=.*$",
                    f"VOYAGE_API_KEY={voyage_api_key}",
                    content,
                    flags=re.MULTILINE,
                )
            else:
                content = content.rstrip("\n") + f"\nVOYAGE_API_KEY={voyage_api_key}\n"
            path.write_text(content)
            updated.append(path.name)
        else:
            skipped.append(f"{path.name} (already exists — not overwritten)")
    else:
        key_value = voyage_api_key or "your_key_here"
        path.write_text(f"VOYAGE_API_KEY={key_value}\n")
        created.append(path.name)


def _update_gitignore(repo_path: Path, updated: list, skipped: list) -> None:
    gitignore = repo_path / ".gitignore"
    entry = ".env"

    if gitignore.exists():
        content = gitignore.read_text()
        lines = content.splitlines()
        if any(line.strip() == entry for line in lines):
            skipped.append(".gitignore (already contains .env)")
            return
        # Append on a new line
        sep = "\n" if content and not content.endswith("\n") else ""
        gitignore.write_text(content + sep + entry + "\n")
        updated.append(".gitignore")
    else:
        gitignore.write_text(entry + "\n")
        updated.append(".gitignore")


def _write_rules_file(path: Path, created: list, updated: list, skipped: list) -> None:
    marker = "## codebot"

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        content = path.read_text()
        if marker in content:
            skipped.append(f"{path} (codebot section already present)")
            return
        # Append the codebot section
        sep = "\n\n" if content and not content.endswith("\n\n") else "\n"
        path.write_text(content + sep + _CODEBOT_RULES)
        updated.append(str(path.relative_to(path.parents[len(path.parents) - 2])))
    else:
        path.write_text(_CODEBOT_RULES)
        created.append(str(path))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def init_project(
    repo_path: str,
    agent: str,
    transport: str | None,
    sse_port: int,
    voyage_api_key: str | None = None,
    run_index: bool = True,
) -> None:
    root = Path(os.path.abspath(repo_path))

    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    # Resolve transport: use explicit value if given, otherwise agent's default
    resolved_transport = transport if transport is not None else _DEFAULT_TRANSPORT[agent]

    agents = AGENTS if agent == "all" else [agent]

    created: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []

    # .vibe/config.toml — Mistral Vibe uses TOML format instead of .mcp.json
    if "vibe" in agents:
        _write_vibe_config(root / ".vibe" / "config.toml", root, created, updated, skipped)

    # .mcp.json — for all other MCP-capable agents (copilot and vibe excluded)
    non_vibe_mcp_agents = [
        a for a in agents if _DEFAULT_TRANSPORT.get(a) is not None and a != "vibe"
    ]
    if non_vibe_mcp_agents and resolved_transport is not None:
        _write_mcp_json(
            root / ".mcp.json", root, resolved_transport, sse_port, created, updated
        )

    # .env — write template or update with provided key
    _write_env(root / ".env", voyage_api_key, created, updated, skipped)

    # Make the key available to _maybe_run_setup even though dotenv was loaded earlier
    if voyage_api_key:
        os.environ["VOYAGE_API_KEY"] = voyage_api_key

    # .gitignore — ensure .env is ignored
    _update_gitignore(root, updated, skipped)

    # Agent rules files
    for ag in agents:
        rules_path = root / _RULES_FILES[ag]
        _write_rules_file(rules_path, created, updated, skipped)

    # Summary
    _print_summary(root, created, updated, skipped, resolved_transport, sse_port)

    # Build the index if a valid API key is available
    _maybe_run_setup(root, run_index)


def _maybe_run_setup(root: Path, run_index: bool) -> None:
    """Run setup if a valid API key is available and the index doesn't exist yet."""
    from codebot_mcp.db import get_db_path

    db_path = get_db_path(str(root))
    if os.path.exists(db_path):
        print("  Index already exists — skipping setup.")
        print()
        return

    if not run_index:
        print("  Skipping index build (--no-index). The server will build it on first launch.")
        print()
        return

    api_key = os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        print("  No API key found — skipping index build.")
        print("  Fill in VOYAGE_API_KEY in .env; the server will index on first launch.")
        print()
        return

    print("  Building index (this may take a few minutes)...")
    print()
    from codebot_mcp.setup import setup_repository
    setup_repository(str(root))
    print()
    print("  Index ready.")
    print()


def _print_summary(
    root: Path,
    created: list,
    updated: list,
    skipped: list,
    transport: str,
    sse_port: int,
) -> None:
    print(f"\ncodebot init — {root}\n")

    if created:
        print("  Created:")
        for f in created:
            print(f"    + {f}")

    if updated:
        print("  Updated:")
        for f in updated:
            print(f"    ~ {f}")

    if skipped:
        print("  Skipped:")
        for f in skipped:
            print(f"    - {f}")

    print()

    env_file = root / ".env"
    if env_file.exists() and "VOYAGE_API_KEY=your_key_here" in env_file.read_text():
        print("  Next step: add your Voyage AI key to .env")
        print("    VOYAGE_API_KEY=your_key_here  →  VOYAGE_API_KEY=pa-...")
        print()

    if transport == "stdio":
        print("  Start the server by opening your agent in this directory.")
        print("  The agent will launch codebot automatically via .mcp.json.")
    else:
        print(f"  Start the server:")
        print(f"    codebot serve-http {root} --sse-port {sse_port}")

    print()
