"""Main entry point for PostgreSQL MCP Server.

This module provides the CLI entry point for running the MCP server
using FastMCP with stdio transport.
"""

from pathlib import Path

import anyio
from dotenv import load_dotenv

from pg_mcp.server import mcp

# .env lives at the project root (two levels above src/pg_mcp/).
# Anchored to this file so the server works regardless of the working
# directory MCP clients launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Main entry point for the PostgreSQL MCP Server.

    This function starts the FastMCP server using stdio transport,
    enabling communication with MCP clients.

    The server lifecycle is managed through the lifespan context manager,
    which handles:
    - Configuration loading
    - Database connection pool creation
    - Schema cache initialization
    - Service component setup
    - Graceful shutdown

    Example:
        Run the server:
        >>> python -m pg_mcp

        Run with environment variables:
        >>> DATABASE_HOST=localhost DATABASE_NAME=mydb python -m pg_mcp
    """
    # Load .env into os.environ before Settings is constructed.
    # Nested BaseSettings sub-models (OpenAIConfig etc.) are instantiated via
    # default_factory and do not inherit the parent's env_file setting, so
    # .env values must be present in the process environment.
    load_dotenv(PROJECT_ROOT / ".env")
    anyio.run(mcp.run_stdio_async)


if __name__ == "__main__":
    main()
