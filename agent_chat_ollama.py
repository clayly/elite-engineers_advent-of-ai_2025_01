from __future__ import annotations

import argparse
import asyncio
import os
import sys
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
# Optional import; we'll guide the user if missing at runtime
try:
    from langchain_community.chat_message_histories.sql import SQLChatMessageHistory  # type: ignore
except Exception:
    SQLChatMessageHistory = None  # type: ignore

from langgraph.constants import START
import json
from pathlib import Path

# MCP client for multiple servers
from langchain_mcp_adapters.client import MultiServerMCPClient

# LangGraph imports
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


def init_llm(model: str, temperature: float, max_tokens: Optional[int], base_url: Optional[str] = None) -> ChatOllama:
    """Initialize a local Ollama chat model via LangChain's ChatOllama.

    Reads base URL from --ollama-base-url or OLLAMA_BASE_URL (default http://localhost:11434).
    Maps max_tokens to Ollama's num_predict.
    """
    resolved_base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "base_url": resolved_base_url,
    }
    # Map max_tokens to num_predict for Ollama
    if max_tokens is not None:
        # ChatOllama forwards unknown kwargs via model_kwargs
        kwargs["num_predict"] = int(max_tokens)

    return ChatOllama(**kwargs)


def build_graph(llm: Any, tools: Optional[List[Any]] = None):
    """Build a LangGraph that supports tool execution if tools are provided.

    - Without tools: single model call, then END (original behavior).
    - With tools: route based on tools_condition to execute tools, then loop back to model.
    """
    graph = StateGraph(MessagesState)

    async def call_model(state: MessagesState):
        # LLM is a ChatModel; pass the conversation so far and get one response
        response = await llm.ainvoke(state["messages"])  # returns an AIMessage
        return {"messages": [response]}

    graph.add_node("model", call_model)

    if tools:
        # Add a ToolNode and conditional routing so tools can be executed
        graph.add_node("tools", ToolNode(tools))
        # Route to tools when the model requested a tool; otherwise finish
        graph.add_conditional_edges(
            "model",
            tools_condition,
            {"tools": "tools", END: END},
        )
        # After tools run, go back to the model to include tool output in context
        graph.add_edge("tools", "model")
        graph.add_edge(START, "model")
    else:
        # Original simple flow
        graph.add_edge(START, "model")
        graph.add_edge("model", END)

    return graph.compile()


# ============ MCP Integration (optional) ============

def _load_mcp_servers_config(config_path: Path) -> Dict[str, Any]:
    try:
        if not config_path.exists():
            return {}
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # ensure elements are dicts
            return data
    except Exception:
        pass
    return {}


def _init_mcp_tools() -> List[Any]:
    """Initialize MultiServerMCPClient from mcp_servers.json and return tools list.
    Also prints the list of all MCP servers and their tools at start.
    Returns empty list on any failure or if package isn't installed.
    """
    # Look for mcp_servers.json next to this script (project root)
    config_path = Path(__file__).resolve().parent / "mcp_servers.json"

    # Load servers from config and print them
    servers = _load_mcp_servers_config(config_path)
    if not servers:
        print("[MCP] No servers configured (mcp_servers.json not found or empty).")
        return []

    print("[MCP] Loaded servers (from mcp_servers.json):")
    for idx, (name, srv) in enumerate(servers.items()):
        command = srv.get("command") or "<command?>"
        transport = srv.get("transport") or "<transport?>"
        print(f"  {idx}. {name} | command: {command} | transport: {transport}")

    # Initialize client and collect tools (get_tools is async)
    try:
        client = MultiServerMCPClient(connections=servers)
    except Exception as e:
        print(f"[MCP] Failed to initialize MultiServerMCPClient: {e}")
        return []

    # Retrieve tools via async get_tools and print summary
    try:
        async def _fetch_tools():
            return await client.get_tools()

        try:
            tools = asyncio.run(_fetch_tools())
        except RuntimeError:
            # Fallback for environments with a running loop (e.g., notebooks)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    new_loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(new_loop)
                        tools = new_loop.run_until_complete(client.get_tools())
                    finally:
                        asyncio.set_event_loop(loop)
                        new_loop.close()
                else:
                    tools = loop.run_until_complete(client.get_tools())
            except Exception as inner_e:
                print(f"[MCP] Error while awaiting tools (event loop): {inner_e}")
                return []

        if not isinstance(tools, list):
            # The adapter promises a list[BaseTool]; guard just in case
            try:
                # Attempt to coerce common container types
                if isinstance(tools, dict):
                    flat: List[Any] = []
                    for v in tools.values():
                        if isinstance(v, list):
                            flat.extend(v)
                    tools = flat
                else:
                    tools = list(tools)  # type: ignore[assignment]
            except Exception:
                print("[MCP] Unexpected tools container type; expected list.")
                return []

        if tools:
            print("[MCP] Available tools across all servers:")
            for i, tool in enumerate(tools, start=1):
                # Support both LangChain tool schemas and plain dicts
                t_name = getattr(tool, "name", None) or (
                    tool.get("name") if isinstance(tool, dict) else None) or f"tool_{i}"
                t_desc = getattr(tool, "description", None) or (
                    tool.get("description") if isinstance(tool, dict) else None) or ""
                print(f"  - {t_name}: {t_desc}")
        else:
            print("[MCP] No tools exposed by configured servers.")
        return tools
    except Exception as e:
        print(f"[MCP] Error while retrieving tools: {e}")
        return []


# ============ Chat history persistence (SQLite) ============

def _default_db_path() -> Path:
    return Path(__file__).resolve().parent / ".chat_history.sqlite"


def _list_sessions(db_path: Path, table_name: str = "message_store") -> List[str]:
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT DISTINCT session_id FROM {table_name} ORDER BY session_id"
            )
            rows = cur.fetchall()
            return [r[0] for r in rows if r and r[0]]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # Table likely doesn't exist yet
        return []
    except Exception:
        return []


def _choose_session_interactive(db_path: Path) -> str:
    sessions = _list_sessions(db_path)
    print("\nChat sessions in DB:")
    if sessions:
        for i, s in enumerate(sessions, start=1):
            print(f"  {i}. {s}")
    else:
        print("  <no sessions yet>")
    while True:
        raw = input("Select session number or type a new session name: ").strip()
        if raw.isdigit() and sessions:
            idx = int(raw)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]
        if raw:
            return raw
        print("Please enter a valid number or non-empty name.")


def _init_history(db_path: Path, session_id: str):
    if SQLChatMessageHistory is None:
        raise RuntimeError(
            "langchain-community is required for SQLChatMessageHistory. "
            "Install with: uv add langchain-community"
        )
    # Build SQLAlchemy SQLite connection string
    db_abs = db_path.resolve()
    conn_str = f"sqlite:///{db_abs.as_posix()}"
    return SQLChatMessageHistory(session_id=session_id, connection_string=conn_str)


async def stream_once(app, messages: List[BaseMessage]) -> str:
    """Stream a single turn through the graph and print tokens as they arrive.

    Returns the full assistant response content.
    """
    full_text_parts: List[str] = []

    # Stream internal events; token chunks will come via chat model stream events
    async for event in app.astream_events({"messages": messages}, version="v2"):
        ev = event.get("event")
        if ev in ("on_chat_model_stream", "on_llm_stream"):
            data = event.get("data", {})
            chunk = data.get("chunk")
            # chunk may be AIMessageChunk or LLMResult chunk; try to get text
            text = None
            if hasattr(chunk, "content"):
                text = chunk.content
            elif isinstance(chunk, dict):
                text = chunk.get("content")
            if text:
                full_text_parts.append(str(text))
                print(str(text), end="", flush=True)
        elif ev in ("on_chat_model_end", "on_llm_end"):
            # End of streaming for this model call; print newline if anything was printed
            if full_text_parts:
                print()
        elif ev == "on_tool_start":
            data = event.get("data", {})
            name = data.get("name") or data.get("tool") or "<tool>"
            input_data = data.get("input") or data.get("inputs") or {}
            print(f"\n[MCP][Tool start] {name} inputs={input_data}")
        elif ev == "on_tool_end":
            data = event.get("data", {})
            name = data.get("name") or data.get("tool") or "<tool>"
            output = data.get("output") or data.get("result")
            print(f"\n[MCP][Tool end] {name} output={output}")
        elif ev == "on_tool_error":
            data = event.get("data", {})
            name = data.get("name") or data.get("tool") or "<tool>"
            error = data.get("error") or data.get("traceback") or data
            print(f"\n[MCP][Tool error] {name}: {error}", file=sys.stderr)
    return "".join(full_text_parts)


async def interactive_chat(app, history_store: Optional[SQLChatMessageHistory], system_prompt: str):
    print("Interactive chat started. Press Ctrl+C to exit.")
    # Ensure system prompt exists only for new sessions
    if history_store is None:
        # Fallback to ephemeral in-memory list
        in_mem: List[BaseMessage] = [SystemMessage(system_prompt)] if system_prompt else []
    else:
        existing = history_store.messages
        if not existing and system_prompt:
            history_store.add_message(SystemMessage(system_prompt))
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            print("Assistant: ", end="", flush=True)
            if history_store is None:
                in_mem.append(HumanMessage(user_input))
                reply = await stream_once(app, in_mem)
                history_store  # no-op to satisfy type checker
                # Persist assistant message in the in-memory history
                in_mem.append(AIMessage(content=reply))
            else:
                history_store.add_user_message(user_input)
                reply = await stream_once(app, history_store.messages)
                history_store.add_ai_message(reply)
    except KeyboardInterrupt:
        print("\nExiting chat.")


async def single_turn(app, history_store: Optional[SQLChatMessageHistory], system_prompt: str, message: str):
    print("Assistant: ", end="", flush=True)
    if history_store is None:
        # Ephemeral
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(system_prompt))
        messages.append(HumanMessage(message))
        await stream_once(app, messages)
    else:
        # Ensure system prompt for new session
        if not history_store.messages and system_prompt:
            history_store.add_message(SystemMessage(system_prompt))
        history_store.add_user_message(message)
        reply = await stream_once(app, history_store.messages)
        history_store.add_ai_message(reply)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async streaming chat agent using LangGraph and local Ollama (uv-friendly)"
    )
    default_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    default_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    parser.add_argument("--model", default=default_model, help="Ollama model tag (e.g., llama3.1, qwen2.5:7b)")
    parser.add_argument("--ollama-base-url", dest="ollama_base_url", default=default_base,
                        help="Ollama base URL (default from $OLLAMA_BASE_URL or http://localhost:11434)")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for the response (mapped to Ollama num_predict)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful AI assistant.",
        help="System prompt/instructions",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="If provided, run a single-turn chat for this message instead of interactive REPL",
    )
    # Persistence/session options
    parser.add_argument("--db-path", default=str(_default_db_path()),
                        help="Path to SQLite DB file for chat history (default: .chat_history.sqlite)")
    parser.add_argument("--session", default=None, help="Session ID to continue")
    parser.add_argument("--new-session", dest="new_session", default=None, help="Create/use a new session ID")
    parser.add_argument("--list-sessions", action="store_true", help="List existing sessions in the DB and exit")
    parser.add_argument("--no-persist", action="store_true", help="Disable persistence and use in-memory history only")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    llm = init_llm(args.model, args.temperature, args.max_tokens, args.ollama_base_url)

    # Initialize MCP tools from mcp_servers.json (if available) and bind to LLM
    tools = _init_mcp_tools()
    if tools:
        try:
            llm = llm.bind_tools(tools)
        except Exception as e:
            print(f"[WARN] Failed to bind MCP tools to LLM: {e}", file=sys.stderr)

    app = build_graph(llm, tools)

    # Persistence setup
    db_path = Path(args.db_path)
    if args.list_sessions:
        sessions = _list_sessions(db_path)
        if sessions:
            print("Existing sessions:")
            for s in sessions:
                print(f" - {s}")
        else:
            print("No sessions found.")
        return 0

    history_store: Optional[SQLChatMessageHistory] = None
    if not args.no_persist:
        session_id = args.new_session or args.session
        if not session_id:
            if args.message:
                session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                print(f"[Chat] Using new session: {session_id}")
            else:
                session_id = _choose_session_interactive(db_path)
        history_store = _init_history(db_path, session_id)
        print(f"[Chat] DB: {db_path} | Session: {session_id}")
    else:
        print("[Chat] Persistence disabled (in-memory only).")

    async def runner():
        if args.message:
            await single_turn(app, history_store, args.system, args.message)
        else:
            await interactive_chat(app, history_store, args.system)

    try:
        asyncio.run(runner())
    except RuntimeError as e:
        # Fallback for nested event loops if any environment already has a loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(runner())
        except Exception:
            print(f"[ERROR] Async runtime failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
