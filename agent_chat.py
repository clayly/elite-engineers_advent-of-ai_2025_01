from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.constants import START
import json
from pathlib import Path

# MCP client for multiple servers
from langchain_mcp_adapters.client import MultiServerMCPClient

# LangGraph imports
from langgraph.graph import StateGraph, END, MessagesState


def init_llm(model: str, temperature: float, max_tokens: Optional[int]) -> ChatOpenAI:
    """Initialize ChatOpenAI client configured to use OpenRouter.

    Follows guidance from https://openrouter.ai/docs/community/lang-chain
    and mirrors conventions used elsewhere in this repo.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    # Recommended extra headers
    default_headers: Dict[str, str] = {}
    site = os.environ.get("OPENROUTER_SITE_URL")
    app = os.environ.get("OPENROUTER_APP_NAME")
    if site:
        default_headers["HTTP-Referer"] = site
    if app:
        default_headers["X-Title"] = app

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if default_headers:
        kwargs["default_headers"] = default_headers
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenAI(**kwargs)


def build_graph(llm: ChatOpenAI):
    """Build a minimal LangGraph that appends model responses to message history."""
    graph = StateGraph(MessagesState)

    async def call_model(state: MessagesState):
        # LLM is a ChatModel; pass the conversation so far and get one response
        response = await llm.ainvoke(state["messages"])  # returns an AIMessage
        return {"messages": [response]}

    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)

    return graph.compile()


# ============ MCP Integration (optional) ============

def _load_mcp_servers_config(config_path: Path) -> List[Dict[str, Any]]:
    try:
        if not config_path.exists():
            return []
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # ensure elements are dicts
            return [s for s in data if isinstance(s, dict)]
    except Exception:
        pass
    return []


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
    for idx, srv in enumerate(servers, start=1):
        name = srv.get("name") or "<unnamed>"
        command = srv.get("command") or "<command?>"
        transport = srv.get("transport") or "<transport?>"
        print(f"  {idx}. {name} | command: {command} | transport: {transport}")

    # Initialize client if available
    try:
        client = MultiServerMCPClient(servers=servers)
    except Exception as e:
        print(f"[MCP] Failed to initialize MultiServerMCPClient: {e}")
        return []

    # Retrieve tools via get_tools or attribute and print summary
    try:
        get_tools = getattr(client, "get_tools", None)
        if callable(get_tools):
            tools = get_tools()
        else:
            tools = getattr(client, "tools", [])
        if not isinstance(tools, list):
            print("[MCP] Unexpected tools container type; expected list.")
            return []

        if tools:
            print("[MCP] Available tools across all servers:")
            for i, tool in enumerate(tools, start=1):
                # Support both LangChain tool schemas and plain dicts
                t_name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None) or f"tool_{i}"
                t_desc = getattr(tool, "description", None) or (tool.get("description") if isinstance(tool, dict) else None) or ""
                print(f"  - {t_name}: {t_desc}")
        else:
            print("[MCP] No tools exposed by configured servers.")
        return tools
    except Exception as e:
        print(f"[MCP] Error while retrieving tools: {e}")
        return []


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
    return "".join(full_text_parts)


async def interactive_chat(app, system_prompt: str):
    print("Interactive chat started. Press Ctrl+C to exit.")
    history: List[BaseMessage] = [SystemMessage(system_prompt)] if system_prompt else []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            history.append(HumanMessage(user_input))
            print("Assistant: ", end="", flush=True)
            reply = await stream_once(app, history)
            # Persist assistant message in history for multi-turn context
            from langchain_core.messages import AIMessage
            history.append(AIMessage(content=reply))
    except KeyboardInterrupt:
        print("\nExiting chat.")


async def single_turn(app, system_prompt: str, message: str):
    messages: List[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(system_prompt))
    messages.append(HumanMessage(message))
    print("Assistant: ", end="", flush=True)
    await stream_once(app, messages)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async streaming chat agent using LangGraph and OpenRouter"
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model ID for OpenRouter")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for the response (model-dependent limits apply)",
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    llm = init_llm(args.model, args.temperature, args.max_tokens)

    # Initialize MCP tools from mcp_servers.json (if available) and bind to LLM
    tools = _init_mcp_tools()
    if tools:
        try:
            llm = llm.bind_tools(tools)
        except Exception as e:
            print(f"[WARN] Failed to bind MCP tools to LLM: {e}", file=sys.stderr)

    app = build_graph(llm)

    async def runner():
        if args.message:
            await single_turn(app, args.system, args.message)
        else:
            await interactive_chat(app, args.system)

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
