from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast
import importlib.util
import textwrap

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# --------------------------- LLM Init (OpenRouter) ---------------------------

def init_llm(model: str, temperature: float, max_tokens: Optional[int]) -> ChatOpenAI:
    """Initialize ChatOpenAI client configured to use OpenRouter.

    Follows guidance from https://openrouter.ai/docs/community/lang-chain
    Same conventions as agent_chat.py for consistency across repo.
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


# --------------------------- Project Scanning ---------------------------

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "site-packages", "dist", "build"}


def iter_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # skip hidden directories/files under root
        parts = list(p.parts)
        if any(s.startswith('.') and s not in ('.', '..') for s in parts):
            # allow root itself to be hidden if user chose so; still skip hidden nested dirs
            pass
        files.append(p)
    return files


@dataclass
class SyntaxIssue:
    file: Path
    lineno: int
    offset: int
    msg: str
    line: str


@dataclass
class ImportIssue:
    file: Path
    module: str
    level: int  # 0 for absolute; >0 for relative
    name: Optional[str]  # for from X import name
    lineno: int
    suggestion: Optional[str] = None


@dataclass
class FileAnalysis:
    path: Path
    syntax_error: Optional[SyntaxIssue]
    import_issues: List[ImportIssue]
    source: str


def analyze_file(path: Path, project_root: Path) -> FileAnalysis:
    source = path.read_text(encoding="utf-8", errors="replace")
    syntax_err: Optional[SyntaxIssue] = None
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        line = ""
        try:
            line = source.splitlines()[e.lineno - 1] if e.lineno and e.lineno - 1 < len(source.splitlines()) else ""
        except Exception:
            line = ""
        syntax_err = SyntaxIssue(
            file=path, lineno=e.lineno or 0, offset=e.offset or 0, msg=e.msg, line=line
        )

    import_issues: List[ImportIssue] = []
    if tree is not None:
        # Collect import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    full = alias.name
                    if not resolve_module(full, project_root):
                        import_issues.append(
                            ImportIssue(file=path, module=full, level=0, name=None, lineno=node.lineno)
                        )
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                mod = node.module or ""
                # If relative, attempt resolve relative module path
                if level > 0:
                    if not resolve_relative_import(path, project_root, level, mod):
                        for alias in node.names:
                            import_issues.append(
                                ImportIssue(file=path, module=mod or ".", level=level, name=alias.name, lineno=node.lineno)
                            )
                else:
                    # absolute from X import Y -> verify base module only
                    base_ok = resolve_module(mod, project_root)
                    if not base_ok:
                        for alias in node.names:
                            import_issues.append(
                                ImportIssue(file=path, module=mod, level=0, name=alias.name, lineno=node.lineno)
                            )

    return FileAnalysis(path=path, syntax_error=syntax_err, import_issues=import_issues, source=source)


def resolve_module(name: str, project_root: Path) -> bool:
    """Try to resolve a module without importing project code.

    - Prefer checking inside project_root for packages or modules
    - Fallback to importlib.util.find_spec
    """
    # Quick path check inside project root
    parts = name.split(".")
    base = project_root.joinpath(*parts)
    if base.with_suffix(".py").exists():
        return True
    if base.is_dir() and (base / "__init__.py").exists():
        return True
    # try to resolve via import system with project_root added to sys.path (temporarily)
    sys_path_added = False
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            sys_path_added = True
        spec = importlib.util.find_spec(name)
        return spec is not None
    except Exception:
        return False
    finally:
        if sys_path_added:
            try:
                sys.path.remove(str(project_root))
            except ValueError:
                pass


def resolve_relative_import(file_path: Path, project_root: Path, level: int, module: str) -> bool:
    # Determine package directory for the file: walk up while __init__.py exists
    pkg_dir = file_path.parent
    for _ in range(level):
        pkg_dir = pkg_dir.parent
    # If module is provided, append its parts
    if module:
        candidate = pkg_dir.joinpath(*module.split("."))
    else:
        candidate = pkg_dir
    # Check for module.py or package/__init__.py
    if candidate.with_suffix(".py").exists():
        return True
    if candidate.is_dir() and (candidate / "__init__.py").exists():
        return True
    # As a fallback, try to resolve as absolute from that path
    rel_pkg_root = _find_package_root(file_path.parent)
    if rel_pkg_root:
        abs_name = _path_to_module_name(candidate, rel_pkg_root)
        if abs_name and resolve_module(abs_name, project_root):
            return True
    return False


def _find_package_root(start: Path) -> Optional[Path]:
    cur = start
    last_pkg: Optional[Path] = None
    while True:
        if (cur / "__init__.py").exists():
            last_pkg = cur
        else:
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    return last_pkg


def _path_to_module_name(path: Path, pkg_root: Path) -> Optional[str]:
    try:
        rel = path.relative_to(pkg_root.parent)
    except Exception:
        return None
    parts = list(rel.parts)
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


# --------------------------- LLM Fix Suggestions ---------------------------

SYSTEM_PROMPT = (
    "You are a senior Python engineer AI agent that fixes syntax errors and import errors. "
    "You will be given the file content and a list of issues. Return the corrected full file content. "
    "Rules: produce ONLY the corrected code wrapped in a single ```python code block. No extra commentary." 
)


def extract_code_blocks(s: str) -> str:
    """Extract python code from ```python blocks; if none, return the text as-is."""
    import re
    blocks: List[str] = []
    for m in re.finditer(r"```(?:python)?\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1).strip())
    if blocks:
        return "\n\n".join(blocks).strip()
    return s.strip()


def _extract_token_usage(resp: Any) -> Tuple[int, int, int]:
    """Extract (input_tokens, output_tokens, total_tokens) from a LangChain AIMessage.

    Priority:
    1) resp.usage_metadata: standardized by LangChain
       - keys: input_tokens, output_tokens, total_tokens
    2) resp.response_metadata: provider-specific
       - openai-compatible: token_usage or usage with prompt_tokens, completion_tokens, total_tokens
    Returns zeros if not available.
    """
    try:
        # 1) Standardized
        usage = getattr(resp, "usage_metadata", None)
        if isinstance(usage, dict):
            it = int(usage.get("input_tokens") or 0)
            ot = int(usage.get("output_tokens") or 0)
            tt = int(usage.get("total_tokens") or (it + ot))
            return it, ot, tt

        # 2) Provider-specific fallbacks
        meta = getattr(resp, "response_metadata", None)
        if isinstance(meta, dict):
            token_usage = meta.get("token_usage") or meta.get("usage")
            if isinstance(token_usage, dict):
                # OpenAI-style
                it = int(token_usage.get("prompt_tokens") or 0)
                ot = int(token_usage.get("completion_tokens") or 0)
                tt = int(token_usage.get("total_tokens") or (it + ot))
                return it, ot, tt
    except Exception:
        pass
    return 0, 0, 0


async def propose_fix_for_file(llm: ChatOpenAI, analysis: FileAnalysis) -> Optional[str]:
    issues_desc: List[str] = []
    if analysis.syntax_error:
        se = analysis.syntax_error
        issues_desc.append(
            f"SyntaxError at line {se.lineno}, col {se.offset}: {se.msg}\nLine: {se.line.strip()}"
        )
    if analysis.import_issues:
        for ii in analysis.import_issues:
            if ii.level > 0:
                issues_desc.append(
                    f"Relative import may be unresolved at line {ii.lineno}: level={ii.level}, module='{ii.module}', name='{ii.name or ''}'"
                )
            else:
                issues_desc.append(
                    f"Import may be missing at line {ii.lineno}: module='{ii.module}'"
                )
    if not issues_desc:
        return None

    # Build a compact context with around-the-error snippet for syntax errors
    context_snippet = analysis.source
    if analysis.syntax_error and analysis.syntax_error.lineno:
        lines = analysis.source.splitlines()
        idx = max(0, analysis.syntax_error.lineno - 1)
        start = max(0, idx - 10)
        end = min(len(lines), idx + 10)
        snippet = "\n".join(lines[start:end])
        context_snippet = snippet

    user_prompt = f"""
    File path: {analysis.path}

    Issues detected:
    {textwrap.indent('\n'.join(issues_desc), '  ')}

    Provide a corrected, complete Python file that resolves all issues. Use minimal changes.
    Return ONLY the corrected code in a single fenced code block.

    Original (or relevant) content:
    ---
    {context_snippet}
    ---
    """.strip()

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)]
    try:
        resp = await llm.ainvoke(messages)
    except Exception as e:
        print(f"[ERROR] LLM call failed for {analysis.path}: {e}", file=sys.stderr)
        return None

    # Token accounting and console output
    in_tok, out_tok, total_tok = _extract_token_usage(resp)
    try:
        model_id = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "unknown-model"
    except Exception:
        model_id = "unknown-model"
    print(f"[TOKENS] file={analysis.path} | model={model_id} | input={in_tok} | output={out_tok} | total={total_tok}")

    content = getattr(resp, "content", "")
    fixed = extract_code_blocks(content)
    if not fixed.strip():
        return None
    return fixed


# --------------------------- CLI and Orchestration ---------------------------

@dataclass
class Args:
    path: Path
    model: str
    temperature: float
    max_tokens: Optional[int]
    apply: bool
    assume_yes: bool


async def run(args: Args) -> int:
    # Lazy-initialize LLM only if needed and API key is present
    llm: Optional[ChatOpenAI] = None

    # Scan project
    root = args.path.resolve()
    if not root.exists():
        print(f"[ERROR] Path not found: {root}", file=sys.stderr)
        return 2
    files = iter_python_files(root if root.is_dir() else root.parent)
    if root.is_file() and root.suffix == ".py":
        files = [root]

    analyses: List[FileAnalysis] = []
    # Perform analysis (can be CPU bound; do in thread pool if many files)
    for p in files:
        try:
            analyses.append(await asyncio.to_thread(analyze_file, p, root))
        except Exception as e:
            print(f"[WARN] Failed to analyze {p}: {e}")

    # Filter files with issues
    targets = [a for a in analyses if a.syntax_error or a.import_issues]
    if not targets:
        print("No syntax or import issues detected.")
        return 0

    print(f"Found {len(targets)} file(s) with potential issues.")

    # Initialize LLM if API key is available; otherwise skip proposals
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            llm = init_llm(args.model, args.temperature, args.max_tokens)
        except Exception as e:
            print(f"[WARN] Failed to initialize LLM. Will skip proposals: {e}")
            llm = None
    else:
        print("[INFO] OPENROUTER_API_KEY/OPENAI_API_KEY not set. Will report issues without proposing fixes.")

    # Propose fixes
    any_applied = False
    for analysis in targets:
        print(f"\n---\nFile: {analysis.path}")
        # Summarize issues
        if analysis.syntax_error:
            se = analysis.syntax_error
            print(f"  SyntaxError at line {se.lineno}, col {se.offset}: {se.msg}")
        if analysis.import_issues:
            for ii in analysis.import_issues[:10]:
                print(f"  Import: line {ii.lineno} -> module='{ii.module}', level={ii.level}")
            if len(analysis.import_issues) > 10:
                print(f"  ... and {len(analysis.import_issues) - 10} more import warnings")

        proposed: Optional[str] = None
        if llm is not None:
            proposed = await propose_fix_for_file(llm, analysis)
        else:
            print("  [INFO] Skipping LLM proposal (no API key).")

        if not proposed:
            if llm is not None:
                print("  [INFO] No fix proposed by LLM or call failed.")
            continue

        # Show a diff-like preview (first 40 lines)
        preview = "\n".join(proposed.splitlines()[:40])
        print("\nProposed fix (first lines):\n" + preview)

        should_apply = False
        if args.assume_yes:
            # Auto-apply all proposed fixes without prompting
            should_apply = True
        else:
            # Ask by default, regardless of --apply flag
            try:
                ans = input("Apply this fix? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            should_apply = ans in ("y", "yes")

        if should_apply:
            # Backup and write
            bak = analysis.path.with_suffix(analysis.path.suffix + ".bak")
            try:
                if not bak.exists():
                    bak.write_text(analysis.source, encoding="utf-8")
                analysis.path.write_text(proposed, encoding="utf-8")
                any_applied = True
                print(f"  [OK] Applied fix. Backup saved at {bak}")
            except Exception as e:
                print(f"  [ERROR] Failed to write fix: {e}")
        else:
            print("  [SKIP] Not applied.")

    if any_applied:
        print("\nSome fixes were applied. Re-run the scanner to verify.")
    else:
        print("\nNo changes were applied.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description="Async agent to find and fix Python syntax/import errors using LangChain via OpenRouter")
    parser.add_argument("--path", "-p", required=True, help="Path to project directory or a single .py file")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model ID for OpenRouter")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for response")
    parser.add_argument("--apply", action="store_true", help="[Deprecated] Prompts are shown by default. Use --yes to auto-apply without prompting.")
    parser.add_argument("--yes", action="store_true", help="Auto-apply all proposed fixes without prompting.")

    ns = parser.parse_args(argv)
    return Args(
        path=Path(ns.path),
        model=ns.model,
        temperature=float(ns.temperature),
        max_tokens=ns.max_tokens,
        apply=bool(ns.apply),
        assume_yes=bool(ns.yes),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    async def runner():
        return await run(args)

    try:
        return asyncio.run(runner())
    except RuntimeError as e:
        # Handle environments with existing event loop
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(runner())
        except Exception:
            print(f"[ERROR] Async runtime failed: {e}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
