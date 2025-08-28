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
    "You are a senior Python engineer AI agent that fixes syntax and import errors. "
    "Return ONLY a unified diff/patch for the specified file, with minimal necessary changes. "
    "Rules: Output a single fenced code block labeled 'diff' or 'patch' containing a standard unified diff. "
    "Do not include any explanations or full file content outside the diff."
)


def extract_code_blocks(s: str) -> str:
    """(Legacy) Extract python code from ```python blocks; if none, return the text as-is."""
    import re
    blocks: List[str] = []
    for m in re.finditer(r"```(?:python)?\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE):
        blocks.append(m.group(1).strip())
    if blocks:
        return "\n\n".join(blocks).strip()
    return s.strip()


def extract_fenced_blocks(s: str) -> List[Tuple[str, str]]:
    """Extract all fenced code blocks, returning list of (lang_lower, content).
    If language is missing, lang_lower is an empty string.
    """
    import re
    results: List[Tuple[str, str]] = []
    for m in re.finditer(r"```([a-zA-Z0-9_-]*)\n(.*?)```", s, flags=re.DOTALL):
        lang = (m.group(1) or "").strip().lower()
        content = m.group(2)
        results.append((lang, content.strip()))
    return results


def extract_unified_diff(s: str) -> Optional[str]:
    """Try to extract a unified diff/patch from text.
    Priority: a fenced block with lang in {diff, patch, udiff}, else any fenced block containing '---' and '+++', else the raw text if it looks like a diff.
    Returns None if no diff found.
    """
    blocks = extract_fenced_blocks(s)
    # 1) Prefer explicitly labeled diff/patch blocks
    for lang, content in blocks:
        if lang in {"diff", "patch", "udiff"}:
            if "---" in content and "+++" in content:
                return content.strip()
    # 2) Any block that contains a unified diff header
    for _, content in blocks:
        if "---" in content and "+++" in content and "@@" in content:
            return content.strip()
    # 3) Fallback to full text if it looks like a diff
    text = s.strip()
    if text.startswith("diff ") or text.startswith("--- ") or "@@" in text:
        return text
    return None


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
    You must produce a unified diff/patch that fixes the issues in the target file.

    Target file path: {analysis.path}

    Issues detected:
    {textwrap.indent('\n'.join(issues_desc), '  ')}

    Instructions:
    - Output ONLY a single fenced code block labeled diff or patch.
    - Use standard unified diff format with headers:
      --- a/{analysis.path.name}
      +++ b/{analysis.path.name}
    - Include one or more @@ hunks with minimal necessary changes.
    - Do NOT include the full file; only the diff.

    Original file content:
    ---
    {analysis.source}
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
    diff_text = extract_unified_diff(content)
    if not diff_text or not diff_text.strip():
        return None
    return diff_text.strip()


# --------------------------- Unified Diff Apply ---------------------------
import re as _re

def _parse_unified_diff(diff_text: str):
    """Parse a unified diff text, returning (old_file, new_file, hunks).
    Each hunk is a dict with keys: orig_start, orig_count, new_start, new_count, lines [(kind, text)].
    """
    old_file = None
    new_file = None
    hunks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("--- "):
            old_file = line[4:].strip()
            continue
        if line.startswith("+++ "):
            new_file = line[4:].strip()
            continue
        m = _re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            if current is not None:
                hunks.append(current)
            current = {
                "orig_start": int(m.group(1)),
                "orig_count": int(m.group(2) or "1"),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or "1"),
                "lines": [],
            }
            continue
        if current is not None and line == "\\ No newline at end of file":
            # Record a marker indicating the previous line (addition/deletion) had no trailing newline
            current["lines"].append(("\\", ""))
            continue
        if current is not None and (line.startswith(" ") or line.startswith("-") or line.startswith("+")):
            current["lines"].append((line[0], line[1:]))
            continue
        # ignore other lines (e.g., index, diff --git)
    if current is not None:
        hunks.append(current)
    return old_file, new_file, hunks


def apply_unified_diff_to_text(original_text: str, diff_text: str) -> Tuple[bool, Optional[str], str]:
    """Apply a unified diff to the provided original_text.
    Returns (ok, new_text_or_None, error_message).
    This is a minimal, strict applier that validates context lines.
    """
    _, _, hunks = _parse_unified_diff(diff_text)
    if not hunks:
        return False, None, "No hunks found in patch."

    orig_lines = original_text.splitlines(keepends=True)
    out_lines: List[str] = []
    cur = 0  # index in orig_lines

    def _strip_eol(s: str) -> str:
        return s[:-1] if s.endswith("\n") or s.endswith("\r") else s

    for h in hunks:
        start = max(0, h["orig_start"] - 1)
        # Append unchanged section before hunk
        if start > len(orig_lines):
            return False, None, f"Hunk starts beyond end of file at line {start+1}."
        out_lines.extend(orig_lines[cur:start])
        cur = start
        # Apply hunk body
        last_op = None
        for kind, text in h["lines"]:
            if kind == " ":
                if cur >= len(orig_lines):
                    return False, None, "Context exceeds original length."
                if _strip_eol(orig_lines[cur]) != text:
                    return False, None, f"Context mismatch at line {cur+1}."
                out_lines.append(orig_lines[cur])
                cur += 1
                last_op = " "
            elif kind == "-":
                if cur >= len(orig_lines):
                    return False, None, "Deletion exceeds original length."
                if _strip_eol(orig_lines[cur]) != text:
                    return False, None, f"Deletion mismatch at line {cur+1}."
                # skip (delete) this line
                cur += 1
                last_op = "-"
            elif kind == "+":
                # insert this line; default to newline-terminated; may be adjusted by a following '\\ No newline at end of file' marker
                eol = "\n"
                out_lines.append(text + eol)
                last_op = "+"
            elif kind == "\\":
                # Honor no-newline marker for the previous line (usually addition)
                if last_op == "+" and out_lines:
                    # remove trailing newline just added
                    if out_lines[-1].endswith("\r\n"):
                        out_lines[-1] = out_lines[-1][:-2]
                    elif out_lines[-1].endswith("\n"):
                        out_lines[-1] = out_lines[-1][:-1]
                # no change to cur; marker doesn't consume original lines
                last_op = "\\"
            else:
                return False, None, "Unknown hunk line kind."
    # Append the rest of original file after last hunk
    out_lines.extend(orig_lines[cur:])
    return True, "".join(out_lines), ""

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

        # Show diff preview (first 80 lines)
        preview = "\n".join(proposed.splitlines()[:80])
        print("\nProposed patch (first lines):\n" + preview)

        should_apply = False
        if args.assume_yes:
            should_apply = True
        else:
            try:
                ans = input("Apply this patch? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            should_apply = ans in ("y", "yes")

        if should_apply:
            # Apply unified diff to original content
            ok, new_text, err = apply_unified_diff_to_text(analysis.source, proposed)
            if not ok or new_text is None:
                print(f"  [ERROR] Failed to apply patch: {err or 'unknown error'}")
                continue
            # Backup and write
            bak = analysis.path.with_suffix(analysis.path.suffix + ".bak")
            try:
                if not bak.exists():
                    bak.write_text(analysis.source, encoding="utf-8")
                analysis.path.write_text(new_text, encoding="utf-8")
                any_applied = True
                print(f"  [OK] Patch applied. Backup saved at {bak}")
            except Exception as e:
                print(f"  [ERROR] Failed to write patched file: {e}")
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
