#!/usr/bin/env python3
"""
Lightweight file CRUD tool-calling agent for Ollama on resource-sparse environments.

Gives the LLM proper Create / Read / Update / Delete capabilities over a sandboxed workspace.

Reusable via the FileAgent class for testing, evaluation, and embedding in other flows.

Key features:
- list / read (ranged) / write / append / edit (safe search-replace) / delete
- Proper multi-turn tool calling with result feedback
- Strong sandbox + size limits
- Designed so small models can be iteratively improved for this specific file-agent purpose
"""

import json
from pathlib import Path
from typing import Any

try:
    import ollama
except ImportError:
    ollama = None  # Will only be needed for real LLM chat, not for direct tool functions or mock mode

WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

# Tunables for sparse environments
MAX_CONTENT_CHARS = 150_000
MAX_READ_LINES = 400


def safe_resolve(rel_path: str, base: Path | None = None) -> Path:
    """Resolve user path under the workspace. Blocks traversal."""
    ws = base or WORKSPACE
    p = (ws / rel_path).resolve()
    if not str(p).startswith(str(ws.resolve())):
        raise PermissionError(f"Path escapes workspace: {rel_path}")
    return p


def list_files(path: str = ".", base: Path | None = None) -> str:
    try:
        ws = base or WORKSPACE
        target = safe_resolve(path, ws)
        if not target.exists():
            return f"ERROR: '{path}' does not exist"
        if not target.is_dir():
            return f"ERROR: '{path}' is not a directory"

        lines = []
        for entry in sorted(target.iterdir()):
            try:
                if entry.is_dir():
                    lines.append(f"DIR  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    lines.append(f"FILE {entry.name} ({size}B)")
            except Exception:
                lines.append(f"???  {entry.name}")
        return "\n".join(lines) if lines else "(empty)"
    except Exception as e:
        return f"ERROR listing: {e}"


def read_file(path: str, start_line: int | None = None, num_lines: int | None = None, base: Path | None = None) -> str:
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        if not p.exists():
            return f"ERROR: File '{path}' does not exist"
        if not p.is_file():
            return f"ERROR: '{path}' is not a regular file"

        text = p.read_text(encoding="utf-8", errors="replace")

        if len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS] + "\n... [truncated - file too large for full read]"

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        start_idx = 0 if start_line is None else max(0, start_line - 1)
        if num_lines is None:
            end_idx = min(start_idx + MAX_READ_LINES, total_lines)
        else:
            end_idx = min(start_idx + num_lines, total_lines)

        slice_text = "".join(lines[start_idx:end_idx])
        header = f"=== {path} (lines {start_idx + 1}-{end_idx} / {total_lines}) ===\n"
        return header + slice_text
    except Exception as e:
        return f"ERROR reading '{path}': {e}"


def write_file(path: str, content: str, base: Path | None = None) -> str:
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        if len(content) > MAX_CONTENT_CHARS:
            return f"ERROR: content too large ({len(content)} chars). Max {MAX_CONTENT_CHARS}"

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing '{path}': {e}"


def append_file(path: str, content: str, base: Path | None = None) -> str:
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        p.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if p.exists():
            existing = p.read_text(encoding="utf-8", errors="replace")
            if len(existing) + len(content) > MAX_CONTENT_CHARS:
                return "ERROR: append would exceed size limit"

        p.write_text(existing + content, encoding="utf-8")
        return f"OK: appended {len(content)} chars to {path} (new size ~{len(existing) + len(content)})"
    except Exception as e:
        return f"ERROR appending to '{path}': {e}"


def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False, base: Path | None = None) -> str:
    """Precise search-and-replace. Forces models to use unique context."""
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        if not p.exists():
            return f"ERROR: '{path}' does not exist. Create it with write_file first."

        text = p.read_text(encoding="utf-8", errors="replace")

        if old_string not in text:
            return ("ERROR: old_string not found. "
                    "Read the file first, then copy a sufficiently long exact snippet.")

        count = text.count(old_string)
        if count > 1 and not replace_all:
            return (f"ERROR: old_string appears {count} times. "
                    "Make old_string longer and more unique (include surrounding lines), "
                    "or set replace_all=true.")

        new_text = text.replace(old_string, new_string, count if replace_all else 1)

        if len(new_text) > MAX_CONTENT_CHARS:
            return "ERROR: resulting file would exceed size limit"

        p.write_text(new_text, encoding="utf-8")
        return f"OK: replaced {count if replace_all else 1} occurrence(s) in {path}"
    except Exception as e:
        return f"ERROR editing '{path}': {e}"


def delete_file(path: str, base: Path | None = None) -> str:
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        if not p.exists():
            return f"ERROR: '{path}' does not exist"

        if p.is_dir():
            try:
                p.rmdir()
                return f"OK: removed empty directory {path}"
            except OSError:
                return f"ERROR: '{path}' is not empty. Delete contents first."
        p.unlink()
        return f"OK: deleted {path}"
    except Exception as e:
        return f"ERROR deleting '{path}': {e}"


def mkdir(path: str, base: Path | None = None) -> str:
    """Create a directory (and parents). Useful for organizing projects."""
    try:
        ws = base or WORKSPACE
        p = safe_resolve(path, ws)
        p.mkdir(parents=True, exist_ok=True)
        return f"OK: ensured directory {path}"
    except Exception as e:
        return f"ERROR creating directory '{path}': {e}"


TOOL_IMPLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "mkdir": mkdir,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the workspace (or a subfolder). Call this whenever you need to know what currently exists before reading or editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path, e.g. '.' or 'src/components'. Defaults to workspace root."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. For big files use start_line + num_lines (1-based) to read only what you need. Always read before you edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root"},
                    "start_line": {"type": "integer", "description": "1-based first line to include"},
                    "num_lines": {"type": "integer", "description": "Maximum number of lines to return"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a brand new file or completely replace the contents of an existing one. Prefer edit_file when only a small part needs to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "The complete new file contents"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Add text to the end of a file (creates the file if missing). Useful for logs, accumulating output, or appending sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Make a targeted change by replacing exact text. old_string must match exactly. If it appears more than once, increase context in old_string or use replace_all=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact substring that will be replaced (must currently exist in the file)"},
                    "new_string": {"type": "string", "description": "Text that will take its place"},
                    "replace_all": {"type": "boolean", "description": "If true, replace every match. Default: false (safer)."}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file or an empty directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Create a directory (including any missing parent directories). Use this when you need to organize files into folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of the directory to create"}
                },
                "required": ["path"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a careful, methodical file and code assistant with tool access to a sandboxed workspace folder.

MANDATORY WORKFLOW (follow this every time):
1. Use list_files to discover current state before assuming anything.
2. Use read_file (or a precise slice with start_line/num_lines) to see the actual content before any modification.
3. For small or precise changes, strongly prefer edit_file over write_file.
4. After any write/edit/append, you may read the result again to verify.
5. Only use write_file when creating brand new files or when a complete rewrite is genuinely required.
6. Use mkdir when you need to create folders to organize files.

Rules:
- All paths are relative to the workspace root (e.g. "notes.txt", "src/main.py", "data/log.txt").
- In edit_file, old_string must be an exact match. Include enough surrounding text to make it unique.
- Keep files small and focused — this environment has limited resources.
- When the task is finished, give a short, clear summary of what you created or changed.
"""


def execute_tool(tc: dict, base: Path | None = None) -> str:
    name = tc.get("function", {}).get("name")
    args = tc.get("function", {}).get("arguments", {})

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}

    impl = TOOL_IMPLS.get(name)
    if not impl:
        return f"ERROR: unknown tool '{name}'"

    try:
        # Pass base through for workspace isolation in tests
        if name == "list_files":
            return impl(args.get("path", "."), base=base)
        if name == "read_file":
            return impl(args.get("path"), args.get("start_line"), args.get("num_lines"), base=base)
        if name in ("write_file", "append_file"):
            return impl(args.get("path"), args.get("content", ""), base=base)
        if name == "edit_file":
            return impl(
                args.get("path"),
                args.get("old_string", ""),
                args.get("new_string", ""),
                bool(args.get("replace_all", False)),
                base=base
            )
        if name == "delete_file":
            return impl(args.get("path"), base=base)
        if name == "mkdir":
            return impl(args.get("path"), base=base)
        return impl(**{k: v for k, v in args.items()})
    except Exception as e:
        return f"ERROR in {name}: {e}"


class FileAgent:
    """Reusable file CRUD agent. Great for interactive use and for automated evaluation."""

    def __init__(self, model: str = "lfm2.5-thinking", workspace: Path | str | None = None):
        global WORKSPACE
        if workspace is not None:
            WORKSPACE = Path(workspace).resolve()
            WORKSPACE.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools_used_in_last_turn: list[dict] = []

    def reset(self):
        """Clear conversation history (keeps same workspace and model)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools_used_in_last_turn = []

    def send(self, user_message: str, max_tool_rounds: int = 8) -> dict[str, Any]:
        """
        Send a natural language request. The agent will keep calling tools
        (and receiving results) until the model produces a final answer.
        Returns a dict with 'assistant' text, 'tools_used', and full transcript snapshot.
        """
        self.messages.append({"role": "user", "content": user_message})
        tools_used: list[dict] = []
        assistant_text = ""

        for _ in range(max_tool_rounds):
            if ollama is None:
                raise RuntimeError("ollama package not installed. Install with `pip install ollama` and make sure the ollama server is running for real LLM mode.")
            resp = ollama.chat(model=self.model, messages=self.messages, tools=TOOLS)
            msg = resp.get("message", {}) or {}
            self.messages.append(msg)

            if msg.get("content"):
                assistant_text = msg["content"].strip()

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            for tc in tool_calls:
                # Use the current global WORKSPACE (FileAgent may have switched it)
                result = execute_tool(tc, base=WORKSPACE)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result
                })

                name = tc.get("function", {}).get("name", "?")
                args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                tools_used.append({
                    "name": name,
                    "args": args,
                    "result": result
                })

        self.tools_used_in_last_turn = tools_used
        return {
            "assistant": assistant_text,
            "tools_used": tools_used,
            "messages": list(self.messages),  # snapshot for debugging,
        }

    def get_workspace_snapshot(self) -> str:
        return list_files(".", base=WORKSPACE)

    def read_file(self, path: str, **kwargs) -> str:
        """Convenience wrapper."""
        return read_file(path, base=WORKSPACE, **kwargs)


def main():
    model = "lfm2.5-thinking"
    print(f"File CRUD agent ready. Model={model}")
    print(f"Workspace: {WORKSPACE}")
    print("Human helpers: /ls   /read <path>   /clear   (or just chat normally)")
    print("Type 'exit' to quit.\n")

    agent = FileAgent(model=model)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            break

        # Human-side shortcuts (bypass LLM, useful for debugging)
        if user.startswith("/"):
            cmd = user[1:].strip()
            if cmd == "ls" or cmd.startswith("ls "):
                target = cmd[3:].strip() or "."
                print(list_files(target))
                continue
            if cmd.startswith("read "):
                target = cmd[5:].strip()
                print(read_file(target))
                continue
            if cmd == "clear":
                agent.reset()
                print("(conversation history cleared)")
                continue
            print("Unknown command. Available: /ls  /read <path>  /clear")
            continue

        if ollama is None:
            print("ERROR: The 'ollama' Python package is not installed in this environment.")
            print("       For interactive use: pip install ollama")
            print("       You can still use --mock mode in test_agent.py to develop tests and reports.")
            break
        result = agent.send(user)

        if result["assistant"]:
            print("Model:", result["assistant"])

        for t in result["tools_used"]:
            name = t["name"]
            preview = str(t["result"]).replace("\n", " ")[:160]
            if len(str(t["result"])) > 160:
                preview += "..."
            print(f"[tool {name}] {preview}")

        print()


if __name__ == "__main__":
    main()
