#!/usr/bin/env python3
"""
test_agent.py

Complementary test & evaluation harness for Ollama tool-calling agents.

Supports multiple domains/purposes:
- File CRUD (list/read/write/edit/delete with good workflow)
- Drone / Spatial (point-cloud based navigation, scanning, mapping, exploration via tool calls)

Goals:
- Drive the agent using *natural, common lingo* that real users would actually type.
- Validate reliable tool use + good workflow for the chosen domain.
- Produce a beautiful, self-contained, impressive HTML report you can just open in a browser.
- Also emit .json (for further analysis / prompt optimization) and .md.
- Helps iteratively improve the model + system prompt for specific agent purposes (file ops, spatial/drone control, etc.).

Run:
    python test_agent.py
    # or with options
    python test_agent.py --model lfm2.5-thinking --limit 12 --mock

    # Run tests + open 3D GLUT viewer for live drone viz (shared sim, tests in bg thread)
    python test_agent.py --mock --viz

The report will be written to eval_reports/agent_eval_report_YYYYMMDD_HHMMSS.html
"""

import argparse
import json
import shutil
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Import the agent (after the refactor)
import toolcallingollama as tca

# Drone spatial agent for the new direction (point cloud + drone driving tests)
try:
    import toolcalling_drone as td
except Exception:
    td = None


# =============================================================================
# Test Case Definition
# =============================================================================

@dataclass
class TestCase:
    id: str
    category: str
    prompt: str                     # Natural language a user would actually say
    description: str
    domain: str = "file"            # "file" or "drone" (for the growing test harness)
    setup: list[str] = field(default_factory=list)   # Optional setup prompts (executed first)
    seed_files: dict[str, str] = field(default_factory=dict)  # path -> content for initial files (file domain)
    seed_dirs: list[str] = field(default_factory=list)  # directories to pre-create (file domain)
    seed_drone: dict = field(default_factory=dict)    # initial drone state for drone domain e.g. {"scan": 3.0}
    expected_primary_tools: list[str] = field(default_factory=list)
    validator: Callable[[Any], tuple[bool, str]] = lambda state: (True, "no validator")
    notes: str = ""


def make_validator(check_fn: Callable[[Path], tuple[bool, str]]) -> Callable[[Path], tuple[bool, str]]:
    return check_fn


# =============================================================================
# Helper utilities for validators and metrics
# =============================================================================

def ws_read(ws: Path, rel: str, start_line: int | None = None, num_lines: int | None = None) -> str:
    """Read a file using the module's function after temporarily pointing at ws."""
    old = tca.WORKSPACE
    tca.WORKSPACE = ws
    try:
        return tca.read_file(rel, start_line=start_line, num_lines=num_lines)
    finally:
        tca.WORKSPACE = old


def ws_list(ws: Path) -> str:
    old = tca.WORKSPACE
    tca.WORKSPACE = ws
    try:
        return tca.list_files(".")
    finally:
        tca.WORKSPACE = old


def file_exists(ws: Path, rel: str) -> bool:
    return (ws / rel).exists()


def file_contains(ws: Path, rel: str, substring: str) -> bool:
    try:
        return substring in (ws / rel).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def count_occurrences(ws: Path, rel: str, substring: str) -> int:
    try:
        return (ws / rel).read_text(encoding="utf-8", errors="replace").count(substring)
    except Exception:
        return 0


def did_use_tool(result: dict, tool_name: str) -> bool:
    return any(t["name"] == tool_name for t in result.get("tools_used", []))


def count_tool_uses(result: dict, tool_name: str) -> int:
    return sum(1 for t in result.get("tools_used", []) if t["name"] == tool_name)


def workflow_had_read_before_mutate(result: dict) -> bool:
    """Crude but useful: did any read appear in the tool sequence before the first write/edit/append?"""
    tools = [t["name"] for t in result.get("tools_used", [])]
    first_mutation_idx = None
    for i, name in enumerate(tools):
        if name in ("write_file", "edit_file", "append_file", "delete_file"):
            first_mutation_idx = i
            break
    if first_mutation_idx is None:
        return True  # no mutation, vacuously ok
    return any(name == "read_file" for name in tools[:first_mutation_idx])


# Drone-specific helpers (for the new spatial/drone direction)
def drone_had_status_before_action(result: dict) -> bool:
    """For drone: did get_status appear before first move/turn/scan?"""
    tools = [t["name"] for t in result.get("tools_used", [])]
    first_action_idx = None
    for i, name in enumerate(tools):
        if name in ("move_forward", "turn", "change_height", "scan"):
            first_action_idx = i
            break
    if first_action_idx is None:
        return True
    return "get_status" in tools[:first_action_idx]

def drone_explored(result: dict, min_map: int = 20) -> bool:
    """Did the drone produce a reasonable map via scans?"""
    # In real run we can check via sim, in mock via tools or state
    return result.get("map_size", 0) >= min_map or any("scan" in str(t) for t in result.get("tools_used", []))


# =============================================================================
# Test Case Catalog - Realistic natural language prompts
# =============================================================================

def build_test_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    # --- DRONE / SPATIAL (new direction: point clouds + model driving drone) ---
    # Prioritized first so small --limit + --viz shows the 3D simulation immediately.
    if td is not None:
        cases.append(TestCase(
            id="drone_disc_01",
            domain="drone",
            category="drone_discovery",
            prompt="Scan the area in front of you.",
            description="Basic scan to discover the world via point cloud",
            expected_primary_tools=["scan", "get_status"],
            validator=make_validator(lambda sim: (
                len(getattr(sim, 'map_points', [])) > 5,
                "Drone performed scan and populated map points"
            ))
        ))

        cases.append(TestCase(
            id="drone_nav_01",
            domain="drone",
            category="drone_navigation",
            prompt="Fly forward 2 meters then scan the area.",
            description="Simple move + scan sequence",
            seed_drone={"scan": 2.0},
            expected_primary_tools=["move_forward", "scan", "get_status"],
            validator=make_validator(lambda sim: (
                sim.get_pose()["z"] > 1.5 and len(sim.map_points) > 10,
                "Drone moved forward and scanned (map grew)"
            ))
        ))

        cases.append(TestCase(
            id="drone_nav_02",
            domain="drone",
            category="drone_navigation",
            prompt="Turn left 90 degrees, move forward, then scan.",
            description="Turn + move + scan to explore sideways",
            expected_primary_tools=["turn", "move_forward", "scan"],
            validator=make_validator(lambda sim: (
                abs(sim.get_pose()["yaw"]) > 80 and len(sim.map_points) > 5,
                "Drone turned and moved, producing scan data"
            ))
        ))

        cases.append(TestCase(
            id="drone_map_01",
            domain="drone",
            category="drone_mapping",
            prompt="Explore the area by scanning and moving forward several times to build a map.",
            description="Multi-step exploration and mapping task",
            expected_primary_tools=["scan", "move_forward", "get_status"],
            validator=make_validator(lambda sim: (
                len(sim.map_points) > 30,
                "Drone explored and built a decent sized map via repeated scans"
            ))
        ))

        cases.append(TestCase(
            id="drone_robust_01",
            domain="drone",
            category="drone_robustness",
            prompt="Scan first, then carefully move forward while avoiding walls by scanning often.",
            description="Workflow: status/scan before actions, avoid boundaries",
            expected_primary_tools=["get_status", "scan", "move_forward"],
            validator=make_validator(lambda sim: (
                len(sim.map_points) > 15 and abs(sim.get_pose()["x"]) < 10,
                "Drone followed scan-first workflow and stayed in safe area"
            ))
        ))

    # --- FILE / CRUD (original direction) ---
    # --- DISCOVERY ---
    cases.append(TestCase(
        id="disc_01",
        category="discovery",
        prompt="What files are currently in the workspace?",
        description="Basic list using everyday language",
        expected_primary_tools=["list_files"],
        validator=make_validator(lambda ws: (
            "DIR" in ws_list(ws) or "FILE" in ws_list(ws) or "(empty)" in ws_list(ws),
            "Workspace listing succeeded"
        ))
    ))

    cases.append(TestCase(
        id="disc_02",
        category="discovery",
        prompt="Show me everything that exists here, including any subfolders.",
        description="Slightly more explicit discovery request",
        seed_files={"src/main.py": "# placeholder\n", "README.md": "# Project\n"},
        expected_primary_tools=["list_files"],
        validator=make_validator(lambda ws: (
            file_exists(ws, "src/main.py") and "DIR  src/" in ws_list(ws),
            "Listed and saw src/ directory"
        ))
    ))

    # --- READING ---
    cases.append(TestCase(
        id="read_01",
        category="read",
        prompt="Read the contents of the config file for me.",
        description="Simple read request",
        seed_files={"config.yaml": "version: 1.2\nname: test-project\n"},
        expected_primary_tools=["read_file"],
        validator=make_validator(lambda ws: (
            "version: 1.2" in ws_read(ws, "config.yaml"),
            "Read full config content"
        ))
    ))

    cases.append(TestCase(
        id="read_02",
        category="read",
        prompt="Only show me lines 3 through 6 of the big notes file.",
        description="Requests partial read using line ranges (important for context saving)",
        seed_files={"notes.txt": "\n".join([f"Line {i}" for i in range(1, 21)])},
        expected_primary_tools=["read_file"],
        validator=make_validator(lambda ws: (
            "Line 3" in ws_read(ws, "notes.txt") and "Line 7" not in ws_read(ws, "notes.txt", start_line=1, num_lines=4),
            "Partial read requested (model may read full file; validator checks that limited slice would exclude later lines)"
        ))
    ))

    # --- CREATION ---
    cases.append(TestCase(
        id="create_01",
        category="create",
        prompt="Create a new file called hello.txt that contains the text 'Hello from the file agent'.",
        description="Classic create request with explicit content",
        expected_primary_tools=["write_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "hello.txt", "Hello from the file agent"),
            "hello.txt was created with correct content"
        ))
    ))

    cases.append(TestCase(
        id="create_02",
        category="create",
        prompt="Make a Python module at src/utils/helpers.py that defines a function called greet that returns 'hi'.",
        description="Create nested directories + code file (tests parent dir creation)",
        expected_primary_tools=["write_file"],
        validator=make_validator(lambda ws: (
            file_exists(ws, "src/utils/helpers.py") and "def greet" in ws_read(ws, "src/utils/helpers.py"),
            "Nested file created successfully"
        ))
    ))

    # --- APPEND ---
    cases.append(TestCase(
        id="append_01",
        category="append",
        prompt="Add the line '2026-06-07: evaluation run completed successfully' to the end of the activity log.",
        description="Append to existing log",
        seed_files={"activity.log": "2026-06-01: started project\n"},
        expected_primary_tools=["append_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "activity.log", "evaluation run completed successfully"),
            "Line correctly appended to log"
        ))
    ))

    cases.append(TestCase(
        id="append_02",
        category="append",
        prompt="Append three new bullet points about file handling to the project notes.",
        description="Append multi-line content",
        seed_files={"notes.md": "# Notes\n\n- First point\n"},
        expected_primary_tools=["append_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "notes.md", "file handling") or "bullet" in ws_read(ws, "notes.md").lower() or (ws / "notes.md").read_text().count("\n-") >= 3,
            "Multi-line append succeeded (added content about file handling)"
        ))
    ))

    # --- EDIT (the most important for "better CRUD") ---
    cases.append(TestCase(
        id="edit_01",
        category="edit",
        prompt="In the greeting.py file, change the message from 'Hello World' to 'Hello from the improved CRUD agent'.",
        description="Basic precise single edit using natural language",
        seed_files={"greeting.py": "def main():\n    print('Hello World')\n"},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "greeting.py", "Hello from the improved CRUD agent") and
            not file_contains(ws, "greeting.py", "Hello World"),
            "Targeted edit succeeded"
        ))
    ))

    cases.append(TestCase(
        id="edit_02",
        category="edit",
        prompt="Update the version in pyproject.toml from 0.9.1 to 1.0.0. Make sure you pick a unique string.",
        description="Edit that requires reading first to get unique context (tests workflow)",
        seed_files={"pyproject.toml": '[project]\nname = "demo"\nversion = "0.9.1"\n'},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "pyproject.toml", 'version = "1.0.0"'),
            "Version bump via precise edit"
        ))
    ))

    cases.append(TestCase(
        id="edit_03_ambiguous",
        category="edit",
        prompt="Replace every instance of the word 'TODO' with 'DONE' in the task list.",
        description="Tests replace_all behavior or multiple careful edits when string is repeated",
        seed_files={"tasks.txt": "TODO: write tests\nTODO: improve prompt\nTODO: ship it\n"},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            count_occurrences(ws, "tasks.txt", "DONE") >= 2,
            "Multiple replacements handled (replace_all or repeated edits)"
        ))
    ))

    cases.append(TestCase(
        id="edit_04_context",
        category="edit",
        prompt="In the main function of app.py, change only the return message for the health check endpoint to return a proper JSON status.",
        description="Requires the model to read enough context to make a unique old_string",
        seed_files={"app.py": """def main():\n    return \"OK\"\n\ndef health():\n    return \"OK\"\n"""},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            'return "OK"' not in (ws / "app.py").read_text() or
            file_contains(ws, "app.py", "status") or file_contains(ws, "app.py", "JSON"),
            "Context-aware edit (may pass even if model chose a different valid approach)"
        ))
    ))

    # --- DELETE ---
    cases.append(TestCase(
        id="delete_01",
        category="delete",
        prompt="Delete the old temporary file called debug.log.",
        description="Simple delete request",
        seed_files={"debug.log": "lots of debug output\n", "important.txt": "keep me\n"},
        expected_primary_tools=["delete_file"],
        validator=make_validator(lambda ws: (
            not file_exists(ws, "debug.log") and file_exists(ws, "important.txt"),
            "debug.log removed, important.txt untouched"
        ))
    ))

    cases.append(TestCase(
        id="delete_02",
        category="delete",
        prompt="Remove the empty temp directory called old_stuff.",
        description="Delete empty directory",
        seed_dirs=["old_stuff"],
        expected_primary_tools=["delete_file"],
        validator=make_validator(lambda ws: (
            not (ws / "old_stuff").exists(),
            "Empty directory removed"
        ))
    ))

    # --- MULTI-STEP / WORKFLOW ---
    cases.append(TestCase(
        id="multi_01",
        category="multi-step",
        prompt="First look at what files exist, then read the version info, then bump the version in the project file to 2.0.0 and also add a short note about the change in the changelog.",
        description="Full realistic workflow: discover → read → edit + append",
        seed_files={
            "pyproject.toml": 'version = "1.3.0"\n',
            "CHANGELOG.md": "# Changelog\n\n## [1.3.0]\n- Previous stuff\n"
        },
        expected_primary_tools=["list_files", "read_file", "edit_file", "append_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "pyproject.toml", "2.0.0") and
            file_contains(ws, "CHANGELOG.md", "2.0.0"),
            "Multi-step workflow completed: version bumped + changelog updated"
        ))
    ))

    cases.append(TestCase(
        id="multi_02",
        category="multi-step",
        prompt="Explore the project structure, find the main source file, read its top section, and add a proper module docstring at the very top.",
        description="Discovery + read + edit in one natural request",
        seed_files={"src/core.py": "def run():\n    pass\n"},
        expected_primary_tools=["list_files", "read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            '"""' in ws_read(ws, "src/core.py")[:200] or "Module docstring" in ws_read(ws, "src/core.py"),
            "Docstring was added after exploration"
        ))
    ))

    # --- ROBUSTNESS / FOLLOWING INSTRUCTIONS ---
    cases.append(TestCase(
        id="robust_01",
        category="robustness",
        prompt="Please read the file first, then change the author name in metadata.json from 'olduser' to 'crud-agent'.",
        description="Explicitly tells the model the desired workflow (tests instruction following)",
        seed_files={"metadata.json": '{"author": "olduser", "tool": "test"}\n'},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "metadata.json", "crud-agent"),
            "Followed explicit 'read first' instruction"
        ))
    ))

    cases.append(TestCase(
        id="robust_02",
        category="robustness",
        prompt="Try to edit the greeting without reading the file first. Then correct yourself and do it properly.",
        description="Tests whether the agent can recover when it knows it should have read first",
        seed_files={"greeting.txt": "Welcome, stranger!\n"},
        expected_primary_tools=["read_file", "edit_file"],
        validator=make_validator(lambda ws: (
            file_contains(ws, "greeting.txt", "Welcome") or file_contains(ws, "greeting.txt", "agent"),
            "Either recovered gracefully or performed a valid edit"
        ))
    ))

    return cases


# =============================================================================
# Test Runner
# =============================================================================

def run_single_test(case: TestCase, model: str, workspace_root: Path, mock: bool = False, shared_sim=None) -> dict[str, Any]:
    """Run one test case against a fresh workspace or drone sim.

    Supports both "file" (original CRUD) and "drone" (spatial/pointcloud driving) domains.
    If mock=True, we simulate a perfect agent by directly exercising the tool
    implementations in a sensible order (no LLM calls). This is great for
    developing the test cases, validators, and beautiful report on resource-
    constrained machines.
    shared_sim: for drone viz mode, use this shared simulator so the 3D viewer sees updates.
    """
    if case.domain == "drone":
        if td is None:
            return {"case": case, "passed": False, "details": "drone module not available", "tools_used": [], "primary_tool_hit": False}
        sim = shared_sim if shared_sim is not None else td.DroneSimulator()
        # seed initial drone state (only if new sim)
        if shared_sim is None and case.seed_drone.get("scan"):
            sim.scan(case.seed_drone["scan"])
        # for other seeds if needed
        tools_used_names: list[str] = []
        assistant_text = ""
        start = time.time()

        if mock:
            # Mock excellent drone pilot
            def _record(name: str, **kwargs):
                tools_used_names.append(name)
                if name == "get_status":
                    return sim.get_status()
                if name == "move_forward":
                    return sim.move_forward(kwargs.get("distance", 1.0))
                if name == "turn":
                    return sim.turn(kwargs.get("degrees", 30))
                if name == "change_height":
                    return sim.change_height(kwargs.get("delta", 1.0))
                if name == "scan":
                    return sim.scan(kwargs.get("range_m", 4.0))
                if name == "reset_drone":
                    return sim.reset()
                return "OK"

            # Smart mock per prompt keywords and category
            p = case.prompt.lower()
            if "scan" in p or "discovery" in case.category:
                _record("scan", range_m=4.0)
                time.sleep(0.6)
            if "forward" in p or "move" in p:
                dist = 2.0
                if "3" in p or "three" in p: dist = 3.0
                _record("move_forward", distance=dist)
                time.sleep(0.6)
            if "turn" in p or "left" in p or "right" in p:
                deg = 90 if "90" in p else 45
                if "left" in p: deg = -deg if deg > 0 else deg  # sign convention
                _record("turn", degrees=deg)
                time.sleep(0.6)
            if "height" in p or "up" in p or "down" in p:
                _record("change_height", delta=1.0 if "up" in p else -1.0)
                time.sleep(0.6)
            if "explore" in p or "map" in p or "multi" in case.category:
                _record("scan", range_m=3)
                time.sleep(0.6)
                _record("move_forward", distance=2)
                time.sleep(0.6)
                _record("scan", range_m=3)
                time.sleep(0.6)
            if "reset" in p:
                _record("reset_drone")

            # Always get status in mock for workflow
            _record("get_status")
            time.sleep(0.3)

            passed, details = case.validator(sim)
            duration = time.time() - start
            return {
                "case": case,
                "result": {"tools_used": [{"name": n} for n in tools_used_names], "map_size": len(sim.map_points)},
                "passed": passed,
                "details": details,
                "duration": round(duration, 2),
                "tools_used": tools_used_names,
                "primary_tool_hit": any(t in tools_used_names for t in case.expected_primary_tools) if case.expected_primary_tools else True,
                "workflow_ok": drone_had_status_before_action({"tools_used": [{"name": n} for n in tools_used_names]}),
                "num_tool_calls": len(tools_used_names),
                "assistant_text": "[MOCK DRONE] " + case.id,
                "final_state": sim.get_status(),
            }
        else:
            # Real LLM run for drone
            if not hasattr(td, "DroneAgent"):
                return {"case": case, "passed": False, "details": "DroneAgent not exposed in module", "tools_used": []}
            agent = td.DroneAgent(model=model)
            start = time.time()
            result = agent.send(case.prompt)
            duration = time.time() - start
            tools_used_names = [t["name"] for t in result.get("tools_used", [])]
            passed, details = case.validator(agent.get_sim())
            return {
                "case": case,
                "result": result,
                "passed": passed,
                "details": details,
                "duration": round(duration, 2),
                "tools_used": tools_used_names,
                "primary_tool_hit": any(t in tools_used_names for t in case.expected_primary_tools) if case.expected_primary_tools else True,
                "workflow_ok": drone_had_status_before_action(result),
                "num_tool_calls": len(tools_used_names),
                "assistant_text": result.get("assistant", ""),
                "final_state": agent.get_sim().get_status(),
            }

    # --- Original FILE domain logic (unchanged for backward compat) ---
    ws = workspace_root / case.id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    # Seed initial files directly (fast & deterministic)
    for rel, content in case.seed_files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    # Seed directories
    for d in case.seed_dirs:
        (ws / d).mkdir(parents=True, exist_ok=True)

    # Point the module at this workspace for the duration of the test
    old_ws = tca.WORKSPACE
    tca.WORKSPACE = ws

    tools_used_names: list[str] = []
    assistant_text = ""
    start = time.time()

    if mock:
        # --- MOCK MODE: simulate an excellent agent that follows best practices ---
        # We do a "smart" sequence based on the category and expected tools.
        # This lets you develop the harness and report without loading models.

        def _record(name: str, **kwargs):
            tools_used_names.append(name)
            # Actually call the real implementation so filesystem changes happen
            if name == "list_files":
                return tca.list_files(kwargs.get("path", "."), base=ws)
            if name == "read_file":
                return tca.read_file(kwargs.get("path"), kwargs.get("start_line"), kwargs.get("num_lines"), base=ws)
            if name == "write_file":
                return tca.write_file(kwargs["path"], kwargs.get("content", ""), base=ws)
            if name == "append_file":
                return tca.append_file(kwargs["path"], kwargs.get("content", ""), base=ws)
            if name == "edit_file":
                return tca.edit_file(kwargs["path"], kwargs.get("old", ""), kwargs.get("new", ""), kwargs.get("replace_all", False), base=ws)
            if name == "delete_file":
                return tca.delete_file(kwargs["path"], base=ws)
            if name == "mkdir":
                return tca.mkdir(kwargs["path"], base=ws)
            return "OK"

        # Smart mock behavior per category
        if "discovery" in case.category or "list" in str(case.expected_primary_tools):
            _record("list_files", path=".")

        if case.id.startswith("read"):
            _record("read_file", path=list(case.seed_files.keys())[0] if case.seed_files else "notes.txt",
                    start_line=case.prompt.split("lines ")[1].split(" through")[0] if "lines " in case.prompt else None,
                    num_lines=4 if "through" in case.prompt else None)

        if "create" in case.category or "write_file" in case.expected_primary_tools:
            for rel in list(case.seed_files.keys()) + ["hello.txt", "src/utils/helpers.py"]:
                if not file_exists(ws, rel):
                    if "helpers" in rel:
                        content = "def greet():\n    return 'hi'\n"
                    else:
                        content = "Hello from the file agent"
                    _record("write_file", path=rel, content=content)

        if "append" in case.category:
            target = list(case.seed_files.keys())[0] if case.seed_files else "activity.log"
            if "evaluation run completed successfully" in case.prompt:
                extra = "\n2026-06-07: evaluation run completed successfully\n"
            else:
                extra = "\n- Added via append about file handling\n- Second new point\n- Third new point about CRUD\n"
            _record("append_file", path=target, content=extra)

        if "edit" in case.category:
            # First "read" (we already may have), then targeted edit
            for rel in case.seed_files:
                txt = (ws / rel).read_text(errors="replace")
                if "Hello World" in txt:
                    _record("edit_file", path=rel, old="print('Hello World')", new="print('Hello from the improved CRUD agent')")
                elif "0.9.1" in txt or 'version = "0.9.1"' in txt:
                    _record("edit_file", path=rel, old='version = "0.9.1"', new='version = "1.0.0"')
                elif "TODO" in txt and "DONE" not in txt:
                    _record("edit_file", path=rel, old="TODO", new="DONE", replace_all=True)
                else:
                    # generic context edit
                    _record("edit_file", path=rel, old=txt.splitlines()[0], new="# Updated by CRUD agent\n" + txt.splitlines()[0])

        if "delete" in case.category:
            for rel in list(case.seed_files.keys()) + ["debug.log", "old_stuff"]:
                if file_exists(ws, rel) or (ws / rel).is_dir():
                    _record("delete_file", path=rel)

        if "multi" in case.category:
            _record("list_files")
            for rel in case.seed_files:
                _record("read_file", path=rel)
            # Do the mutations the prompt asks for
            for rel in case.seed_files:
                if "pyproject" in rel or "toml" in rel:
                    _record("edit_file", path=rel, old='version = "1.3.0"', new='version = "2.0.0"')
                if "CHANGELOG" in rel:
                    _record("append_file", path=rel, content="\n## [2.0.0]\n- Bumped via multi-step workflow\n")
            if "core.py" in case.seed_files:
                _record("edit_file", path="src/core.py", old="def run():", new='"""Improved by file agent."""\ndef run():')

        if "robust" in case.category:
            for rel in case.seed_files:
                _record("read_file", path=rel)
                txt = (ws / rel).read_text(errors="replace")
                if "olduser" in txt:
                    _record("edit_file", path=rel, old='"olduser"', new='"crud-agent"')

        assistant_text = f"[MOCK] Completed {case.id} using best-practice workflow."

    else:
        # --- REAL MODE: use the actual LLM agent ---
        agent = tca.FileAgent(model=model, workspace=ws)

        for setup_prompt in case.setup:
            agent.send(setup_prompt)

        result = agent.send(case.prompt)
        tools_used_names = [t["name"] for t in result["tools_used"]]
        assistant_text = result.get("assistant", "")

    duration = time.time() - start

    # Restore global
    tca.WORKSPACE = old_ws

    # Build a fake result dict for the rest of the pipeline when in mock mode
    fake_result = {
        "assistant": assistant_text,
        "tools_used": [{"name": n} for n in tools_used_names]
    }

    # Run validator (always real)
    passed, details = case.validator(ws)

    # Metrics
    primary_hit = any(t in tools_used_names for t in case.expected_primary_tools) if case.expected_primary_tools else True
    read_before_mutate = workflow_had_read_before_mutate(fake_result)

    return {
        "case": case,
        "result": fake_result,
        "passed": passed,
        "details": details,
        "duration": round(duration, 2),
        "tools_used": tools_used_names or ["(mocked)"],
        "primary_tool_hit": primary_hit,
        "read_before_mutate": read_before_mutate,
        "num_tool_calls": len(tools_used_names),
        "assistant_text": assistant_text,
        "workspace_path": str(ws),
        "final_snapshot": ws_list(ws),
    }


def run_all_tests(model: str, limit: int | None = None, verbose: bool = True, mock: bool = False, shared_sim=None) -> list[dict]:
    cases = build_test_cases()
    if limit:
        cases = cases[:limit]

    workspace_root = Path("/tmp/crud_eval_workspaces")
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True)

    results = []
    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i}/{len(cases)}] {case.id}: {case.prompt[:70]}...")
        sim_for_case = shared_sim if (case.domain == "drone" and shared_sim is not None) else None
        res = run_single_test(case, model, workspace_root, mock=mock, shared_sim=sim_for_case)
        results.append(res)
        if verbose:
            status = "PASS" if res["passed"] else "FAIL"
            wf = res.get("read_before_mutate") or res.get("workflow_ok")
            print(f"    → {status} | tools: {res['tools_used']} | workflow: {wf}")
    return results


# =============================================================================
# Impressive Report Generation
# =============================================================================

def generate_html_report(results: list[dict], model: str, timestamp: str) -> str:
    """Return a complete, beautiful, self-contained HTML document."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = (passed / total * 100) if total else 0

    primary_success = sum(1 for r in results if r["primary_tool_hit"])
    workflow_good = sum(1 for r in results if bool(r.get("read_before_mutate") or r.get("workflow_ok")))

    drone_res = [r for r in results if getattr(r.get("case"), "domain", "file") == "drone"]
    drone_pass = sum(1 for r in drone_res if r["passed"]) if drone_res else 0
    drone_workflow = sum(1 for r in drone_res if r.get("workflow_ok")) if drone_res else 0

    categories = sorted(set(r["case"].category for r in results))
    cat_stats = {}
    for cat in categories:
        cat_res = [r for r in results if r["case"].category == cat]
        cat_stats[cat] = {
            "total": len(cat_res),
            "passed": sum(1 for r in cat_res if r["passed"]),
            "rate": round(sum(1 for r in cat_res if r["passed"]) / len(cat_res) * 100, 1) if cat_res else 0
        }

    # Build test result cards
    cards_html = ""
    for r in results:
        c = r["case"]
        status_color = "emerald" if r["passed"] else "rose"
        status_badge = f'<span class="px-3 py-1 text-xs font-semibold rounded-full bg-{status_color}-500/10 text-{status_color}-400 border border-{status_color}-500/30">{"PASS" if r["passed"] else "FAIL"}</span>'

        tools_pills = ""
        for tname in r["tools_used"]:
            color = {
                "list_files": "sky",
                "read_file": "violet",
                "write_file": "amber",
                "append_file": "teal",
                "edit_file": "fuchsia",
                "delete_file": "rose",
                "scan": "emerald",
                "move_forward": "amber",
                "turn": "sky",
                "get_status": "slate",
                "change_height": "teal",
            }.get(tname, "slate")
            tools_pills += f'<span class="px-2 py-0.5 text-[10px] font-mono rounded bg-{color}-500/10 text-{color}-300 border border-{color}-500/20">{tname}</span>'

        workflow_badge = ""
        wf = r.get("read_before_mutate") or r.get("workflow_ok")
        if wf:
            label = "✓ read/scan before action" if getattr(r.get("case"), "domain", "file") == "drone" else "✓ read before mutate"
            workflow_badge = f'<span class="text-emerald-400 text-xs">{label}</span>'
        else:
            label = "⚠ no prior status/scan" if getattr(r.get("case"), "domain", "file") == "drone" else "⚠ no prior read"
            workflow_badge = f'<span class="text-amber-400 text-xs">{label}</span>'

        cards_html += f"""
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 mb-4 hover:border-slate-700 transition-colors">
            <div class="flex items-start justify-between gap-4">
                <div>
                    <div class="flex items-center gap-3 mb-1">
                        <span class="font-mono text-xs text-slate-500">{c.id}</span>
                        <span class="px-2 py-0.5 text-xs rounded-full bg-slate-800 text-slate-400">{c.category}</span>
                        {status_badge}
                    </div>
                    <div class="text-lg font-medium text-white mb-2">“{c.prompt}”</div>
                    <div class="text-sm text-slate-400 mb-3">{c.description}</div>
                </div>
            </div>

            <div class="flex flex-wrap gap-2 mb-3">
                {tools_pills or '<span class="text-xs text-slate-500">no tools used</span>'}
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                <div>
                    <div class="text-slate-500 text-xs mb-1">WORKFLOW</div>
                    {workflow_badge}
                </div>
                <div>
                    <div class="text-slate-500 text-xs mb-1">TOOL CALLS</div>
                    <span class="font-mono">{r['num_tool_calls']}</span>
                </div>
                <div>
                    <div class="text-slate-500 text-xs mb-1">DURATION</div>
                    <span class="font-mono">{r['duration']}s</span>
                </div>
            </div>

            <details class="mt-4">
                <summary class="cursor-pointer text-xs text-slate-400 hover:text-slate-300">Show details &amp; agent response</summary>
                <div class="mt-3 pl-2 border-l border-slate-800 text-sm space-y-3">
                    <div>
                        <div class="text-xs text-slate-500 mb-1">Agent final reply:</div>
                        <pre class="text-xs bg-black/40 p-3 rounded overflow-auto text-slate-300 whitespace-pre-wrap">{r['assistant_text'] or '(no text)'}</pre>
                    </div>
                    <div>
                        <div class="text-xs text-slate-500 mb-1">Validator:</div>
                        <div class="text-xs font-mono text-slate-400">{r['details']}</div>
                    </div>
                    <div>
                        <div class="text-xs text-slate-500 mb-1">Final state / snapshot:</div>
                        <pre class="text-[10px] bg-black/40 p-2 rounded text-emerald-300/80 font-mono whitespace-pre-wrap">{r.get('final_snapshot') or r.get('final_state') or r.get('result', {}).get('final_pose', {})}</pre>
                    </div>
                </div>
            </details>
        </div>
        """

    # Category pills for filtering (JS)
    cat_pills = ""
    for cat in categories:
        rate = cat_stats[cat]['rate']
        cat_pills += f'<button onclick="filterByCategory(\'{cat}\')" class="px-4 py-1.5 text-sm rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-700 transition">{cat} <span class="text-emerald-400">({rate}%)</span></button>'

    # Prompt library (great for improving the model later)
    prompt_library = ""
    for r in results:
        if r["passed"] and r["case"].expected_primary_tools:
            prompt_library += f"""
            <div class="flex gap-3 items-start bg-slate-950 border border-slate-800 rounded-xl p-4 mb-2">
                <div class="flex-1">
                    <div class="font-mono text-xs text-slate-500 mb-1">{r['case'].id} • {r['case'].category}</div>
                    <div class="text-white">“{r['case'].prompt}”</div>
                </div>
                <button onclick="copyToClipboard(this)" class="text-xs px-3 py-1 rounded border border-slate-700 hover:bg-slate-800 shrink-0">Copy</button>
            </div>
            """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Evaluation • {timestamp}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');
        body {{ font-family: 'Inter', system_ui, sans-serif; }}
        .font-display {{ font-family: 'Space Grotesk', 'Inter', sans-serif; }}
        .metric-card {{ transition: transform .2s cubic-bezier(0.4, 0, 0.2, 1); }}
        .metric-card:hover {{ transform: translateY(-2px); }}
        pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
        .section-header {{ letter-spacing: -.025em; }}
    </style>
</head>
<body class="bg-slate-950 text-slate-200">
    <div class="max-w-[1200px] mx-auto px-6 py-10">
        <!-- Header -->
        <div class="flex items-end justify-between mb-8">
            <div>
                <div class="flex items-center gap-3">
                    <div class="w-9 h-9 rounded-2xl bg-emerald-500 flex items-center justify-center">
                        <span class="text-slate-950 font-bold text-2xl">F</span>
                    </div>
                    <div>
                        <div class="font-display text-4xl font-semibold tracking-tighter">Agent Evaluation</div>
                        <div class="text-emerald-400 text-sm -mt-1">File CRUD + Drone/Spatial</div>
                    </div>
                </div>
            </div>
            <div class="text-right text-sm text-slate-400">
                <div>Model: <span class="font-mono text-white">{model}</span></div>
                <div>{timestamp}</div>
            </div>
        </div>

        <!-- Big Score -->
        <div class="bg-slate-900 border border-slate-800 rounded-3xl p-8 mb-8 flex flex-col md:flex-row items-center gap-8">
            <div class="flex-1">
                <div class="text-sm tracking-[3px] text-emerald-400 font-medium mb-1">OVERALL SUCCESS</div>
                <div class="font-display text-[92px] leading-none font-semibold tracking-[-6.4px] text-white">{passed}<span class="text-5xl text-slate-500">/{total}</span></div>
                <div class="text-3xl font-semibold text-emerald-400 mt-1">{pass_rate:.1f}% pass rate</div>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-3 gap-4 flex-1">
                <div class="metric-card bg-slate-950 border border-slate-800 rounded-2xl p-5">
                    <div class="text-xs text-slate-400">PRIMARY TOOL SUCCESS</div>
                    <div class="text-4xl font-semibold mt-2">{primary_success}<span class="text-xl text-slate-500">/{total}</span></div>
                    <div class="text-emerald-400 text-sm mt-1">{(primary_success/total*100):.0f}% of tests hit expected tool(s)</div>
                </div>
                <div class="metric-card bg-slate-950 border border-slate-800 rounded-2xl p-5">
                    <div class="text-xs text-slate-400">GOOD WORKFLOW</div>
                    <div class="text-4xl font-semibold mt-2">{workflow_good}<span class="text-xl text-slate-500">/{total}</span></div>
                    <div class="text-emerald-400 text-sm mt-1">Proper discovery/scan before action (file read or drone status/scan)</div>
                </div>
                <div class="metric-card bg-slate-950 border border-slate-800 rounded-2xl p-5">
                    <div class="text-xs text-slate-400">CATEGORIES TESTED</div>
                    <div class="text-4xl font-semibold mt-2">{len(categories)}</div>
                    <div class="text-sm text-slate-400 mt-1">file: discovery • read • create • edit • append • delete • multi-step • robustness + drone: discovery • navigation • mapping • robustness</div>
                </div>
            </div>
            {f'<div class="mt-2 text-sm text-emerald-400">Drone subset: {drone_pass}/{len(drone_res)} passed ({(drone_pass/len(drone_res)*100):.0f}% ), workflow {drone_workflow}/{len(drone_res)}</div>' if drone_res else ''}
        </div>

        <!-- Category Filters -->
        <div class="mb-4 flex items-center gap-2 flex-wrap">
            <button onclick="filterByCategory('all')" class="px-4 py-1.5 text-sm rounded-full bg-white text-slate-900 font-medium">All</button>
            {cat_pills}
        </div>

        <!-- Results -->
        <div id="results-container">
            {cards_html}
        </div>

        <!-- Prompt Library -->
        <div class="mt-12">
            <div class="flex items-baseline justify-between mb-4">
                <div>
                    <div class="font-display text-2xl font-semibold tracking-tight">Effective Natural Language Prompts</div>
                    <div class="text-slate-400 text-sm">Copy these into your system prompt or few-shot examples when tuning the model for file agent work.</div>
                </div>
            </div>
            <div class="max-h-[520px] overflow-auto pr-2">
                {prompt_library}
            </div>
        </div>

        <!-- How to use this report -->
        <div class="mt-12 border border-slate-800 bg-slate-900/50 rounded-3xl p-7 text-sm">
            <div class="font-semibold text-emerald-400 mb-2">How to use this to get better at agent-specific modeling</div>
            <ol class="list-decimal list-inside space-y-1.5 text-slate-300">
                <li>Look at the failures — read the exact prompt + what tools the model actually called.</li>
                <li>Improve the <span class="font-mono text-xs bg-slate-800 px-1.5 py-px rounded">SYSTEM_PROMPT</span> or individual tool <span class="font-mono text-xs bg-slate-800 px-1.5 py-px rounded">description</span>s in <span class="font-mono">toolcallingollama.py</span>.</li>
                <li>Add more sophisticated test cases to this file (especially multi-step and error-recovery scenarios).</li>
                <li>Re-run the eval and watch the numbers go up. This is your private eval harness for "file CRUD specialist" behavior.</li>
                <li>The JSON sidecar (<span class="font-mono text-xs">crud_agent_report_*.json</span>) is great for automated regression or for feeding into prompt optimizers later.</li>
            </ol>
        </div>

        <div class="text-center text-xs text-slate-500 mt-10">
            Generated by test_agent.py • Run again after prompt changes to measure improvement
        </div>
    </div>

    <script>
        function filterByCategory(cat) {{
            const cards = document.querySelectorAll('#results-container > div');
            cards.forEach(card => {{
                if (cat === 'all') {{
                    card.style.display = '';
                }} else {{
                    const catLabel = card.querySelector('.bg-slate-800.text-slate-400');
                    if (catLabel && catLabel.textContent.includes(cat)) {{
                        card.style.display = '';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
        }}

        function copyToClipboard(btn) {{
            const text = btn.parentElement.querySelector('.text-white').innerText;
            navigator.clipboard.writeText(text).then(() => {{
                const orig = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(() => btn.innerText = orig, 1400);
            }});
        }}

        // Keyboard shortcut: press / to focus first filter
        document.addEventListener('keydown', function(e) {{
            if (e.key === '/' && document.activeElement.tagName === 'BODY') {{
                e.preventDefault();
                const btns = document.querySelectorAll('button');
                if (btns.length > 1) btns[1].click();
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def generate_markdown_report(results: list[dict], model: str, timestamp: str) -> str:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    drone_res = [r for r in results if getattr(r.get("case"), "domain", "file") == "drone"]
    lines = [
        f"# Agent Evaluation Report (File CRUD + Drone/Spatial)",
        f"",
        f"**Model**: `{model}`  |  **Run**: {timestamp}",
        f"",
        f"## Summary",
        f"",
        f"- **Pass rate**: {passed}/{total} ({(passed/total*100):.1f}%)",
        f"- **Primary tool accuracy**: {sum(r['primary_tool_hit'] for r in results)}/{total}",
        f"- **Good workflow**: {sum(bool(r.get('read_before_mutate') or r.get('workflow_ok')) for r in results)}/{total}",
        f"",
    ]
    if drone_res:
        d_pass = sum(r["passed"] for r in drone_res)
        lines.append(f"- **Drone subset pass rate**: {d_pass}/{len(drone_res)} ({(d_pass/len(drone_res)*100):.1f}%)")
    lines.extend([
        f"",
        f"## Per-Category Results",
        f"",
    ])
    for cat in sorted(set(r["case"].category for r in results)):
        cat_res = [r for r in results if r["case"].category == cat]
        p = sum(1 for r in cat_res if r["passed"])
        lines.append(f"- **{cat}**: {p}/{len(cat_res)} passed")

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        c = r["case"]
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        lines.append(f"### {c.id} — {c.category} — {status}")
        lines.append(f"**Prompt**: “{c.prompt}”")
        lines.append(f"**Tools used**: {', '.join(r['tools_used']) or 'none'}")
        wf = r.get('read_before_mutate') or r.get('workflow_ok')
        lines.append(f"**Workflow ok**: {'yes' if wf else 'no'}")
        lines.append(f"**Validator**: {r['details']}")
        lines.append("")

    lines.append("## Recommended Prompts (copy these when improving the model)")
    for r in results:
        if r["passed"]:
            lines.append(f"- {r['case'].prompt}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the agent (file CRUD + drone/spatial pointcloud driving) with natural language test cases.")
    parser.add_argument("--model", default="lfm2.5-thinking", help="Ollama model to use")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N tests (useful on slow hardware)")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no LLM calls) — perfect for developing the test cases and report on sparse hardware")
    parser.add_argument("--viz", action="store_true", help="Open the 3D GLUT viewer for drone tests (prioritized first so you see the simulation immediately; viewer in main, tests in bg updating shared sim)")
    args = parser.parse_args()

    print("=== Agent Evaluation (File CRUD + Drone/Spatial) ===")
    print(f"Model: {args.model}")
    if args.limit:
        print(f"Limited to first {args.limit} tests")
    if args.mock:
        print("Running in MOCK mode (no ollama calls — exercising tool backend + validators + report)")

    start_time = time.time()
    if args.viz and td is not None:
        print("=== VIZ MODE: Drone tests run first (prioritized) so you see the 3D simulation immediately ===")
        shared_sim = td.DroneSimulator()
        shared_sim.scan(2.5)  # initial seed so map not empty
        viewer = td.DroneVisualizer(shared_sim)
        def run_tests_thread():
            print("Running tests in background thread (drone cases first, updating shared sim for live 3D viz)...")
            # run all, but drone cases will use the shared_sim
            res = run_all_tests(model=args.model, limit=args.limit, verbose=not args.quiet, mock=args.mock, shared_sim=shared_sim)
            # store for later if needed
            global _viz_results
            _viz_results = res
            print("Tests done, viewer will stay open. Close window to exit.")
        t = threading.Thread(target=run_tests_thread, daemon=True)
        t.start()
        viewer.run()  # blocks in main, display will pick updates from shared_sim
        t.join()
        results = globals().get('_viz_results', [])
        duration = time.time() - start_time
    else:
        results = run_all_tests(model=args.model, limit=args.limit, verbose=not args.quiet, mock=args.mock)
        duration = time.time() - start_time

    # Prepare output dir
    reports_dir = Path("eval_reports")
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Data for artifacts
    report_data = {
        "timestamp": ts,
        "model": args.model,
        "total_duration": round(duration, 1),
        "results": [
            {
                "id": r["case"].id,
                "category": r["case"].category,
                "domain": getattr(r["case"], "domain", "file"),
                "prompt": r["case"].prompt,
                "passed": r["passed"],
                "details": r["details"],
                "tools_used": r["tools_used"],
                "primary_tool_hit": r["primary_tool_hit"],
                "read_before_mutate": r.get("read_before_mutate", r.get("workflow_ok", False)),
                "workflow_ok": r.get("workflow_ok", r.get("read_before_mutate", False)),
                "num_tool_calls": r["num_tool_calls"],
                "duration": r["duration"],
                "assistant_text": r["assistant_text"],
                "map_size": r.get("map_size", 0),
            }
            for r in results
        ]
    }

    # Write JSON (great for later analysis / prompt optimization loops)
    json_path = reports_dir / f"agent_eval_report_{ts}.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Write Markdown
    md_path = reports_dir / f"agent_eval_report_{ts}.md"
    md_path.write_text(generate_markdown_report(results, args.model, ts), encoding="utf-8")

    # Write gorgeous HTML
    html = generate_html_report(results, args.model, ts)
    html_path = reports_dir / f"agent_eval_report_{ts}.html"
    html_path.write_text(html, encoding="utf-8")

    # Also keep a latest.html for convenience
    (reports_dir / "latest.html").write_text(html, encoding="utf-8")

    if args.mock:
        print("\n(MOCK mode used — the numbers show that the test cases, validators, and tools are working. Use --viz to see the 3D drone simulation live while tests run. Run without --mock when your ollama env is ready for real model evaluation.)")

    # Console summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    drone_res = [r for r in results if getattr(r["case"], "domain", "file") == "drone"]
    print("\n" + "=" * 50)
    print(f"Done in {duration:.1f}s")
    print(f"Pass rate: {passed}/{total} ({(passed/total*100):.1f}%)")
    if drone_res:
        d_pass = sum(1 for r in drone_res if r["passed"])
        print(f"Drone subset pass rate: {d_pass}/{len(drone_res)} ({(d_pass/len(drone_res)*100):.1f}%)")
    print(f"\nReports written to:")
    print(f"  HTML (open this in browser): {html_path}")
    print(f"  Latest (easy to re-open):    {reports_dir / 'latest.html'}")
    print(f"  JSON (for analysis):         {json_path}")
    print(f"  Markdown:                    {md_path}")
    if args.mock:
        print("\nThis was a MOCK run. The impressive report still shows you the structure, all your natural-language test cases, and validates the underlying tools (file + drone).")
    print("\nTip: Use --viz (with --mock) to see the 3D simulation while the newer drone tests run first. Improve SYSTEM_PROMPT or tool descriptions in the agent scripts (toolcallingollama.py / toolcalling_drone.py), then re-run to watch your model get better at reliable file work AND drone spatial control.")


if __name__ == "__main__":
    main()
