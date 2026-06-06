# LLM Agent Eval

Lightweight dual-domain tool-calling agent harness for small local Ollama models, designed for resource-sparse hardware (Jetson Orin, etc).

Two practical domains in one growing test harness:

- **File CRUD** — reliable list/read (with ranges)/write/append/edit (safe search-replace)/delete/mkdir with strong "read before mutate" workflow discipline and sandboxing.
- **Drone / Spatial** — drive a simulated drone through a 3D point-cloud world using `get_status`, `scan`, `move_forward`, `turn`, `change_height`. Watch the live map (green points) build in an interactive GLUT viewer while the agent works.

The emphasis is on **natural, common-language prompts** that real users would type, measurable workflow adherence, beautiful self-contained reports, and a tight feedback loop for gradual specialization of tiny models.

## Highlights

- One "growing test file" (`test_agent.py`) that serves as both specification and evaluation suite.
- `--mock` mode: perfect for fast iteration on the harness, validators, and reports with zero model cost or Ollama load.
- `--viz` mode: shared `DroneSimulator` instance + background test runner + foreground GLUT main loop. You see the 3D simulation (gray env + green scanned map + red drone + orange path) update live as the (mock or real) tests execute the prioritized drone cases.
- Reusable `FileAgent` and `DroneAgent` classes with full conversation history + explicit `role="tool"` result messages so the model can observe effects and chain actions.
- Impressive single-file Tailwind dark HTML reports (with per-test cards, colored tool pills, workflow badges, JS category filters, "Effective Natural Language Prompts" library section, and KPIs for pass rate / primary-tool accuracy / workflow rate).
- Also emits .json (for later analysis or prompt-optimization loops) and .md.
- Deliberately lightweight: coarse point clouds, history trimming, status injection, Orin-friendly immediate-mode OpenGL.

## Quick Start

```bash
# Use the venv where you already have ollama + PyOpenGL + freeglut working
python test_agent.py --mock                 # Fast, no LLM. Validates everything + produces nice report
python test_agent.py --mock --viz           # See the live 3D point-cloud drone simulation (recommended first time)
python test_agent.py --limit 6              # Real model on first N cases (drone cases are first)
python test_agent.py --model gemma3:1b
```

Open the generated `eval_reports/latest.html` (or the timestamped one) in any browser.

`python test_agent.py --help` for all flags.

## Running the Agents Directly (interactive)

```bash
python toolcallingollama.py     # File CRUD chat (supports /ls /read /clear)
python toolcalling_drone.py     # Drone + opens the GLUT 3D viewer (type natural missions in the terminal)
```

## Project Goals

This is infrastructure for **gradual, public, measurable improvement** of a small model's reliability at specific agent skills (disciplined file work today, spatial reasoning + safe tool sequencing for drone/robot control tomorrow).

- Add harder multi-step or recovery test cases to `test_agent.py`.
- Improve `SYSTEM_PROMPT` or individual tool `description`s.
- Re-run (with or without `--mock`) and watch the numbers and "Effective Prompts" section improve.
- Use the JSON reports for automated regression or future training data.

The same agent + harness pattern is intended to transfer later from simulation to real hardware.

## Repository Layout

- `test_agent.py` — the dual-domain test harness, mock logic, shared-sim viz glue, and full report generators (HTML/MD/JSON)
- `toolcallingollama.py` — `FileAgent`, 7 CRUD tools (list/read/write/append/edit/delete/mkdir), sandboxing, and the pedagogical SYSTEM_PROMPT
- `toolcalling_drone.py` — `DroneSimulator` (thread-safe point cloud world), 6 spatial tools + `DroneAgent`, GLUT `DroneVisualizer` (mouse orbit/zoom, r/c/ESC keys)
- `eval_reports/` — generated artifacts (gitignored)
- `workspace/` — runtime sandbox for file tests (gitignored)

## Requirements (what you already use on Orin)

- Python 3 + venv
- `ollama` Python package + running Ollama server with your target small model pulled
- PyOpenGL + PyOpenGL_accelerate + freeglut (for `--viz` / standalone drone viewer)

No heavy frameworks. Stays friendly to limited RAM / CPU.

## Status & Future

Early but already useful dual-purpose harness. Drone 3D viz and the reprioritized natural-language drone tests are the newest direction. More robustness cases, composite skills, and real-sensor paths will be added over time.

Contributions, new interesting natural-language test cases, better prompts that help tiny models succeed more often, and ideas for the next domain are very welcome.

## License

MIT — see LICENSE.

---

**Repo created so the project can be improved gradually in public.**

https://github.com/shawncres/llm-agent-eval