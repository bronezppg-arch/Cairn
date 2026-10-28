"""Extract a structured network-topology graph from a project's free-text facts.

Facts in Cairn are prose written by agents, so there is no structured host/cred
data to draw from. This module asks a locally-configured LLM CLI (the same ones
Cairn's local-mode workers reuse) to read all facts and emit a strict JSON
topology: hosts (ports, creds found, creds that work, access level, zone) and
edges (visibility / pivot / cred-works). The result is cached by the caller.

No API keys are used — it relies on a host CLI that is already logged in, exactly
like local-mode execution. Two providers are supported out of the box:

* ``claude``  -> `claude -p --tools "" --max-turns 1 [--model X]`  (JSON on stdout)
* ``codex``   -> `codex exec -s read-only -o FILE [-m X] -`         (JSON in FILE)

Which models appear in the UI is decided by :func:`list_models`, and availability
is detected at runtime from whether the backing CLI is on PATH. Power users can
point at any other backend (Qwen, GLM, DeepSeek, …) with the CAIRN_TOPO_CMD env
var, which fully overrides the command and reads JSON from stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

TIMEOUT_S = int(os.environ.get("CAIRN_TOPO_TIMEOUT", "300"))

# Ordered registry of extractor options offered in the UI. `cli` is the binary
# that must be on PATH for the option to be usable; `model` is the alias passed
# to that CLI (None = use the CLI's own default / logged-in model).
_MODELS: list[dict] = [
    {
        "id": "claude-sonnet",
        "label": "Claude Sonnet",
        "provider": "claude",
        "cli": "claude",
        "model": "sonnet",
        "note": "Fast and reliable on security notes.",
    },
    {
        "id": "claude-opus",
        "label": "Claude Opus",
        "provider": "claude",
        "cli": "claude",
        "model": "opus",
        "note": "Most capable, but ~2.5x slower.",
    },
    {
        "id": "claude-haiku",
        "label": "Claude Haiku",
        "provider": "claude",
        "cli": "claude",
        "model": "haiku",
        "note": "Not recommended: cyber safeguards reject offensive-security notes.",
        "discouraged": True,
    },
    {
        "id": "codex",
        "label": "ChatGPT (Codex)",
        "provider": "codex",
        "cli": "codex",
        "model": None,  # use whatever model the codex CLI is logged into
        "note": "Uses your ChatGPT / codex login.",
    },
]

# First available wins when auto-picking the green "recommended" option.
_PREFERENCE = ["claude-sonnet", "codex", "claude-opus", "claude-haiku"]

_SCHEMA_HINT = """\
Output ONLY one raw JSON object (no markdown, no prose, no code fences) with this shape:

{
  "hosts": [
    {
      "id": "<ip or hostname, stable key>",
      "name": "<short label, e.g. 'MSSQL' or 'dc.corp.pipeline.stf'>",
      "zone": "attacker|external|dmz|internal|ics",
      "ports": [{"port": 1433, "service": "MSSQL"}],
      "creds_found": ["user:pass or key description discovered ON/ABOUT this host"],
      "creds_work": ["cred that is CONFIRMED working against this host, with the level it grants"],
      "access": "none|user|admin|system|root",
      "notes": "<one short line: how it was compromised / why it matters>"
    }
  ],
  "edges": [
    {"from": "<host id>", "to": "<host id>", "type": "visibility|pivot|cred-works", "label": "<optional short>"}
  ]
}

Rules:
- Always include a host with id "attacker" (zone "attacker") for our attacking machine.
- Include every host, subnet pivot, and ICS/OT target mentioned in the facts.
- "access" is the HIGHEST confirmed foothold on that host: root (Linux root), system (NT SYSTEM/SYSTEM),
  admin (local/domain admin or www-data+SUID-to-root path not yet run -> use user), user (any shell/valid login), none (only scanned).
- edge "visibility" = X can see/scan Y; "pivot" = we route through X to reach Y; "cred-works" = a credential proven to authenticate X->Y.
- Deduplicate hosts by IP. Keep notes to one short sentence. Do not invent data not supported by the facts.
"""


def _custom_cmd() -> list[str] | None:
    raw = os.environ.get("CAIRN_TOPO_CMD")
    return raw.split() if raw else None


def _entry(model_id: str) -> dict | None:
    return next((m for m in _MODELS if m["id"] == model_id), None)


def _recommended(available: dict[str, bool], custom: bool) -> str | None:
    """The option to highlight green: a custom command wins, else the first
    available model in preference order."""
    if custom:
        return "custom"
    for mid in _PREFERENCE:
        if available.get(mid):
            return mid
    return None


def list_models() -> dict:
    """Describe the extractor models for the settings UI.

    Returns {"models": [...], "default": <id|None>}. Each model carries
    availability (is its CLI installed?) and whether it is the recommended pick.
    """
    available = {m["id"]: bool(shutil.which(m["cli"])) for m in _MODELS}
    custom = _custom_cmd() is not None
    recommended = _recommended(available, custom)

    models: list[dict] = []
    if custom:
        models.append({
            "id": "custom",
            "label": "Custom (CAIRN_TOPO_CMD)",
            "provider": "custom",
            "available": True,
            "recommended": recommended == "custom",
            "discouraged": False,
            "note": "Uses the command in CAIRN_TOPO_CMD (any backend).",
            "unavailable_reason": None,
        })
    for m in _MODELS:
        ok = available[m["id"]]
        models.append({
            "id": m["id"],
            "label": m["label"],
            "provider": m["provider"],
            "available": ok,
            "recommended": m["id"] == recommended,
            "discouraged": m.get("discouraged", False),
            "note": m["note"],
            "unavailable_reason": None if ok else f"{m['cli']} CLI not found on PATH",
        })
    return {"models": models, "default": recommended}


def _default_model_id() -> str | None:
    """Pick the model when the caller did not specify one."""
    env = os.environ.get("CAIRN_TOPO_MODEL")
    if env:
        return env
    if _custom_cmd() is not None:
        return "custom"
    return list_models()["default"]


def _build_prompt(origin: str, goal: str, facts: list[dict]) -> str:
    lines = [
        "You are a penetration-test note analyst. Read the ORIGIN, GOAL and FACTS of an",
        "engagement and build a network topology graph from them.",
        "",
        f"ORIGIN:\n{origin}",
        "",
        f"GOAL:\n{goal}",
        "",
        "FACTS (chronological, free text written by the testing agents):",
    ]
    for f in facts:
        fid = f.get("id", "?")
        desc = (f.get("description") or "").strip()
        lines.append(f"\n[{fid}]\n{desc}")
    lines.append("\n\n" + _SCHEMA_HINT)
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    # strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    # fall back to slicing the outermost braces
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict) or "hosts" not in obj:
        raise ValueError("model output is not a topology object")
    obj.setdefault("edges", [])
    return obj


def _guard_refusals(text: str) -> None:
    low = text.lower()
    if "session limit" in low or "usage limit" in low or "rate limit" in low:
        raise RuntimeError(f"model backend is rate/usage limited: {text.strip()[:200]}")
    # Claude Haiku (and tighter models) reject offensive-security content.
    if "[cyber]" in low or "safeguards flagged" in low:
        raise RuntimeError(
            "model refused the content via cyber safeguards — pick a different "
            "model (Claude Sonnet/Opus handle security notes)"
        )


def _run_claude(exe: str, model: str | None, prompt: str) -> str:
    # Pure text->JSON transform: disable all tools so the model cannot spend its
    # single turn on tool calls and must answer directly.
    cmd = [exe, "-p", "--tools", "", "--max-turns", "1"]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S)
    _guard_refusals(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"extractor exited {proc.returncode}: {tail}")
    return proc.stdout


def _run_codex(exe: str, model: str | None, prompt: str) -> str:
    # Read-only sandbox (no command execution); -o writes just the final message
    # so we get clean JSON instead of the event log.
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        outfile = tf.name
    try:
        cmd = [exe, "exec", "-s", "read-only", "--skip-git-repo-check", "--color", "never", "-o", outfile]
        if model:
            cmd += ["-m", model]
        cmd.append("-")  # read prompt from stdin
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S)
        _guard_refusals(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise RuntimeError(f"extractor exited {proc.returncode}: {tail}")
        try:
            with open(outfile, encoding="utf-8") as fh:
                out = fh.read()
        except OSError:
            out = proc.stdout
        return out or proc.stdout
    finally:
        try:
            os.unlink(outfile)
        except OSError:
            pass


def extract_topology(origin: str, goal: str, facts: list[dict], model_id: str | None = None) -> dict:
    """Run a host CLI over the facts and return the parsed topology dict.

    ``model_id`` selects an option from :func:`list_models`; None auto-picks.
    Raises RuntimeError with a readable message on any failure.
    """
    prompt = _build_prompt(origin, goal, facts)
    model_id = model_id or _default_model_id()

    custom = _custom_cmd()
    try:
        if model_id == "custom" or (custom is not None and model_id is None):
            if not custom:
                raise RuntimeError("custom model selected but CAIRN_TOPO_CMD is not set")
            proc = subprocess.run(custom, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S)
            _guard_refusals(proc.stdout + "\n" + proc.stderr)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-400:]
                raise RuntimeError(f"extractor exited {proc.returncode}: {tail}")
            raw = proc.stdout
        else:
            entry = _entry(model_id) if model_id else None
            if entry is None:
                raise RuntimeError(f"unknown extractor model '{model_id}'")
            exe = shutil.which(entry["cli"])
            if not exe:
                raise RuntimeError(f"{entry['cli']} CLI not found on PATH for model '{model_id}'")
            if entry["provider"] == "codex":
                raw = _run_codex(exe, entry["model"], prompt)
            else:
                raw = _run_claude(exe, entry["model"], prompt)
    except FileNotFoundError:
        raise RuntimeError(f"extractor command not found (model '{model_id}')")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"topology extraction timed out after {TIMEOUT_S}s")

    try:
        return _parse_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"could not parse topology JSON: {e}")
