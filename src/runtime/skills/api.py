"""
Canonical skill discovery and execution APIs.

Implements the four NAT-facing runtime interfaces for directory-backed skills:
    list_skills()        -- discover all skills with name, description, tags, files
    search_skills()      -- metadata-only search over skills
    get_skill()          -- load full SKILL.md body or a specific sidecar file
    run_skill_command()  -- execute a script in a skill's directory

Skills are discovered from the filesystem under the skills package directory.
Each skill lives in its own subdirectory and must contain a SKILL.md file
with YAML frontmatter for machine-parseable metadata.

Owns:
    - Skill discovery from directory structure
    - SKILL.md parsing (YAML frontmatter + markdown body)
    - Skill metadata indexing and search
    - Sidecar file loading
    - Script execution within skill directories

Does NOT own:
    - Skill execution logic (see runtime.workflows)
    - Tool implementations (see runtime.tools)
    - Agent orchestration or prompt policy
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Skill metadata types
# ---------------------------------------------------------------------------

@dataclass
class SkillInfo:
    """Parsed metadata from a SKILL.md frontmatter block."""
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    preconditions: list[str] | str = field(default_factory=list)
    next_skills: list[str] = field(default_factory=list)
    directory: str = ""
    files: list[str] = field(default_factory=list)


@dataclass
class SkillSearchResult:
    """Compact search result with metadata only (no body content)."""
    name: str
    description: str
    tags: list[str]
    tools: list[str]
    files: list[str]
    directory: str


@dataclass
class SkillDetail:
    """Full skill content: metadata plus SKILL.md body or sidecar file."""
    info: SkillInfo
    content: str  # full SKILL.md body or sidecar file content


@dataclass
class CommandResult:
    """Result of executing a skill command/script."""
    skill_name: str
    command: str
    return_code: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Internal: SKILL.md parsing
# ---------------------------------------------------------------------------

def _skills_root() -> Path:
    """Return the root directory of the skills package."""
    return Path(__file__).parent


def _parse_skill_md(skill_dir: Path) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into (frontmatter_dict, body_text).

    Returns empty frontmatter and body if the file doesn't exist.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return {}, ""

    text = skill_md.read_text(encoding="utf-8")

    # Split YAML frontmatter from markdown body
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return frontmatter, body
    return {}, text.strip()


def _list_files(skill_dir: Path) -> list[str]:
    """List all files in a skill directory (relative paths)."""
    files = []
    for item in sorted(skill_dir.rglob("*")):
        if item.is_file() and item.name != "__pycache__":
            files.append(str(item.relative_to(skill_dir)))
    return files


def _build_skill_info(skill_dir: Path) -> SkillInfo | None:
    """Build a SkillInfo from a skill directory."""
    frontmatter, _body = _parse_skill_md(skill_dir)
    if not frontmatter:
        return None

    return SkillInfo(
        name=frontmatter.get("name", skill_dir.name),
        description=frontmatter.get("description", ""),
        tags=frontmatter.get("tags", []),
        tools=frontmatter.get("tools", []),
        inputs=frontmatter.get("inputs", {}),
        outputs=frontmatter.get("outputs", {}),
        preconditions=frontmatter.get("preconditions", []),
        next_skills=frontmatter.get("next_skills", []),
        directory=str(skill_dir.relative_to(_skills_root())),
        files=_list_files(skill_dir),
    )


def _discover_skill_dirs() -> list[Path]:
    """Find all subdirectories under the skills root that contain a SKILL.md."""
    root = _skills_root()
    dirs = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and (item / "SKILL.md").exists():
            dirs.append(item)
    return dirs


# ---------------------------------------------------------------------------
# Public API: list_skills
# ---------------------------------------------------------------------------

def list_skills() -> list[SkillInfo]:
    """Discover all skills with name, description, tags, and files.

    Scans the skills directory for subdirectories containing SKILL.md,
    parses their frontmatter, and returns a list of SkillInfo records.
    """
    skills = []
    for skill_dir in _discover_skill_dirs():
        info = _build_skill_info(skill_dir)
        if info is not None:
            skills.append(info)
    return skills


# ---------------------------------------------------------------------------
# Public API: search_skills
# ---------------------------------------------------------------------------

def search_skills(
    query: str = "",
    tags: list[str] | None = None,
    tool: str | None = None,
) -> list[SkillSearchResult]:
    """Search skills by name, description, tags, or tool usage.

    Metadata-only search — returns compact results without loading full
    SKILL.md bodies.

    Args:
        query: Free-text search over name and description (case-insensitive).
        tags: Filter to skills that have ALL of these tags.
        tool: Filter to skills that use this specific tool.
    """
    results = []
    query_lower = query.lower()

    for info in list_skills():
        # Free-text match on name and description
        if query_lower and query_lower not in info.name.lower() and query_lower not in info.description.lower():
            continue

        # Tag filter: skill must have ALL specified tags
        if tags and not all(t in info.tags for t in tags):
            continue

        # Tool filter: skill must include this tool
        if tool and tool not in info.tools:
            continue

        results.append(SkillSearchResult(
            name=info.name,
            description=info.description,
            tags=info.tags,
            tools=info.tools,
            files=info.files,
            directory=info.directory,
        ))

    return results


# ---------------------------------------------------------------------------
# Public API: get_skill
# ---------------------------------------------------------------------------

def get_skill(skill_name: str, file_path: str | None = None) -> SkillDetail | None:
    """Load the full SKILL.md body or a specific sidecar file for a skill.

    Args:
        skill_name: The skill name (from frontmatter) or directory name.
        file_path: Optional relative path to a sidecar file within the
                   skill directory. If None, returns the full SKILL.md body.

    Returns:
        SkillDetail with metadata and content, or None if not found.
    """
    for skill_dir in _discover_skill_dirs():
        info = _build_skill_info(skill_dir)
        if info is None:
            continue
        if info.name != skill_name and skill_dir.name != skill_name:
            continue

        if file_path is None:
            _frontmatter, body = _parse_skill_md(skill_dir)
            return SkillDetail(info=info, content=body)

        target = skill_dir / file_path
        if target.exists() and target.is_file():
            content = target.read_text(encoding="utf-8")
            return SkillDetail(info=info, content=content)

        return None

    return None


# ---------------------------------------------------------------------------
# Public API: run_skill_command
# ---------------------------------------------------------------------------

def run_skill_command(
    skill_name: str,
    command: str,
    timeout: int = 30,
) -> CommandResult | None:
    """Execute a script or command within a skill's directory.

    The command is run with the skill directory as the working directory.
    Only scripts that exist within the skill directory are allowed.

    Args:
        skill_name: The skill name or directory name.
        command: The script filename or command to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        CommandResult with stdout, stderr, and return code, or None if
        the skill or command is not found.
    """
    for skill_dir in _discover_skill_dirs():
        info = _build_skill_info(skill_dir)
        if info is None:
            continue
        if info.name != skill_name and skill_dir.name != skill_name:
            continue

        # Security: only allow executing files that exist in the skill dir
        script_path = skill_dir / command
        if not script_path.exists() or not script_path.is_file():
            return CommandResult(
                skill_name=skill_name,
                command=command,
                return_code=1,
                stdout="",
                stderr=f"Script '{command}' not found in skill directory.",
            )

        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=str(skill_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "SKILL_DIR": str(skill_dir)},
            )
            return CommandResult(
                skill_name=skill_name,
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                skill_name=skill_name,
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s.",
            )

    return None


# ---------------------------------------------------------------------------
# Convenience: build skill transition graph from SKILL.md metadata
# ---------------------------------------------------------------------------

def build_skill_transitions() -> dict[str, set[str]]:
    """Build the valid skill transition graph from SKILL.md metadata.

    Returns a dict of skill_name -> set of valid next skill names.
    """
    transitions: dict[str, set[str]] = {}
    for info in list_skills():
        transitions[info.name] = set(info.next_skills)
    return transitions


def build_skill_tool_patterns() -> dict[str, list[str]]:
    """Build the allowed tool patterns per skill from SKILL.md metadata.

    Returns a dict of skill_name -> ordered list of tool names.
    """
    return {info.name: info.tools for info in list_skills()}


def build_skill_order() -> list[str]:
    """Build the canonical skill execution order from SKILL.md metadata.

    Topologically sorts skills based on preconditions and next_skills.
    Falls back to discovery order if the graph can't be sorted.
    """
    skills = list_skills()
    name_to_info = {s.name: s for s in skills}

    # Find the entry skill (no preconditions or preconditions == "none")
    entry_skills = [
        s for s in skills
        if not s.preconditions or s.preconditions == "none"
    ]

    if not entry_skills:
        return [s.name for s in skills]

    # Walk the next_skills chain from the entry point
    order = []
    visited = set()
    queue = [entry_skills[0].name]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        order.append(current)

        info = name_to_info.get(current)
        if info:
            for next_skill in info.next_skills:
                if next_skill not in visited:
                    queue.append(next_skill)

    return order
