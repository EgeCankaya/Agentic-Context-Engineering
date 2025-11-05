"""
Versioning utilities for ACE playbook management.
Handles semantic versioning, Git integration, and playbook comparison.
"""

import re
import subprocess
from typing import Dict, List


def increment_version(version: str, bump_type: str) -> str:
    """
    Increment semantic version based on bump type.

    Args:
        version: Current version (e.g., "1.2.3")
        bump_type: "major", "minor", or "patch"

    Returns:
        New version string
    """
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = map(int, version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump_type: {bump_type}. Must be 'major', 'minor', or 'patch'")


def get_version_info(version: str) -> Dict[str, int]:
    """Parse version string into components."""
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = map(int, version.split("."))
    return {"major": major, "minor": minor, "patch": patch}


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two semantic versions.

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    v1_info = get_version_info(version1)
    v2_info = get_version_info(version2)

    for key in ["major", "minor", "patch"]:
        if v1_info[key] < v2_info[key]:
            return -1
        elif v1_info[key] > v2_info[key]:
            return 1

    return 0


def suggest_version_bump(change_types: List[str]) -> str:
    """
    Suggest version bump type based on change types.

    Args:
        change_types: List of change types from HistoryEntry

    Returns:
        Suggested bump type: "major", "minor", or "patch"
    """
    if any("heuristic_removed" in ct or "instruction_updated" in ct for ct in change_types):
        return "major"  # Breaking changes
    elif any("heuristic_added" in ct for ct in change_types):
        return "minor"  # New features
    else:
        return "patch"  # Bug fixes and refinements


def commit_playbook(playbook_path: str, version: str, message: str = None) -> bool:
    """
    Commit playbook to Git with appropriate message.

    Args:
        playbook_path: Path to playbook YAML file
        version: Playbook version
        message: Custom commit message (optional)

    Returns:
        True if commit successful, False otherwise
    """
    try:
        if message is None:
            message = f"ACE playbook v{version}: Update context and heuristics"

        # Add file to git
        subprocess.run(["git", "add", playbook_path], check=True, capture_output=True)

        # Commit with message
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Git commit failed: {e}")
        return False


def tag_playbook_version(version: str, message: str = None) -> bool:
    """
    Create Git tag for playbook version.

    Args:
        version: Playbook version
        message: Tag message (optional)

    Returns:
        True if tag created successfully, False otherwise
    """
    try:
        tag_name = f"v{version}"
        if message is None:
            message = f"ACE playbook version {version}"

        subprocess.run(["git", "tag", "-a", tag_name, "-m", message], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Git tag failed: {e}")
        return False


def compare_playbooks(playbook1_path: str, playbook2_path: str) -> Dict:
    """
    Compare two playbook versions and return differences.

    Args:
        playbook1_path: Path to first playbook
        playbook2_path: Path to second playbook

    Returns:
        Dictionary with comparison results
    """
    from .playbook_schema import Playbook

    playbook1 = Playbook.from_yaml(playbook1_path)
    playbook2 = Playbook.from_yaml(playbook2_path)

    comparison = {
        "version_diff": {"from": playbook1.version, "to": playbook2.version},
        "heuristics": {"added": [], "removed": [], "updated": []},
        "examples": {
            "added": len(playbook2.context.few_shot_examples) - len(playbook1.context.few_shot_examples),
            "total_before": len(playbook1.context.few_shot_examples),
            "total_after": len(playbook2.context.few_shot_examples),
        },
        "performance": {
            "accuracy_delta": playbook2.metadata.performance_metrics.accuracy
            - playbook1.metadata.performance_metrics.accuracy,
            "bleu_delta": (playbook2.metadata.performance_metrics.bleu_score or 0)
            - (playbook1.metadata.performance_metrics.bleu_score or 0),
            "tokens_delta": playbook2.metadata.performance_metrics.avg_tokens
            - playbook1.metadata.performance_metrics.avg_tokens,
        },
    }

    # Compare heuristics
    h1_ids = {h.id for h in playbook1.context.heuristics}
    h2_ids = {h.id for h in playbook2.context.heuristics}

    # Added heuristics
    for h in playbook2.context.heuristics:
        if h.id not in h1_ids:
            comparison["heuristics"]["added"].append({"id": h.id, "rule": h.rule, "confidence": h.confidence})

    # Removed heuristics
    for h in playbook1.context.heuristics:
        if h.id not in h2_ids:
            comparison["heuristics"]["removed"].append({"id": h.id, "rule": h.rule})

    # Updated heuristics
    for h2 in playbook2.context.heuristics:
        h1 = playbook1.get_heuristic_by_id(h2.id)
        if h1 and (h1.rule != h2.rule or h1.confidence != h2.confidence):
            comparison["heuristics"]["updated"].append({
                "id": h2.id,
                "rule_before": h1.rule,
                "rule_after": h2.rule,
                "confidence_before": h1.confidence,
                "confidence_after": h2.confidence,
            })

    return comparison


def get_playbook_history(playbook_path: str) -> List[Dict]:
    """
    Get Git history for a playbook file.

    Args:
        playbook_path: Path to playbook file

    Returns:
        List of commit information
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--follow", playbook_path], capture_output=True, text=True, check=True
        )

        commits = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    commits.append({"hash": parts[0], "message": parts[1]})

        return commits
    except subprocess.CalledProcessError:
        return []


def rollback_to_version(playbook_path: str, version: str) -> bool:
    """
    Rollback playbook to a specific version.

    Args:
        playbook_path: Path to playbook file
        version: Target version to rollback to

    Returns:
        True if rollback successful, False otherwise
    """
    try:
        # Find commit with the target version
        result = subprocess.run(
            ["git", "log", "--grep", f"v{version}", "--oneline", playbook_path], capture_output=True, text=True
        )

        if not result.stdout.strip():
            print(f"No commit found for version {version}")
            return False

        # Get the commit hash
        commit_hash = result.stdout.split()[0]

        # Checkout the specific version of the file
        subprocess.run(["git", "checkout", commit_hash, "--", playbook_path], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Rollback failed: {e}")
        return False
