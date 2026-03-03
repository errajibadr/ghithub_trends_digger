#!/usr/bin/env python3
"""
Monorepo Version Management Script.

Two modes:
  1. bump   - Simple unified versioning (all packages same version)
  2. release - Advanced per-package versioning (semantic release based on commits)

Usage:
    # Simple: Bump all packages to same version
    python scripts/version.py bump patch          # 0.1.0 → 0.1.1
    python scripts/version.py bump minor          # 0.1.0 → 0.2.0
    python scripts/version.py bump major          # 0.1.0 → 1.0.0
    python scripts/version.py bump --dry-run patch

    # Advanced: Per-package semantic release
    python scripts/version.py release --dry-run   # Preview releases
    python scripts/version.py release             # Release packages with changes
    python scripts/version.py release --package core --force-bump minor
"""

import argparse
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================


class BumpType(Enum):
    """Version bump types ordered by priority."""

    NONE = 0
    PATCH = 1
    MINOR = 2
    MAJOR = 3


@dataclass
class Package:
    """Package configuration."""

    name: str
    path: Path
    pyproject_path: Path
    init_path: Path | None


# Package registry - maps scope keyword to package config
PACKAGES: dict[str, Package] = {
    "core": Package(
        name="sta-agent-core",
        path=Path("packages/sta_agent_core"),
        pyproject_path=Path("packages/sta_agent_core/pyproject.toml"),
        init_path=Path("packages/sta_agent_core/src/sta_agent_core/__init__.py"),
    ),
    "engine": Package(
        name="sta-agent-engine",
        path=Path("packages/sta_agent_engine"),
        pyproject_path=Path("packages/sta_agent_engine/pyproject.toml"),
        init_path=Path("packages/sta_agent_engine/src/sta_agent_engine/__init__.py"),
    ),
    "frontend": Package(
        name="sta-agent-frontend",
        path=Path("packages/sta_agent_frontend"),
        pyproject_path=Path("packages/sta_agent_frontend/pyproject.toml"),
        init_path=None,  # No __version__ in frontend
    ),
    "toolkit": Package(
        name="sta-nxgraph-toolkit",
        path=Path("packages/sta_nxgraph_toolkit"),
        pyproject_path=Path("packages/sta_nxgraph_toolkit/pyproject.toml"),
        init_path=Path("packages/sta_nxgraph_toolkit/src/sta_nxgraph_toolkit/__init__.py"),
    ),
}


# =============================================================================
# COMMIT PARSING
# =============================================================================

# Strict Conventional Commit pattern
COMMIT_PATTERN = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r":[ ]?(?P<message>.+)$",
    re.IGNORECASE,
)

# Legacy patterns for non-conventional commits
LEGACY_FEAT_PATTERN = re.compile(
    r"^(feat|feature|add|implement|new)[:/\s]",
    re.IGNORECASE,
)
LEGACY_FIX_PATTERN = re.compile(
    r"^(fix|bugfix|hotfix|patch|resolve)[:/\s]",
    re.IGNORECASE,
)


def parse_commit(message: str) -> tuple[str | None, str | None, bool]:
    """Parse a commit message. Returns (type, scope, is_breaking).

    Supports both strict Conventional Commits and relaxed legacy formats.
    """
    # Try strict conventional commit format first
    match = COMMIT_PATTERN.match(message)
    if match:
        return (
            match.group("type").lower(),
            match.group("scope"),
            bool(match.group("breaking")) or "BREAKING CHANGE" in message,
        )

    # Try legacy feature patterns (Feat/, Feature, Add, etc.)
    if LEGACY_FEAT_PATTERN.match(message):
        return "feat", None, "BREAKING" in message.upper()

    # Try legacy fix patterns (Fix/, Bugfix, Hotfix, etc.)
    if LEGACY_FIX_PATTERN.match(message):
        return "fix", None, "BREAKING" in message.upper()

    return None, None, False


# =============================================================================
# GIT UTILITIES
# =============================================================================


def run_git(args: list[str], check: bool = True) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def validate_clean_state() -> bool:
    """Fail fast if working tree has uncommitted changes. Returns False if dirty."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if status:
        print("❌ Working directory is dirty. Commit or stash changes first.")
        print(f"   Dirty files:\n{status}")
        print("   Use --no-require-clean to skip this check.")
        return False
    return True


def tag_exists(tag_name: str) -> bool:
    """Return True if the given tag exists locally."""
    result = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag_name}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def commit_with_retry(message: str, files_to_stage: list[Path], max_retries: int = 3) -> bool:
    """Commit with automatic retry if pre-commit hooks modify files.

    Pre-commit hooks (like ruff --fix, end-of-file-fixer) may modify staged files,
    causing the commit to fail. This function detects that case and re-stages + retries.

    Args:
        message: Commit message
        files_to_stage: List of file paths to stage
        max_retries: Maximum number of retry attempts

    Returns:
        True if commit succeeded, False otherwise
    """
    for attempt in range(max_retries):
        for f in files_to_stage:
            if f.exists():
                run_git(["add", str(f)])

        uv_lock_status = run_git(["status", "--porcelain", "uv.lock"], check=False)
        if uv_lock_status:
            run_git(["add", "uv.lock"])

        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return True

        # Detect working-tree changes: column 2 of `git status --porcelain` is
        # the working-tree indicator. Catches  M, MM, AM, etc.
        status = run_git(["status", "--porcelain"], check=False)
        has_working_changes = any(len(line) >= 2 and line[1] != " " for line in status.split("\n") if line.strip())

        if not has_working_changes:
            print(f"  ⚠️  Commit failed: {result.stderr.strip() or result.stdout.strip()}")
            return False

        if attempt < max_retries - 1:
            print(f"  ↻ Pre-commit modified files, re-staging... (attempt {attempt + 2}/{max_retries})")
        else:
            print(f"  ⚠️  Commit failed after {max_retries} attempts. Pre-commit keeps modifying files.")
            return False

    return False


def run_uv_lock() -> bool:
    """Run uv lock to update the lockfile after version changes.

    Returns True if lockfile was updated, False otherwise.
    """
    print("  🔒 Updating uv.lock...")
    try:
        subprocess.run(
            ["uv", "lock"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check if uv.lock was modified
        status = run_git(["status", "--porcelain", "uv.lock"])
        if status:
            print("  ✓ uv.lock updated")
            return True
        print("  ✓ uv.lock unchanged")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Warning: uv lock failed: {e.stderr}")
        return False


def parse_version_from_tag(tag: str, package_name: str) -> tuple[int, int, int] | None:
    """Extract version tuple from a tag string. Returns (major, minor, patch) or None."""
    prefix = f"{package_name}-v"
    if not tag.startswith(prefix):
        return None
    version_str = tag[len(prefix) :]
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def parse_version_string(version: str) -> tuple[int, int, int]:
    """Parse a version string like '0.3.3' into (major, minor, patch)."""
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2].split("-")[0])
    return major, minor, patch


def get_last_tag_for_package(package_name: str, max_version: str | None = None) -> str | None:
    """Get the last git tag for a specific package.

    Args:
        package_name: The package name (e.g., 'sta-agent-core')
        max_version: Optional max version string (e.g., '0.3.3'). If provided,
                     only tags with version <= max_version are considered.
                     This prevents issues when a higher version tag exists
                     (e.g., from a different branch) than the current pyproject version.

    Returns:
        The tag string or None if no matching tag found.
    """
    try:
        tags = run_git(["tag", "-l", f"{package_name}-v*", "--sort=-v:refname"])
        if not tags:
            return None

        tag_list = tags.split("\n")

        # If no max_version constraint, return the highest tag
        if max_version is None:
            return tag_list[0]

        # Filter tags to only those <= max_version
        max_ver_tuple = parse_version_string(max_version)

        for tag in tag_list:
            tag_ver = parse_version_from_tag(tag, package_name)
            if tag_ver and tag_ver <= max_ver_tuple:
                return tag

        # No tag found <= current version (all tags are higher)
        return None

    except subprocess.CalledProcessError:
        pass
    return None


def get_commits_since_tag(tag: str | None, package_path: Path) -> list[str]:
    """Get commits affecting a package since the last tag."""
    range_arg = f"{tag}..HEAD" if tag else "HEAD"

    try:
        commits = run_git(
            [
                "log",
                range_arg,
                "--pretty=format:%s",
                "--",
                str(package_path),
            ]
        )
        return [c for c in commits.split("\n") if c.strip()]
    except subprocess.CalledProcessError:
        return []


# =============================================================================
# VERSION UTILITIES
# =============================================================================


def get_current_version(pyproject_path: Path) -> str:
    """Extract current version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError(f"Could not find version in {pyproject_path}")


def calculate_new_version(version: str, bump_type: BumpType) -> str:
    """Calculate new version based on bump type."""
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2].split("-")[0])

    if bump_type == BumpType.MAJOR:
        return f"{major + 1}.0.0"
    elif bump_type == BumpType.MINOR:
        return f"{major}.{minor + 1}.0"
    elif bump_type == BumpType.PATCH:
        return f"{major}.{minor}.{patch + 1}"
    return version


def calculate_prerelease_version(base_version: str, suffix: str, package_name: str) -> str:
    """Generate PEP 440 pre-release: 0.2.0.dev1, 0.2.0.dev2, etc.

    Build number is derived from counting existing pre-release tags.
    """
    existing = run_git(
        ["tag", "-l", f"{package_name}-v{base_version}.{suffix}*", "--sort=-v:refname"],
        check=False,
    )
    tag_list = [t for t in existing.split("\n") if t.strip()]
    build_num = len(tag_list)
    return f"{base_version}.{suffix}{build_num + 1}"


def update_version_in_file(file_path: Path, old_version: str, new_version: str) -> bool:
    """Update version string in a file. Returns True if changed."""
    if not file_path or not file_path.exists():
        return False

    content = file_path.read_text()
    new_content = content.replace(f'version = "{old_version}"', f'version = "{new_version}"')
    new_content = new_content.replace(f'__version__ = "{old_version}"', f'__version__ = "{new_version}"')

    if content != new_content:
        file_path.write_text(new_content)
        return True
    return False


# =============================================================================
# CHANGELOG
# =============================================================================


def generate_changelog_entry(analysis: dict) -> str:
    """Build markdown changelog entry grouped by commit type."""
    version = analysis["new_version"]
    date = datetime.date.today().isoformat()

    features, fixes, others = [], [], []
    for commit in analysis["commits"]:
        if commit.startswith("chore: bump version") or (commit.startswith("chore(") and ("release" in commit.lower() or "version" in commit.lower())):
            continue
        commit_type, _, _ = parse_commit(commit)
        if commit_type == "feat":
            features.append(commit)
        elif commit_type == "fix":
            fixes.append(commit)
        elif commit_type:
            others.append(commit)

    lines = [f"## [{version}] - {date}\n"]
    if features:
        lines += ["### Added"] + [f"- {c}" for c in features]
    if fixes:
        lines += ["### Fixed"] + [f"- {c}" for c in fixes]
    if others:
        lines += ["### Other"] + [f"- {c}" for c in others]
    return "\n".join(lines) + "\n"


def update_changelog(package: Package, entry: str) -> Path | None:
    """Prepend entry to package CHANGELOG.md (create if missing). Returns path if updated."""
    changelog_path = package.path / "CHANGELOG.md"
    if changelog_path.exists():
        existing = changelog_path.read_text()
        if existing.startswith("# Changelog"):
            header, rest = existing.split("\n", 1)
            content = f"{header}\n\n{entry}\n{rest}"
        else:
            content = f"# Changelog\n\n{entry}\n{existing}"
    else:
        content = f"# Changelog\n\n{entry}"

    if not content.endswith("\n"):
        content += "\n"
    changelog_path.write_text(content)
    return changelog_path


# =============================================================================
# BUMP COMMAND (Simple Unified Versioning)
# =============================================================================


def cmd_bump(args: argparse.Namespace) -> int:
    """Bump all packages to the same version."""
    bump_type_map = {"patch": BumpType.PATCH, "minor": BumpType.MINOR, "major": BumpType.MAJOR}
    bump_type = bump_type_map[args.bump_type]

    current_version = get_current_version(PACKAGES["core"].pyproject_path)
    new_version = calculate_new_version(current_version, bump_type)

    print(f"📦 Version Bump: {current_version} → {new_version} ({args.bump_type.upper()})")
    print()

    if args.dry_run:
        print("[DRY RUN] Would update:")
        for pkg in PACKAGES.values():
            print(f"  • {pkg.pyproject_path}")
            if pkg.init_path:
                print(f"  • {pkg.init_path}")
        print(f"\n[DRY RUN] Would create tag: v{new_version}")
        return 0

    # Update all files
    files_updated = []
    for pkg in PACKAGES.values():
        if update_version_in_file(pkg.pyproject_path, current_version, new_version):
            files_updated.append(pkg.pyproject_path)
            print(f"  ✓ Updated {pkg.pyproject_path}")

        if pkg.init_path and update_version_in_file(pkg.init_path, current_version, new_version):
            files_updated.append(pkg.init_path)
            print(f"  ✓ Updated {pkg.init_path}")

    if not files_updated:
        print("⚠️  No files were updated (versions may already be set)")
        return 1

    # Update lockfile to match new versions (prevents pre-commit from failing)
    print("\n🔒 Syncing lockfile...")
    lockfile_updated = run_uv_lock()

    # Git operations
    if not args.no_commit:
        print("\n📝 Committing changes...")
        # Include uv.lock in files to stage if it was updated
        all_files = list(files_updated)
        if lockfile_updated:
            all_files.append(Path("uv.lock"))

        if commit_with_retry(f"chore: bump version to {new_version}", all_files):
            print("  ✓ Changes committed")
        else:
            print("  ❌ Failed to commit changes")
            return 1

    if not args.no_tag:
        print("\n🏷️  Creating tag...")
        tag_name = f"v{new_version}"
        run_git(["tag", "-a", tag_name, "-m", f"Release {new_version}"])
        print(f"  ✓ Created tag: {tag_name}")

    print(f"\n✅ Version bumped to {new_version}")
    print("\nNext steps:")
    print("  git push && git push --tags")

    return 0


# =============================================================================
# RELEASE COMMAND (Per-Package Semantic Versioning)
# =============================================================================


def determine_bump_from_commits(commits: list[str]) -> BumpType:
    """Determine version bump type from commits.

    If commits exist but none are parseable, defaults to PATCH.
    """
    bump = BumpType.NONE
    has_unparsed_commits = False

    for commit in commits:
        # Skip version bump commits
        if commit.startswith("chore: bump version") or commit.startswith("chore(") and ("release" in commit.lower() or "version" in commit.lower()):
            continue

        commit_type, _, is_breaking = parse_commit(commit)

        if commit_type is None:
            has_unparsed_commits = True
            continue

        # Breaking change = immediate major bump
        if is_breaking:
            return BumpType.MAJOR

        if commit_type == "feat":
            bump = max(bump, BumpType.MINOR, key=lambda x: x.value)
        elif commit_type == "fix":
            bump = max(bump, BumpType.PATCH, key=lambda x: x.value)

    # If we have commits but couldn't parse any, default to PATCH
    if bump == BumpType.NONE and has_unparsed_commits:
        bump = BumpType.PATCH

    return bump


def analyze_package(scope: str, package: Package) -> dict:
    """Analyze a package and determine if it needs a release."""
    current_version = get_current_version(package.pyproject_path)

    # Pass current_version as max_version to avoid using tags higher than pyproject version.
    # This handles cases where a higher version tag exists (e.g., from a different branch).
    last_tag = get_last_tag_for_package(package.name, max_version=current_version)
    commits = get_commits_since_tag(last_tag, package.path)
    bump_type = determine_bump_from_commits(commits)

    new_version = calculate_new_version(current_version, bump_type) if bump_type != BumpType.NONE else current_version

    return {
        "scope": scope,
        "package": package,
        "last_tag": last_tag,
        "commits": commits,
        "bump_type": bump_type,
        "current_version": current_version,
        "new_version": new_version,
        "needs_release": bump_type != BumpType.NONE,
    }


def release_package(
    analysis: dict,
    dry_run: bool = False,
    changelog: bool = False,
    quiet: bool = False,
) -> str | None:
    """Release a single package. Returns the tag name if released."""
    if not analysis["needs_release"]:
        return None

    package: Package = analysis["package"]
    old_version = analysis["current_version"]
    new_version = analysis["new_version"]

    def out(msg: str) -> None:
        if not quiet:
            print(msg)

    out(f"\n{'[DRY RUN] ' if dry_run else ''}Releasing {package.name}: {old_version} → {new_version}")
    out(f"  Bump type: {analysis['bump_type'].name}")
    out(f"  Commits since {analysis['last_tag'] or 'beginning'}:")
    for commit in analysis["commits"][:5]:
        out(f"    - {commit[:60]}...")
    if len(analysis["commits"]) > 5:
        out(f"    ... and {len(analysis['commits']) - 5} more")

    if not dry_run:
        # Update pyproject.toml
        if update_version_in_file(package.pyproject_path, old_version, new_version):
            out(f"  ✓ Updated {package.pyproject_path}")

        # Update __init__.py if it exists
        if package.init_path and update_version_in_file(package.init_path, old_version, new_version):
            out(f"  ✓ Updated {package.init_path}")

        # Changelog
        if changelog:
            entry = generate_changelog_entry(analysis)
            changelog_path = update_changelog(package, entry)
            if changelog_path:
                out(f"  ✓ Updated {changelog_path}")

        # Update lockfile to match new versions (prevents pre-commit from failing)
        if not quiet:
            lockfile_updated = run_uv_lock()
        else:
            subprocess.run(["uv", "lock"], capture_output=True, text=True, check=True)
            status = subprocess.run(
                ["git", "status", "--porcelain", "uv.lock"],
                capture_output=True,
                text=True,
            )
            lockfile_updated = bool(status.stdout.strip())

        # Collect files to stage
        files_to_stage: list[Path] = [package.pyproject_path]
        if package.init_path:
            files_to_stage.append(package.init_path)
        if changelog:
            cl_path = package.path / "CHANGELOG.md"
            if cl_path.exists():
                files_to_stage.append(cl_path)
        if lockfile_updated:
            files_to_stage.append(Path("uv.lock"))

        # Commit version bump with retry (handles pre-commit auto-fixes)
        if not commit_with_retry(f"chore({analysis['scope']}): release {new_version}", files_to_stage):
            if not quiet:
                print(f"  ❌ Failed to commit release for {package.name}")
            return None

        # Create tag
        tag_name = f"{package.name}-v{new_version}"
        result = subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {package.name} v{new_version}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if not quiet:
                print(f"  ❌ Failed to create tag: {result.stderr.strip() or result.stdout.strip() or 'unknown error'}")
            return None
        out(f"  ✓ Created tag: {tag_name}")

        return tag_name

    return f"{package.name}-v{new_version}"


def cmd_release(args: argparse.Namespace) -> int:
    """Release packages based on conventional commits."""
    if not args.no_require_clean and not args.dry_run and not validate_clean_state():
        return 1

    json_output = getattr(args, "json_output", False)
    if not json_output:
        print("🔍 Analyzing packages for release...\n")

    # Filter packages if specific one requested
    packages_to_check = {args.package: PACKAGES[args.package]} if args.package else PACKAGES

    analyses = []
    for scope, package in packages_to_check.items():
        analysis = analyze_package(scope, package)

        # Override bump type if forced
        if args.force_bump and args.package:
            forced_bump = {"patch": BumpType.PATCH, "minor": BumpType.MINOR, "major": BumpType.MAJOR}[args.force_bump]
            analysis["bump_type"] = forced_bump
            analysis["new_version"] = calculate_new_version(analysis["current_version"], forced_bump)
            analysis["needs_release"] = True

        analyses.append(analysis)

    releases_needed = [a for a in analyses if a["needs_release"]]

    if not json_output:
        print("📦 Package Analysis:")
        print("-" * 60)
        for a in analyses:
            status = "🚀 RELEASE" if a["needs_release"] else "⏸️  no change"
            version_info = f"{a['current_version']} → {a['new_version']}" if a["needs_release"] else a["current_version"]
            print(f"  {a['package'].name:25} {status:15} {version_info}")

    if not releases_needed:
        if not json_output:
            print("\n✅ No releases needed. All packages are up to date.")
        if json_output:
            import json

            result = {
                "released": [],
                "unchanged": [{"package": a["package"].name, "version": a["current_version"]} for a in analyses],
                "dry_run": args.dry_run,
            }
            print(json.dumps(result, indent=2))
        return 0

    if getattr(args, "prerelease", None):
        for a in releases_needed:
            a["new_version"] = calculate_prerelease_version(a["new_version"], args.prerelease, a["package"].name)

    # Abort entire release if any target tag already exists (no partial releases)
    if not args.dry_run:
        existing_tags = []
        for a in releases_needed:
            tag_name = f"{a['package'].name}-v{a['new_version']}"
            if tag_exists(tag_name):
                existing_tags.append(tag_name)
        if existing_tags:
            if not json_output:
                print("\n❌ Release aborted: the following tags already exist.")
                for t in existing_tags:
                    print(f"   • {t}")
                print("\n   Remove them with:")
                for t in existing_tags:
                    print(f"   git tag -d {t}")
            return 1

    if not json_output:
        print(f"\n{'=' * 60}")
        print(f"{'[DRY RUN MODE]' if args.dry_run else '🚀 RELEASING'}")
        print(f"{'=' * 60}")

    tags_created = []
    released_items = []
    for analysis in releases_needed:
        tag = release_package(
            analysis,
            dry_run=args.dry_run,
            changelog=getattr(args, "changelog", False),
            quiet=json_output,
        )
        if tag:
            tags_created.append(tag)
            released_items.append({"analysis": analysis, "tag": tag})
        elif not args.dry_run:
            # Abort: one package failed (e.g. tag creation); no partial release
            if not json_output:
                print("\n❌ Release aborted after failure. Fix the issue and re-run.")
            return 1

    if not json_output:
        if tags_created and not args.dry_run:
            print(f"\n✅ Released {len(tags_created)} package(s)")
            if args.push:
                branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
                run_git(["push", "origin", branch])
                run_git(["push", "origin", "--tags"])
                print("  ✓ Pushed commits and tags to remote")
            else:
                print("\nTo push releases to remote:")
                print("  git push && git push --tags")
        elif tags_created:
            print(f"\n✅ [DRY RUN] Would release {len(tags_created)} package(s)")
            print("Run without --dry-run to apply changes.")

    if json_output:
        import json

        result = {
            "released": [
                {
                    "package": item["analysis"]["package"].name,
                    "old_version": item["analysis"]["current_version"],
                    "new_version": item["analysis"]["new_version"],
                    "bump_type": item["analysis"]["bump_type"].name,
                    "tag": item["tag"],
                    "commits": item["analysis"]["commits"],
                }
                for item in released_items
            ],
            "unchanged": [{"package": a["package"].name, "version": a["current_version"]} for a in analyses if not a["needs_release"]],
            "dry_run": args.dry_run,
        }
        print(json.dumps(result, indent=2))

    return 0


# =============================================================================
# SHOW COMMAND (Display Current Versions)
# =============================================================================


def cmd_show(args: argparse.Namespace) -> int:
    """Show current versions of all packages."""
    print("📦 Current Package Versions:")
    print("-" * 40)
    warnings = []
    for _, pkg in PACKAGES.items():
        version = get_current_version(pkg.pyproject_path)
        # Get highest tag (no version filter)
        highest_tag = get_last_tag_for_package(pkg.name, max_version=None)
        # Get matching tag (<= current version)
        matching_tag = get_last_tag_for_package(pkg.name, max_version=version)

        tag_info = f"(tag: {matching_tag})" if matching_tag else "(no tag)"
        print(f"  {pkg.name:25} {version:10} {tag_info}")

        # Warn if highest tag is different (indicates version mismatch)
        if highest_tag and highest_tag != matching_tag:
            warnings.append(f"  ⚠️  {pkg.name}: highest tag {highest_tag} > pyproject version {version}")

    if warnings:
        print("\n⚠️  Version/Tag Mismatches Detected:")
        for warning in warnings:
            print(warning)
        print("\n  Run 'python scripts/version.py sync' to fix these mismatches.")

    return 0


# =============================================================================
# SYNC COMMAND (Fix Version/Tag Mismatches)
# =============================================================================


def cmd_sync(args: argparse.Namespace) -> int:
    """Diagnose and fix version/tag mismatches."""
    print("🔍 Checking for version/tag mismatches...\n")

    mismatches = []
    for scope, pkg in PACKAGES.items():
        version = get_current_version(pkg.pyproject_path)
        highest_tag = get_last_tag_for_package(pkg.name, max_version=None)

        if not highest_tag:
            continue

        tag_version = parse_version_from_tag(highest_tag, pkg.name)
        current_version = parse_version_string(version)

        if tag_version and tag_version > current_version:
            tag_version_str = ".".join(map(str, tag_version))
            mismatches.append(
                {
                    "scope": scope,
                    "package": pkg,
                    "current_version": version,
                    "tag": highest_tag,
                    "tag_version": tag_version_str,
                }
            )

    if not mismatches:
        print("✅ No mismatches found. All packages are in sync with their tags.")
        return 0

    print("⚠️  Found version/tag mismatches:\n")
    print("-" * 70)
    for m in mismatches:
        print(f"  {m['package'].name}")
        print(f"    pyproject.toml:  {m['current_version']}")
        print(f"    highest tag:     {m['tag']} (v{m['tag_version']})")
        print()
    print("-" * 70)

    if args.to_tag:
        # Upgrade pyproject versions to match tags
        print("\n🔧 Upgrading pyproject versions to match tags...\n")

        if args.dry_run:
            print("[DRY RUN] Would update:")

        files_updated = []
        for m in mismatches:
            pkg = m["package"]
            old_version = m["current_version"]
            new_version = m["tag_version"]

            if args.dry_run:
                print(f"  • {pkg.pyproject_path}: {old_version} → {new_version}")
                if pkg.init_path:
                    print(f"  • {pkg.init_path}: {old_version} → {new_version}")
            else:
                if update_version_in_file(pkg.pyproject_path, old_version, new_version):
                    files_updated.append(pkg.pyproject_path)
                    print(f"  ✓ Updated {pkg.pyproject_path}: {old_version} → {new_version}")
                if pkg.init_path and update_version_in_file(pkg.init_path, old_version, new_version):
                    files_updated.append(pkg.init_path)
                    print(f"  ✓ Updated {pkg.init_path}")

        if not args.dry_run and files_updated:
            # Update lockfile
            lockfile_updated = run_uv_lock()
            if lockfile_updated:
                files_updated.append(Path("uv.lock"))

            # Commit changes
            if commit_with_retry("chore: sync versions to match existing tags", files_updated):
                print("\n✅ Versions synced to match tags.")
                print("   Run 'git push' to update remote.")
            else:
                print("\n❌ Failed to commit changes.")
                return 1

        return 0

    elif args.delete_tags:
        # Show commands to delete orphan tags
        print("\n🗑️  To delete orphan tags, run these commands:\n")
        print("# Delete local tags:")
        for m in mismatches:
            print(f"git tag -d {m['tag']}")

        print("\n# Delete remote tags:")
        for m in mismatches:
            print(f"git push origin --delete {m['tag']}")

        print("\n⚠️  Only delete tags if they represent failed/aborted releases!")
        print("   If packages were actually published, use --to-tag instead.")
        return 0

    else:
        # Show options
        print("\n📋 How to fix:\n")
        print("  Option A: Upgrade pyproject to match tags (if tags are valid releases)")
        print("    python scripts/version.py sync --to-tag [--dry-run]")
        print()
        print("  Option B: Delete orphan tags (if tags were from failed releases)")
        print("    python scripts/version.py sync --delete-tags")
        print()
        print("💡 TIP: Check if the tagged versions were actually published to PyPI")
        print("   before deciding which option to use.")

    return 0


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monorepo version management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple unified bump (all packages same version)
  %(prog)s bump patch              # 0.1.0 → 0.1.1
  %(prog)s bump minor              # 0.1.0 → 0.2.0
  %(prog)s bump --dry-run patch    # Preview only

  # Per-package semantic release
  %(prog)s release --dry-run       # Preview releases
  %(prog)s release                 # Release all with changes
  %(prog)s release --package core  # Release specific package

  # Show versions
  %(prog)s show

  # Fix version/tag mismatches
  %(prog)s sync                    # Diagnose mismatches
  %(prog)s sync --to-tag           # Upgrade pyproject to match tags
  %(prog)s sync --delete-tags      # Show commands to delete orphan tags
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Bump command
    bump_parser = subparsers.add_parser("bump", help="Bump all packages to same version")
    bump_parser.add_argument("bump_type", choices=["patch", "minor", "major"], help="Type of version bump")
    bump_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    bump_parser.add_argument("--no-tag", action="store_true", help="Don't create git tag")
    bump_parser.add_argument("--no-commit", action="store_true", help="Don't commit changes")
    bump_parser.set_defaults(func=cmd_bump)

    # Release command
    release_parser = subparsers.add_parser("release", help="Per-package semantic release")
    release_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    release_parser.add_argument("--no-require-clean", action="store_true", help="Skip dirty working directory check (default: enforced)")
    release_parser.add_argument("--push", action="store_true", help="Push commits and tags to remote after release")
    release_parser.add_argument(
        "--prerelease",
        nargs="?",
        const="dev",
        default=None,
        metavar="SUFFIX",
        help="Pre-release suffix (default: 'dev'). E.g., --prerelease, --prerelease rc",
    )
    release_parser.add_argument("--changelog", action="store_true", help="Generate/update CHANGELOG.md for each released package")
    release_parser.add_argument("--json", action="store_true", dest="json_output", help="Output results as JSON (for CI scripting)")
    release_parser.add_argument("--package", choices=list(PACKAGES.keys()), help="Release specific package only")
    release_parser.add_argument("--force-bump", choices=["patch", "minor", "major"], help="Force a specific bump type")
    release_parser.set_defaults(func=cmd_release)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show current versions")
    show_parser.set_defaults(func=cmd_show)

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Fix version/tag mismatches")
    sync_parser.add_argument("--to-tag", action="store_true", help="Upgrade pyproject versions to match highest tags")
    sync_parser.add_argument("--delete-tags", action="store_true", help="Show commands to delete orphan tags")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    sync_parser.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
