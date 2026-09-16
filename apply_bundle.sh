#!/bin/sh
set -eu

usage() {
    echo "Usage: $0 BUNDLE_PATH REPOSITORY_PATH" >&2
    exit 2
}

[ "$#" -eq 2 ] || usage

bundle_path=$1
repo_path=$2

[ -f "$bundle_path" ] || {
    echo "Bundle not found: $bundle_path" >&2
    exit 1
}

case "$bundle_path" in
    /*) ;;
    *) bundle_path=$PWD/$bundle_path ;;
esac

git -C "$repo_path" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Not a Git repository: $repo_path" >&2
    exit 1
}

repo_path=$(git -C "$repo_path" rev-parse --show-toplevel)
branch=$(git -C "$repo_path" symbolic-ref --quiet --short HEAD) || {
    echo "Repository is in detached HEAD state: $repo_path" >&2
    exit 1
}

[ -z "$(git -C "$repo_path" status --porcelain)" ] || {
    echo "Repository has uncommitted changes: $repo_path" >&2
    exit 1
}

git -C "$repo_path" bundle verify "$bundle_path"
git -C "$repo_path" fetch "$bundle_path" refs/heads/main
git -C "$repo_path" merge --ff-only FETCH_HEAD

echo "Updated $branch to $(git -C "$repo_path" rev-parse --short HEAD)."
