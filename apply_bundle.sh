#!/bin/bash
# Update a repository, or rebuild one from a full bundle and its increments.
set -euo pipefail
# Do not let an inherited checkout or shell search path redirect Git operations.
git_local_env=$(git rev-parse --local-env-vars)
unset CDPATH GIT_NAMESPACE $git_local_env

usage() {
    cat <<EOF
Usage: $0 [--update] [BUNDLE_PATH [REPOSITORY_PATH]]
       $0 --rebuild [BUNDLE_DIR [REPOSITORY_PATH]]

Update defaults: highest numbered bundle/project_small_N.bundle, bundle/project.
Rebuild defaults: bundle/project.bundle + bundle/project_small*.bundle, bundle/project.
Rebuild orders increments by Git prerequisites and preserves an existing destination
in a sibling .backup.* directory. All updates must be fast-forwards.
EOF
}

fail() { echo "$*" >&2; exit 1; }

# Existing creation commands advertise either main or only HEAD.
bundle_ref() {
    local refs
    refs=$(git bundle list-heads "$1") || return 1
    refs+=$'\n'
    if [[ "$refs" == *" refs/heads/main"$'\n'* ]]; then
        echo refs/heads/main
    elif [[ "$refs" == *" HEAD"$'\n'* ]]; then
        echo HEAD
    else
        fail "Bundle advertises neither main nor HEAD: $1"
    fi
}

apply_bundle() {
    local repo=$1 bundle=$2 ref
    ref=$(bundle_ref "$bundle") || return 1
    git -C "$repo" bundle verify "$bundle"
    git -C "$repo" fetch --no-tags "$bundle" "$ref"
    git -C "$repo" merge --ff-only FETCH_HEAD
}

mode=update
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --rebuild) mode=rebuild; shift ;;
    --update) shift ;;
    -*) usage >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || { usage >&2; exit 2; }
shopt -s nullglob
bundle_dir=${BUNDLE_DIR:-bundle}

if [[ "$mode" == update ]]; then
    bundle_path=${1:-}
    repo_path=${2:-$bundle_dir/project}
    if [[ -z "$bundle_path" ]]; then
        # Sort the numeric suffix, not the filename (_10 must follow _9).
        bundle_path=$(
            for path in "$bundle_dir"/project_small_*.bundle; do
                index=${path##*/}
                index=${index#project_small_}
                index=${index%.bundle}
                [[ "$index" =~ ^[0-9]+$ ]] || continue
                printf '%s\t%s\n' "$index" "$path"
            done | sort -n | tail -n 1 | cut -f 2-
        )
        [[ -n "$bundle_path" ]] || fail "No numbered $bundle_dir/project_small_N.bundle found; pass a bundle path explicitly."
    fi
    [[ -f "$bundle_path" ]] || fail "Bundle not found: $bundle_path"
    [[ "$bundle_path" == /* ]] || bundle_path=$PWD/$bundle_path
    [[ -e "$repo_path/.git" ]] || fail "Not a Git repository root: $repo_path"
    branch=$(git -C "$repo_path" symbolic-ref --quiet --short HEAD) || fail "Repository is in detached HEAD state: $repo_path"
    [[ -z "$(git -C "$repo_path" status --porcelain --untracked-files=all)" ]] || fail "Repository has uncommitted changes: $repo_path"
    apply_bundle "$repo_path" "$bundle_path"
    echo "Updated $branch to $(git -C "$repo_path" rev-parse --short HEAD)."
    exit 0
fi

bundle_dir=${1:-$bundle_dir}
repo_path=${2:-$bundle_dir/project}
while [[ "$repo_path" == */ ]]; do repo_path=${repo_path%/}; done
[[ -n "$repo_path" ]] || fail "Use a named destination directory."
bundle_dir=$(cd "$bundle_dir" && pwd -P) || fail "Bundle directory not found: $bundle_dir"
[[ -f "$bundle_dir/project.bundle" ]] || fail "Full bundle not found: $bundle_dir/project.bundle"
[[ ! -L "$repo_path" ]] || fail "Rebuild destination must not be a symlink: $repo_path"
[[ ! -e "$repo_path" || -d "$repo_path" ]] || fail "Rebuild destination is not a directory: $repo_path"
[[ ! -f "$repo_path/.git" ]] || fail "Cannot replace a linked Git worktree: $repo_path"
[[ ! -d "$repo_path/.git/worktrees" ]] || fail "Cannot replace a repository with registered worktrees: $repo_path"
# Resolve the parent so temporary output and backup stay on the same filesystem.
repo_path=$(cd "$(dirname "$repo_path")" && printf '%s/%s' "$(pwd -P)" "$(basename "$repo_path")")
case "$(basename "$repo_path")" in .|..) fail "Use a named destination directory." ;; esac
case "$bundle_dir/" in "$repo_path/"*) fail "Destination must not contain the input bundles." ;; esac

stage=$(mktemp -d "${repo_path}.rebuild.XXXXXX")
backup=
cleanup() {
    if [[ -n "${backup:-}" && ! -e "$repo_path" && -d "$backup/project" ]]; then
        mv -- "$backup/project" "$repo_path" || {
            echo "Restore failed. Original: $backup/project; rebuilt repository: $stage/project" >&2
            return 1
        }
    fi
    rm -rf -- "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
git init -q "$stage/project"
git -C "$stage/project" symbolic-ref HEAD refs/heads/main
apply_bundle "$stage/project" "$bundle_dir/project.bundle"
pending=("$bundle_dir"/project_small*.bundle)
# ponytail: repeated passes suit small bundle sets; use a dependency graph for thousands.
while [[ ${#pending[@]} -gt 0 ]]; do
    remaining=()
    progress=0
    for bundle_path in "${pending[@]}"; do
        if git -C "$stage/project" bundle verify "$bundle_path" >/dev/null 2>&1; then
            apply_bundle "$stage/project" "$bundle_path"
            progress=1
        else
            remaining+=("$bundle_path")
        fi
    done
    if [[ "$progress" == 0 ]]; then
        printf 'Cannot apply bundle (missing prerequisites or invalid data): %s\n' "${remaining[@]}" >&2
        exit 1
    fi
    # Bash 3 treats an empty array as unset under nounset.
    pending=(${remaining[@]+"${remaining[@]}"})
done

if [[ -e "$repo_path" ]]; then
    backup=$(mktemp -d "${repo_path}.backup.XXXXXX")
    echo "Preserving previous directory at $backup/project"
    mv -- "$repo_path" "$backup/project"
    mv -- "$stage/project" "$repo_path"
else
    mv -- "$stage/project" "$repo_path"
fi
echo "Rebuilt $repo_path at $(git -C "$repo_path" rev-parse --short HEAD)."
