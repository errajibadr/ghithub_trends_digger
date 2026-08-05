#!/bin/bash
# Concatenate the diff between two remote branches into a single output file.
# This provides the equivalent of a pull-request diff using only Git.
#
# Usage: ./scripts/concat_pr_diff.sh <source_branch> <base_branch> [output_file]
#        source_branch  required — branch containing the proposed changes.
#        base_branch    required — branch the changes will be merged into.
#        output_file    optional — default: pr_<source_branch>_diff.txt
#
# Examples:
#   ./scripts/concat_pr_diff.sh feature/my-change main
#   ./scripts/concat_pr_diff.sh feature/my-change develop review.txt
#
# Notes:
#   - Requires only Git and a remote named `origin`.
#   - Both branch names are resolved on `origin`; do not prefix them with
#     `origin/`.

set -euo pipefail

SOURCE_BRANCH="${1:-}"
BASE_BRANCH="${2:-}"

if [ -z "$SOURCE_BRANCH" ] || [ -z "$BASE_BRANCH" ]; then
    echo "Usage: $0 <source_branch> <base_branch> [output_file]" >&2
    exit 1
fi

if ! git check-ref-format --branch "$SOURCE_BRANCH" >/dev/null 2>&1; then
    echo "Invalid source branch name: $SOURCE_BRANCH" >&2
    exit 1
fi

if ! git check-ref-format --branch "$BASE_BRANCH" >/dev/null 2>&1; then
    echo "Invalid base branch name: $BASE_BRANCH" >&2
    exit 1
fi

SAFE_SOURCE_BRANCH=${SOURCE_BRANCH//\//_}
OUTPUT_FILE="${3:-pr_${SAFE_SOURCE_BRANCH}_diff.txt}"

echo "Fetching '$SOURCE_BRANCH' and '$BASE_BRANCH' from origin..."
git fetch origin \
    "refs/heads/${SOURCE_BRANCH}:refs/remotes/origin/${SOURCE_BRANCH}" \
    "refs/heads/${BASE_BRANCH}:refs/remotes/origin/${BASE_BRANCH}"

SOURCE_REF="refs/remotes/origin/${SOURCE_BRANCH}"
BASE_REF="refs/remotes/origin/${BASE_BRANCH}"
CHANGED_FILES=$(git diff --name-only "${BASE_REF}...${SOURCE_REF}")

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files between '$SOURCE_BRANCH' and '$BASE_BRANCH'."
    exit 0
fi

# The three-dot range compares the source branch with its merge base against
# the base branch, matching the usual pull-request diff semantics.
{
    echo "Branch diff: $SOURCE_BRANCH -> $BASE_BRANCH"
    echo "----"
    echo "Changed files:"
    echo "$CHANGED_FILES"
    echo "-------"
    echo ""
    git diff "${BASE_REF}...${SOURCE_REF}"
} > "$OUTPUT_FILE"

echo "Branch diff concatenated to: $OUTPUT_FILE"
echo "Files included:"
echo "$CHANGED_FILES"
