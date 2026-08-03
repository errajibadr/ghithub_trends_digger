#!/bin/bash
# Concatenate every file diff of a GitHub PR into a single output file.
# Mirrors concat_staged.sh / concat_since.sh, but the content comes from the
# PR's diff (via the gh CLI) instead of the working tree.
#
# Usage: ./scripts/concat_pr_diff.sh <pr> [output_file]
#        <pr>          required — PR number (e.g. 61), URL, or head branch name.
#                      Anything `gh pr diff` accepts.
#        output_file   optional — default: pr_<number>_diff.txt
#
# Examples:
#   ./scripts/concat_pr_diff.sh 61
#   ./scripts/concat_pr_diff.sh https://github.com/owner/repo/pull/61 review.txt
#
# Notes:
#   - Requires the gh CLI, authenticated (`gh auth status`).
#   - Works from any repo clone with a GitHub remote; the PR is resolved against
#     the current repo unless a full URL is given.

set -euo pipefail

PR="${1:-}"

if [ -z "$PR" ]; then
    echo "Usage: $0 <pr_number|pr_url|branch> [output_file]" >&2
    exit 1
fi

if ! command -v gh >/dev/null; then
    echo "gh CLI not found — install it first (brew install gh)." >&2
    exit 1
fi

# Resolve PR metadata once; also validates that the PR exists.
PR_JSON=$(gh pr view "$PR" --json number,title,url,baseRefName,headRefName)
PR_NUMBER=$(echo "$PR_JSON" | jq -r .number)
PR_TITLE=$(echo "$PR_JSON" | jq -r .title)
PR_URL=$(echo "$PR_JSON" | jq -r .url)
PR_BASE=$(echo "$PR_JSON" | jq -r .baseRefName)
PR_HEAD=$(echo "$PR_JSON" | jq -r .headRefName)

OUTPUT_FILE="${2:-pr_${PR_NUMBER}_diff.txt}"

CHANGED_FILES=$(gh pr diff "$PR" --name-only)

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files in PR #$PR_NUMBER."
    exit 0
fi

# Header, then the full unified diff (already one section per file, delimited
# by `diff --git a/... b/...` lines).
{
    echo "PR #$PR_NUMBER: $PR_TITLE"
    echo "$PR_URL"
    echo "$PR_HEAD -> $PR_BASE"
    echo "----"
    echo "Changed files:"
    echo "$CHANGED_FILES"
    echo "-------"
    echo ""
    gh pr diff "$PR"
} > "$OUTPUT_FILE"

echo "PR #$PR_NUMBER diff concatenated to: $OUTPUT_FILE"
echo "Files included:"
echo "$CHANGED_FILES"
