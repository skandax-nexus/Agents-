#!/usr/bin/env bash
# Move the voicecore/ directory out of the Agents- repository into its own repository,
# keeping its commit history.  Run from the Agents- checkout, after creating an empty
# GitHub repository (e.g. https://github.com/<you>/voicecore, no README).
#
#   ./voicecore/scripts/split_to_own_repo.sh git@github.com:<you>/voicecore.git
set -euo pipefail
REMOTE="${1:?usage: split_to_own_repo.sh <new-remote-url>}"
BRANCH="${2:-ccr-1934b12d-83vo2i}"
cd "$(git rev-parse --show-toplevel)"
git fetch origin "$BRANCH" >/dev/null 2>&1 || true
SPLIT=$(git subtree split --prefix=voicecore "$BRANCH")
echo "split commit: $SPLIT"
git push "$REMOTE" "$SPLIT:refs/heads/main"
echo "pushed voicecore history to $REMOTE (branch main)"
echo "clone it with: git clone $REMOTE"
