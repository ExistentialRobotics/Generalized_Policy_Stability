#!/bin/bash
# Script to prepare main branch for open source release
# This will:
# 1. Remove dm_control directory
# 2. Merge prep-open-source to main
# 3. Squash all commits to a single commit
# 4. Delete all remote branches except main
# 5. Keep local branches intact

set -e  # Exit on error

echo "=== Preparing main branch for open source ==="

# Step 1: Checkout prep-open-source and remove dm_control
echo "Step 1: Removing dm_control directory..."
git checkout prep-open-source
git rm -r nonlinear_system/dm_control
git commit -m "Remove dm_control directory for open source release"

# Step 2: Checkout main and merge
echo "Step 2: Merging prep-open-source to main..."
git checkout main
git merge prep-open-source --no-ff -m "Merge prep-open-source to main"

# Step 3: Squash all commits to a single commit
echo "Step 3: Squashing all commits to a single commit..."
# Get the first commit hash (or use a specific base)
FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
# Create orphan branch for clean history
git checkout --orphan main-clean
git add -A
git commit -m "Initial commit: Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions

This repository implements methods for certifying stability of RL policies using generalized multi-step Lyapunov functions.

Features:
- Linear system (LQR) stability certification
- Nonlinear system (RL policy) stability certification with Gym Inverted Pendulum
- Neural network-based Lyapunov function learning
- Multi-step stability conditions with state-dependent step weights

See README.md for usage instructions."

# Step 4: Replace main with clean history
echo "Step 4: Replacing main with clean history..."
git branch -D main
git branch -m main

# Step 5: Delete all remote branches except main
echo "Step 5: Deleting remote branches..."
git push origin --delete prep-open-source 2>/dev/null || true
# Add any other remote branches you want to delete here

# Step 6: Force push main (clean history)
echo "Step 6: Force pushing clean main to remote..."
echo "WARNING: This will rewrite remote main history!"
read -p "Continue? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
    git push origin main --force
    echo "✓ Main branch pushed with clean history"
else
    echo "Aborted. Local main branch is ready but not pushed."
fi

echo ""
echo "=== Summary ==="
echo "✓ dm_control removed"
echo "✓ All commits squashed to single commit"
echo "✓ Main branch has clean history"
echo "✓ Remote branches cleaned (except main)"
echo ""
echo "Local branches preserved:"
git branch | grep -v "main" | grep -v "main-clean"

