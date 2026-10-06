#!/bin/bash
set -euo pipefail
task_scratch=/media/carruthers/mnt3/codex/scratch/docs-completion-20261005
sed -n '/^received_bytes() {/,/^}/p; /^human() {/p; /^downloaded_column() {/,/^}/p' packaging/measure_clone_forms.sh > "$task_scratch/clone_measure_functions.sh"
source "$task_scratch/clone_measure_functions.sh"
printf '%-14s %12s %12s %12s\n' form downloaded .git worktree
for form in full depth1 depth1-filter light; do
    destination="$task_scratch/clone-measure-20261005-r1/$form"
    received=$(received_bytes "$task_scratch/clone-measure-20261005-r1/$form.err")
    git_bytes=$(du -sb "$destination/.git" | cut -f1)
    tree_bytes=$(du -sb --exclude=.git "$destination" | cut -f1)
    printf '%-14s %12s %12s %12s\n' "$form" "$(downloaded_column "$received" "$form")" "$(human "$git_bytes")" "$(human "$tree_bytes")"
done
