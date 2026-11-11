#!/usr/bin/env bash
# Run the lines of a grid file written by `nlicorrupt grid`, one after another.
# Comments and blank lines are skipped. Optional FIRST and LAST select a block of lines,
# so a grid can be split across GPUs or jobs of any scheduler.
#   bash scripts/run_grid.sh grids/encoders.txt          # all lines
#   bash scripts/run_grid.sh grids/encoders.txt 1 8      # lines 1-8
# A failed line does not stop the others; completed runs are skipped when run again.
set -uo pipefail
grid=${1:?usage: run_grid.sh GRID_FILE [FIRST LAST]}
mapfile -t cmds < <(grep -v '^\s*\(#\|$\)' "$grid")
first=${2:-1}
last=${3:-${#cmds[@]}}
failed=0
for (( i = first; i <= last && i <= ${#cmds[@]}; i++ )); do
    echo "=== [$i] ${cmds[$((i - 1))]}"
    if ! bash -c "${cmds[$((i - 1))]}"; then
        echo "=== [$i] FAILED" >&2
        failed=$((failed + 1))
    fi
done
echo "=== done: $failed failed"
[ "$failed" -eq 0 ]
