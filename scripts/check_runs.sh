#!/bin/bash
# Check status of all budget-injection training runs.
# Reports: running, completed, crashed, or not started.

BASE_DIR="/pscratch/sd/s/siddart2/budget-injection/prime-rl"
MAX_STEPS=400

# All expected runs
RUNS=(
    # Phase 1
    "baseline-seed42" "baseline-seed123"
    "inject-2048-seed42" "inject-2048-seed123"
    "inject-1024-seed42" "inject-1024-seed123"
    "inject-4096-seed42" "inject-4096-seed123"
    # Phase 2
    "inject-ratio-seed42" "inject-ratio-seed123"
    "inject-urgency-seed42" "inject-urgency-seed123"
    "inject-minimal-seed42" "inject-minimal-seed123"
    # Phase 3
    "inject-variable-seed42" "inject-variable-seed123"
    # Phase 4
    "baseline-06b-seed42"
    "inject-06b-seed42"
)

echo "=== Budget Injection Run Status ==="
echo "$(date)"
echo ""

running_jobs=$(squeue --me --noheader --format="%j" 2>/dev/null || echo "")

for run in "${RUNS[@]}"; do
    output_dir="${BASE_DIR}/outputs/${run}"

    # Check if running
    if echo "$running_jobs" | grep -q "bi-${run}"; then
        status="RUNNING"
    elif [ -d "$output_dir/run_default/broadcasts" ]; then
        # Find latest checkpoint
        latest_step=$(ls -d "$output_dir/run_default/broadcasts/step_"* 2>/dev/null | \
            sed 's/.*step_//' | sort -n | tail -1)
        if [ -n "$latest_step" ] && [ "$latest_step" -ge "$MAX_STEPS" ]; then
            status="COMPLETED (step $latest_step)"
        elif [ -n "$latest_step" ]; then
            status="CRASHED at step $latest_step (needs resume)"
        else
            status="STARTED but no checkpoints"
        fi
    else
        status="NOT STARTED"
    fi

    printf "%-30s %s\n" "$run" "$status"
done
