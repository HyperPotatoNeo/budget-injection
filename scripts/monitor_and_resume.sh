#!/bin/bash
# Monitor all runs and resubmit any that crashed.
# Run periodically (every ~2 hours) to keep experiments going.
# Usage: bash scripts/monitor_and_resume.sh [--phase 1|2|3|4|all]

set -e

PHASE="${1:-all}"
BASE_DIR="/pscratch/sd/s/siddart2/budget-injection/prime-rl"
MAX_STEPS=400

# Define runs per phase
declare -A PHASE_RUNS
PHASE_RUNS[1]="baseline:configs/baseline.toml inject-2048:configs/inject_2048.toml inject-1024:configs/inject_1024.toml inject-4096:configs/inject_4096.toml"
PHASE_RUNS[2]="inject-ratio:configs/inject_ratio.toml inject-urgency:configs/inject_urgency.toml inject-minimal:configs/inject_minimal.toml"
PHASE_RUNS[3]="inject-variable:configs/inject_variable.toml"
PHASE_RUNS[4]="baseline-06b:configs/baseline_06b.toml inject-06b:configs/inject_06b.toml"

get_seeds() {
    local name="$1"
    if [[ "$name" == *"06b"* ]]; then
        echo "42"  # 0.6B runs: 1 seed only
    else
        echo "42 123"  # 4B runs: 2 seeds
    fi
}

running_jobs=$(squeue --me --noheader --format="%j" 2>/dev/null || echo "")
resubmitted=0

for phase_num in 1 2 3 4; do
    if [ "$PHASE" != "all" ] && [ "$PHASE" != "$phase_num" ]; then
        continue
    fi

    for entry in ${PHASE_RUNS[$phase_num]}; do
        name="${entry%%:*}"
        config="${entry#*:}"

        for seed in $(get_seeds "$name"); do
            run="${name}-seed${seed}"

            # Skip if running
            if echo "$running_jobs" | grep -q "bi-${run}"; then
                continue
            fi

            output_dir="${BASE_DIR}/outputs/${run}"

            # Check if completed
            if [ -d "$output_dir/run_default/broadcasts" ]; then
                latest_step=$(ls -d "$output_dir/run_default/broadcasts/step_"* 2>/dev/null | \
                    sed 's/.*step_//' | sort -n | tail -1)
                if [ -n "$latest_step" ] && [ "$latest_step" -ge "$MAX_STEPS" ]; then
                    continue  # Completed
                fi
            fi

            # Not running and not completed -> resubmit
            echo "Resubmitting: $run (config=$config, seed=$seed)"
            sbatch --job-name="bi-${run}" scripts/launch.sh "$config" --seed "$seed"
            resubmitted=$((resubmitted + 1))
            sleep 1
        done
    done
done

echo "Resubmitted $resubmitted jobs."
