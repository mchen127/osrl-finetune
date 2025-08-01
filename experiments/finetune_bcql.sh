#!/bin/bash

# ======================================================================================
# launch_finetune_bcql.sh
#
# Description:
#   This script launches parallel fine-tuning jobs for the BCQL algorithm.
#   It finds pretrained models based on a task name and a list of seeds,
#   and then runs train_bcql.py for a new range of seeds.
#
# Usage:
#   ./launch_finetune_bcql.sh --task <task_name> [options]
#
# Example:
#   ./launch_finetune_bcql.sh \
#       --task OfflineCarCircle-v0-cost-40 \
#       --pretrained_seeds "1 5 10" \
#       --new_seeds "21 22 23" \
#       --max_jobs 8
#
# Arguments:
#   --task <name>             (Required) The name of the task, e.g., 'OfflineCarCircle-v0-cost-40'.
#   --pretrained_seeds "s1 s2" (Optional) A space-separated string of seeds for the pretrained models.
#                               Default: "1 2 3".
#   --new_seeds "s1 s2"       (Optional) A space-separated string of seeds for the new fine-tuning run.
#                               Default: "21 22 23".
#   --max_jobs <num>          (Optional) The maximum number of parallel jobs to run.
#                               Default: 4.
# ======================================================================================

# --- Default Configuration ---
MAX_JOBS=8
TASK=""
PRETRAINED_SEEDS="1 2 3"
NEW_SEEDS="101 102 103"
BASE_LOG_DIR="/home/r14725021_mc/OSRL/logs"

# --- Argument Parsing ---
# This loop processes command-line arguments.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift ;;
        --pretrained_seeds) PRETRAINED_SEEDS="$2"; shift ;;
        --new_seeds) NEW_SEEDS="$2"; shift ;;
        --max_jobs) MAX_JOBS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate Required Arguments ---
if [ -z "$TASK" ]; then
    echo "Error: --task is a required argument."
    echo "Usage: ./launch_finetune_bcql.sh --task <task_name> [--pretrained_seeds \"1 5 10\"] [--new_seeds \"21 22 23\"] [--max_jobs 4]"
    exit 1
fi

# --- Extract Cost Limit from Task Name ---
# This uses 'grep' with a Perl-compatible regular expression (PCRE) to find
# the number immediately following '-cost-'.
COST_LIMIT=$(echo "$TASK" | grep -oP '(?<=-cost-)\d+')

if [ -z "$COST_LIMIT" ]; then
    echo "Error: Could not extract cost limit from task name: $TASK"
    echo "The task name must contain a pattern like '-cost-XX'."
    exit 1
fi

# --- Print Job Summary ---
echo "=========================================="
echo "Starting Fine-tuning Job Launcher"
echo "=========================================="
echo "Task:                 $TASK"
echo "Extracted Cost Limit: $COST_LIMIT"
echo "Pretrained Seeds:     $PRETRAINED_SEEDS"
echo "New Fine-tuning Seeds: $NEW_SEEDS"
echo "Max Parallel Jobs:    $MAX_JOBS"
echo "------------------------------------------"

# --- Main Execution Logic ---
# Loop over each seed provided in the --pretrained_seeds argument.
for pretrained_seed in $PRETRAINED_SEEDS; do
    
    # Construct the search pattern for the pretrained model directories.
    # The wildcard '*' handles the random string at the end of the folder name.
    search_pattern="${BASE_LOG_DIR}/${TASK}/BCQL_cost${COST_LIMIT}_seed${pretrained_seed}*/BCQL_cost${COST_LIMIT}_seed${pretrained_seed}*"
    
    # Find all directories that match the pattern.
    # 'shopt -s nullglob' ensures that if no matches are found, the array will be empty.
    shopt -s nullglob
    pretrained_paths=($search_pattern)
    shopt -u nullglob

    # Check if any pretrained models were found for the current seed.
    if [ ${#pretrained_paths[@]} -eq 0 ]; then
        echo "Warning: No pretrained model paths found for seed ${pretrained_seed} with pattern:"
        echo "  ${search_pattern}"
        continue # Skip to the next pretrained_seed
    fi

    # Loop over every directory that was found for the given pretrained_seed.
    for pretrained_path in "${pretrained_paths[@]}"; do

        # figure out high/low/all from the dirname
        name=$(basename "$pretrained_path")
        if [[ "$name" =~ trajcosthigh ]]; then
            cost_type="high"
        elif [[ "$name" =~ trajcostlow ]]; then
            cost_type="low"
        else
            cost_type="all"
        fi

        # Loop over each seed for the new fine-tuning run.
        for new_seed in $NEW_SEEDS; do
        
            # --- Job Concurrency Control ---
            # Check the number of currently running background jobs.
            # If it's greater than or equal to MAX_JOBS, wait for any one job to finish.
            # The 'wait -n' command is efficient as it waits for the next job to complete.
            while (($(jobs -p | wc -l) >= MAX_JOBS)); do
                wait -n
            done

            echo "Launching Job:"
            echo "  - Pretrained Model: $(basename "$pretrained_path")"
            echo "  - New Seed:         $new_seed"
            
            # Launch the Python training script in the background using '&'.
            # All output from this command will be piped to a log file for organization.
            (
              python -m train.train_bcql \
                --device "cuda" \
                --pretrain_seed "$pretrained_seed" \
                --finetune_seed "$new_seed" \
                --cost_limit "$COST_LIMIT" \
                --pretrained_model "$pretrained_path" \
                --trajectory_cost "$cost_type" \
                --threads 8 \
            ) &> "logs-exp/finetune_${TASK}_pretrained-seed-${pretrained_seed}_new-seed-${new_seed}_trajectory-cost-${cost_type}.log" &

            # A small delay to prevent all processes from starting at the exact same moment.
            sleep 1
        done
    done
done

# --- Final Wait ---
# Wait for all remaining background jobs to complete before exiting the script.
echo "------------------------------------------"
echo "All jobs have been launched. Waiting for remaining jobs to finish..."
wait
echo "=========================================="
echo "All fine-tuning jobs are complete."
echo "=========================================="
