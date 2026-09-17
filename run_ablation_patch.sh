#!/bin/bash
# =============================================================================
# Ablation patch: loss-term ablations (A) + sparsity robustness (B, our variants)
# Runs after the main pipeline (run_all.sh). New tags all contain "_std" so
# step6/step7 pick them up automatically (CURRENT_TAGS=['_std']).
#
# A. Loss-term ablations of L = 0.5*L_coarse + 0.2*L_recon + 0.3*L_denoise:
#    coarse_only (w_recon=0, w_denoise=0) / recon_only (w_coarse=0, w_denoise=0)
#    / no_denoise (w_denoise=0)
# B. Simulated data dropout 0.5/0.7/0.9 on: full moe / concat / pathway_mlp
#    (dropout rate is encoded in suffix_tag because checkpoint dirs and obsm
#    keys are isolated by tag only)
# Usage:  cd <project_root> && nohup bash run_ablation_patch.sh &
# =============================================================================
source /opt/miniconda3/etc/profile.d/conda.sh
set -u
cd "$(dirname "$0")"
mkdir -p logs
FAILED=()
R=enhanced_results/results

if [ ! -e "${R}/step4-5_add_special_harmony.h5ad" ]; then
    echo "ERROR: missing ${R}/step4-5_add_special_harmony.h5ad (run the main pipeline first)"
    exit 1
fi

S2=step5-2_deepfusion_multitoken.py
S3=step5-3_pathway_mlp_control.py

run_task () {  # run_task <name> <script> <args...>
    local name="$1" script="$2"; shift 2
    echo ""
    echo ">> [$(date '+%H:%M:%S')] ${name} | args: $*"
    local t0=$SECONDS
    set +u; conda activate deepfusion; set -u
    python -u "${script}" "$@"
    local rc=$?
    set +u; conda deactivate; set -u
    if [ ${rc} -eq 0 ]; then
        echo "OK [$(date '+%H:%M:%S')] ${name} done | elapsed $(( (SECONDS-t0)/60 )) min"
    else
        echo "FAIL ${name} (exit=${rc}), aborting"; FAILED+=("${name}"); exit 1
    fi
}

echo "=== Ablation patch (A: loss terms, B: dropout robustness) | start: $(date '+%F %T') ==="

echo ""
echo "===== A. Loss-term ablations (seed 42) ====="
run_task coarse_only_std ${S2} --targets 40 --fusion_mode moe \
    --w_recon 0 --w_denoise 0 --suffix_tag std_coarse_only
run_task recon_only_std  ${S2} --targets 40 --fusion_mode moe \
    --w_coarse 0 --w_denoise 0 --suffix_tag std_recon_only
run_task no_denoise_std  ${S2} --targets 40 --fusion_mode moe \
    --w_denoise 0 --suffix_tag std_no_denoise

echo ""
echo "===== B. Sparsity robustness (data dropout 0.5/0.7/0.9, seed 42) ====="
for DD in 50 70 90; do
    RATE=$(echo "${DD}" | awk '{printf "%.1f", $1/100}')
    run_task full_dd${DD}    ${S2} --targets 40 --fusion_mode moe \
        --data_dropout ${RATE} --suffix_tag std_dd${DD}
    run_task concat_dd${DD}  ${S2} --targets 40 --fusion_mode concat \
        --data_dropout ${RATE} --suffix_tag std_concat_dd${DD}
    run_task pathmlp_dd${DD} ${S3} --targets 40 \
        --data_dropout ${RATE} --suffix_tag pathmlp_std_dd${DD}
done

echo ""
echo "===== Evaluation (incremental; new _std* tags auto-included) ====="
set +u; conda activate scib; set -u
python -u step6_scib_evaluate.py || { echo "FAIL step6"; exit 1; }
set +u; conda deactivate; conda activate deepfusion; set -u
python -u step7_evaluate_embeddings_unsupervised.py || { echo "FAIL step7"; exit 1; }
set +u; conda deactivate; set -u

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DONE | end: $(date '+%F %T')"
    echo "Reminder: freeze step6/7 results to reference/ immediately"
else
    echo "FAILED tasks: ${FAILED[*]}"
fi
