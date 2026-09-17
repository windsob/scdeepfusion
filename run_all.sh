#!/bin/bash
# =============================================================================
# DeepFusion standard pipeline (ifnb flagship dataset, batch=donor) — from step5
# Scope: teacher (5-0/5-1) -> model + ablations (5-2/5-3) -> evaluation (6/7)
#        -> pathway-scoring baselines (pb)
# The upstream baseline chain (step0-4) is run manually beforehand; this script
# only checks that its input artifact exists.
# Discipline: this .sh only schedules and skips existing artifacts; all model
# configurations live in the code defaults.
# Performance: every step has a built-in PerfRecorder writing to
# enhanced_results/perf_report.jsonl
# Usage:  cd <project_root> && nohup bash run_all.sh &
# =============================================================================
source /opt/miniconda3/etc/profile.d/conda.sh
set -u
cd "$(dirname "$0")"
mkdir -p logs enhanced_results/results
FAILED=()
R=enhanced_results/results

# ---------- Preflight: upstream artifacts (step0-4) are produced manually ----------
if [ ! -e "${R}/step4_add_scGPT.h5ad" ]; then
    echo "ERROR: missing upstream input ${R}/step4_add_scGPT.h5ad"
    echo "       Run the step0-4 baseline chain first, then re-run this script."
    exit 1
fi

run_step () {  # run_step <name> <env> <artifact> <cmd...>
    local name="$1" env="$2" artifact="$3"; shift 3
    if [ -n "${artifact}" ] && [ -e "${artifact}" ]; then
        echo "SKIP ${name}: artifact exists (${artifact})"; return 0
    fi
    echo ""
    echo ">> [$(date '+%H:%M:%S')] ${name} (env=${env})"
    local t0=$SECONDS
    if [ "${env}" != "none" ]; then set +u; conda activate "${env}"; set -u; fi
    "$@"
    local rc=$?
    if [ "${env}" != "none" ]; then set +u; conda deactivate; set -u; fi
    if [ ${rc} -eq 0 ]; then
        echo "OK [$(date '+%H:%M:%S')] ${name} done | elapsed $(( (SECONDS-t0)/60 )) min"
    else
        echo "FAIL ${name} (exit=${rc}), aborting"; FAILED+=("${name}"); exit 1
    fi
}

echo "=== Standard pipeline (from step5) | start: $(date '+%Y-%m-%d %H:%M:%S') ==="

# ---------- A. HVG subset + teacher (special harmony) ----------
run_step step5-0_hvg scib  ${R}/step4-4_filter_HVG3000.h5ad \
    python -u step5-0_create_HVGsubset.py
run_step step5-1_teacher scib ${R}/step4-5_add_special_harmony.h5ad \
    python -u step5-1_harmony_teacher.py

# ---------- B. DeepFusion model + ablations (deepfusion env; tag-level resume in code) ----------
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

echo ""
echo "===== B. Model and ablations (tag scheme: _std*) ====="
# Flagship comparison: full moe vs pathway_mlp, 3 seeds each
run_task full_std        ${S2} --targets 40 --fusion_mode moe   --suffix_tag std
run_task full_std_s43    ${S2} --targets 40 --fusion_mode moe   --seed 43 --suffix_tag std_s43
run_task full_std_s44    ${S2} --targets 40 --fusion_mode moe   --seed 44 --suffix_tag std_s44
run_task pathmlp_std     ${S3} --targets 40                     --suffix_tag pathmlp_std
run_task pathmlp_std_s43 ${S3} --targets 40                     --seed 43 --suffix_tag pathmlp_std_s43
run_task pathmlp_std_s44 ${S3} --targets 40                     --seed 44 --suffix_tag pathmlp_std_s44
# Ablations (single seed 42)
run_task concat_std      ${S2} --targets 40 --fusion_mode concat    --suffix_tag std_concat
run_task no_module_std   ${S2} --targets 40 --fusion_mode no_module --suffix_tag std_no_module
run_task no_coarse_std   ${S2} --targets 40 --w_coarse 0            --suffix_tag std_no_coarse
run_task distill_std     ${S2} --targets 40 --objective distill     --suffix_tag std_distill

# ---------- C. Evaluation ----------
run_step step6_scib scib "" \
    python -u step6_scib_evaluate.py
run_step step7_unsupervised deepfusion "" \
    python -u step7_evaluate_embeddings_unsupervised.py

# ---------- D. Pathway-scoring baselines (AUCell/GSVA/UCell + MLP readout control) ----------
# Depends on the step5-2 data cache (preprocessed_data.h5ad / valid_modules_cache.pkl),
# so it must run after phase B.
echo ""
echo "===== D. Pathway-scoring baselines ====="
run_step pb_step1 none pathway_baselines/results/pb_scores.h5ad \
    pathway_baselines/.venv/bin/python -u pathway_baselines/pb_step1_compute_scores.py
run_step pb_step8 deepfusion "" \
    python -u pathway_baselines/pb_step8_mlp_readout.py
# pb_step6/7 run AFTER pb_step8 so the *_mlp readout embeddings are evaluated too
run_step pb_step6 scib "" \
    python -u pathway_baselines/pb_step6_scib.py
run_step pb_step7 deepfusion "" \
    python -u pathway_baselines/pb_step7_unsupervised.py

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DONE | end: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Performance summary: python perf_utils.py --csv enhanced_results/perf_report.csv"
    echo "Reminder: freeze step6/7 results to reference/ immediately (live CSVs get overwritten)"
else
    echo "FAILED tasks: ${FAILED[*]}"
fi
