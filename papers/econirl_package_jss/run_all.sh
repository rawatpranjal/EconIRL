#!/usr/bin/env bash
# Reproduce every numbered listing, figure, and table in the paper.
# Run from the repository root: bash papers/econirl_package_jss/run_all.sh
# Writes a manifest of every output to manifest.csv at the same path.

set -e

DIR="papers/econirl_package_jss"
SNIP="$DIR/code_snippets"
MANIFEST="$DIR/manifest.csv"

echo "script,exit_code,duration_seconds" > "$MANIFEST"

run() {
  local script="$1"
  local t0=$(date +%s)
  python "$script"
  local rc=$?
  local t1=$(date +%s)
  echo "$script,$rc,$((t1 - t0))" >> "$MANIFEST"
}

echo "[1/3] Pre-estimation diagnostics"
run "$SNIP/diagnostics_rust_bus.py"
run "$SNIP/diagnostics_keane_wolpin.py"

echo "[2/3] Listings and figures"
run "$SNIP/teaser_nfxp_rust.py"
run "$SNIP/listing2_nfxp_example.py"
run "$SNIP/listing2b_bootstrap.py"
run "$SNIP/fig1_rust_bus_ccp.py"
run "$SNIP/fig2_rust_bus_value.py"
run "$SNIP/listing3_mce_irl_example.py"
run "$SNIP/fig3_mce_irl_reward.py"
run "$SNIP/listing4_nnes_example.py"
run "$SNIP/fig4_keane_wolpin_policy.py"

echo "[3/3] Cross-estimator benchmark, transfer tests, summary plot"
run "examples/rust-bus-engine/benchmark_all_estimators.py"
run "$SNIP/transfer_mce_irl.py"
run "$SNIP/transfer_gladius.py"
run "$SNIP/transfer_airl.py"
run "$SNIP/transfer_iq_learn.py"
run "$SNIP/transfer_f_irl.py"
run "$SNIP/fig5_time_accuracy.py"
run "$SNIP/table1_estimator_taxonomy.py"
run "$SNIP/table2_library_comparison.py"

echo "GPU listing is optional and runs only when a GPU is available."
if python -c "import jax; assert jax.default_backend() == 'gpu'" 2>/dev/null; then
  run "$SNIP/listing4b_nnes_gpu.py"
fi

echo
echo "Done. See $MANIFEST."
