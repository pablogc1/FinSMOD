# -*- coding: utf-8 -*-
"""
INSMOD/FINSMOD Canary (Serpentine Test)
========================================
Validates the Python/Numba INSMOD/FINSMOD core before running the full C++ pipeline.
Uses the serpentine canary test with expected values: INSMOD=21, FINSMOD=37
"""
import sys
import numpy as np

from pipeline_core.sod_logic import (
    smod_pair_python,
    smod_pair_numba,
    TYPE_NAME
)

def format_smod_log(levels_data, total_score_unc, label="SMOD pair"):
    """
    Log for a SMOD pair (used by both INSMOD and FINSMOD):
      - shows which pair we are comparing (label),
      - shows paths with parentheses around canceled row/col components,
      - per level: #canceled, #uncanceled, 'Unc * level = ...',
      - at the bottom: total Σ (Unc * level).
    """
    lines = []
    lines.append(f"=== {label} ===")
    lines.append("--- SMOD Detailed Uncanceled Contribution Log ---\n")
    header = (
        f"{'Level':^5} | {'Side A (r,c)':^20} | "
        f"{'Side B (r,c)':^20} | {'Canc':^4} | {'Unc':^4} | {'Unc * level':^18}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for d in levels_data:
        lvl = d["level"]
        rA, cA = d["rA"], d["cA"]
        rB, cB = d["rB"], d["cB"]

        canc_count = d["canc_count"]
        unc_count  = d["uncanc_count"]
        contrib    = d["unc_contrib"]

        # Parenthesis around canceled row / col
        sA_r = f"({rA:>2})" if d["canc_Ar"] else f" {rA:>2} "
        sA_c = f"({cA:>2})" if d["canc_Ac"] else f" {cA:>2} "
        sB_r = f"({rB:>2})" if d["canc_Br"] else f" {rB:>2} "
        sB_c = f"({cB:>2})" if d["canc_Bc"] else f" {cB:>2} "

        sideA_str = f"({sA_r}, {sA_c})"
        sideB_str = f"({sB_r}, {sB_c})"

        lines.append(
            f"{lvl:^5} | {sideA_str:^20} | {sideB_str:^20} | "
            f"{canc_count:^4d} | {unc_count:^4d} | "
            f"{unc_count} * {lvl} = {contrib:.1f}"
        )

    lines.append("")
    lines.append(f"Total (Σ Unc * level) = {total_score_unc:.1f}")
    return "\n".join(lines)

def run_sod_canary_test():
    """
    Periodic serpentine canary:
      - The serpentine path is a cycle: after the last cell, it continues at the first cell.
      - We emulate "finite time" by taking a finite number of steps (TIME_UNITS),
        with 1 step per time unit.

    INSMOD: compare forward finite trajectories (A vs B).
    FINSMOD: compare reversed finite trajectories (A_rev vs B_rev), so Level 0 is the final cell.
    """
    print("=" * 80)
    print("             SERPENTINE CANARY TEST (INSMOD/FINSMOD)")
    print("=" * 80)
    
    size = 6
    TIME_UNITS = 10          # 10 time units
    STEPS = TIME_UNITS       # 1 step per time unit
    LENGTH = STEPS + 1       # positions include the start, so steps+1 points

    def gen_serpentine(s=6):
        path = []
        for r in range(s):
            cols = range(s) if r % 2 == 0 else range(s - 1, -1, -1)
            for c in cols:
                path.append((r, c))
        return np.array(path, dtype=np.int64)

    def cyclic_slice(full_path, start_idx, length):
        """
        Take 'length' points along the serpentine cycle starting at start_idx.
        """
        N = full_path.shape[0]
        idxs = (start_idx + np.arange(length)) % N
        return full_path[idxs]

    full_snake = gen_serpentine(size)
    Ncells = full_snake.shape[0]  # = size*size

    # Display the full serpentine path
    print(f"\n--- FULL SERPENTINE PATH (Size {size}x{size}, {Ncells} cells) ---")
    print("Complete path (row, col) sequence:")
    for i, (r, c) in enumerate(full_snake):
        print(f"  Index {i:2d}: ({r}, {c})")
    print()

    # Pick two source cells
    idx_A, idx_B = 5, 10
    start_A_rc, start_B_rc = full_snake[idx_A], full_snake[idx_B]
    print(f"Selected cells for testing:")
    print(f"  Cell A: Index {idx_A} = {start_A_rc}")
    print(f"  Cell B: Index {idx_B} = {start_B_rc}")

    # Finite "time-window" trajectories with wrap-around (periodic snake)
    traj_A = cyclic_slice(full_snake, idx_A, LENGTH)
    traj_B = cyclic_slice(full_snake, idx_B, LENGTH)

    # Reversed trajectories (final -> initial): Level 0 becomes the final cell
    traj_A_rev = traj_A[::-1].copy()
    traj_B_rev = traj_B[::-1].copy()

    # Compute INSMOD (forward) and FINSMOD (reversed)
    ti, si, term_i, li = smod_pair_python(traj_A, traj_B)
    ts, ss, term_s, ls = smod_pair_python(traj_A_rev, traj_B_rev)

    # Strong reverse checks: the reverses should be exact inverses
    check_A = np.array_equal(traj_A, traj_A_rev[::-1])
    check_B = np.array_equal(traj_B, traj_B_rev[::-1])

    # Print detailed report
    print(f"\n--- TEST PARAMETERS ---")
    print(f"Grid size: {size}x{size} (Ncells={Ncells})")
    print(f"Time units: {TIME_UNITS} | 1 step/unit => STEPS={STEPS} | Path length={LENGTH} points")
    print(f"\n--- REVERSIBILITY VERIFICATION ---")
    print(f"Check Reverse Identity (Path A): {check_A} (traj_A == traj_A_rev[::-1])")
    print(f"Check Reverse Identity (Path B): {check_B} (traj_B == traj_B_rev[::-1])")
    if not (check_A and check_B):
        print("  WARNING: Reversibility check failed!")
    else:
        print("  ✓ Reversibility verified: forward and reversed paths are exact inverses")
    print(f"\n--- FORWARD PATHS (INSMOD inputs) ---")
    print(f"Path A forward (len={len(traj_A)}): {traj_A.tolist()}")
    print(f"Path B forward (len={len(traj_B)}): {traj_B.tolist()}")
    print(f"\n--- REVERSED PATHS (FINSMOD inputs; Level 0 is final cell) ---")
    print(f"Path A reversed (len={len(traj_A_rev)}): {traj_A_rev.tolist()}")
    print(f"Path B reversed (len={len(traj_B_rev)}): {traj_B_rev.tolist()}")
    print(f"\nForward ends: A_end={traj_A[-1]} | B_end={traj_B[-1]}")
    print("Note: In FINSMOD (Reversed), Level 0 corresponds to A_end and B_end.")
    
    print("\n" + "="*80)
    print("INSMOD CALCULATION (Forward Paths)")
    print("="*80)
    print("Using forward trajectories: traj_A vs traj_B")
    print("This tests the SMOD logic function (smod_pair_python) with forward paths")
    print("as it will be used in the INSMOD calculation in the pipeline.")
    print(format_smod_log(li, si, label=f"INSMOD (Forward) A vs B - Result: {TYPE_NAME[ti]}"))
    
    print("\n" + "="*80)
    print("FINSMOD CALCULATION (Reversed Paths)")
    print("="*80)
    print("Using reversed trajectories: traj_A_rev vs traj_B_rev")
    print("This tests the SMOD logic function (smod_pair_python) with reversed paths")
    print("as it will be used in the FINSMOD calculation in the pipeline.")
    print(format_smod_log(ls, ss, label=f"FINSMOD (Reversed) A_rev vs B_rev - Result: {TYPE_NAME[ts]}"))

    # Numba validation
    ti_nb, si_nb = smod_pair_numba(traj_A, traj_B)
    ts_nb, ss_nb = smod_pair_numba(traj_A_rev, traj_B_rev)

    # Expected values
    expected_insmod_score = 21.0
    expected_finsmod_score = 37.0

    # Validation
    insmod_py_nb_match = (ti == ti_nb) and (abs(si - si_nb) < 1e-9)
    finsmod_py_nb_match = (ts == ts_nb) and (abs(ss - ss_nb) < 1e-9)
    insmod_expected_match = abs(si - expected_insmod_score) < 1e-9
    finsmod_expected_match = abs(ss - expected_finsmod_score) < 1e-9

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print("This validates that:")
    print("  1. The SMOD logic function (smod_pair_python) works correctly")
    print("  2. The Numba-accelerated version (smod_pair_numba) matches Python")
    print("  3. Both INSMOD (forward) and FINSMOD (reversed) produce expected results")
    print("  4. These are the SAME functions used in the pipeline calculations")
    print("="*80)
    print(f"\nINSMOD Validation (Forward Paths):")
    print(f"  - Python/Numba Match: {insmod_py_nb_match} (Python: {si:.1f}, Numba: {si_nb:.1f})")
    print(f"  - Expected Score Match: {insmod_expected_match} (Expected: {expected_insmod_score:.1f}, Got: {si:.1f})")
    print(f"  - Type: {TYPE_NAME[ti]} ({ti})")
    if insmod_py_nb_match and insmod_expected_match:
        print(f"  ✓ INSMOD calculation verified: SMOD logic works correctly with forward paths")
    
    print(f"\nFINSMOD Validation (Reversed Paths):")
    print(f"  - Python/Numba Match: {finsmod_py_nb_match} (Python: {ss:.1f}, Numba: {ss_nb:.1f})")
    print(f"  - Expected Score Match: {finsmod_expected_match} (Expected: {expected_finsmod_score:.1f}, Got: {ss:.1f})")
    print(f"  - Type: {TYPE_NAME[ts]} ({ts})")
    if finsmod_py_nb_match and finsmod_expected_match:
        print(f"  ✓ FINSMOD calculation verified: SMOD logic works correctly with reversed paths")

    if insmod_py_nb_match and finsmod_py_nb_match and insmod_expected_match and finsmod_expected_match:
        print("\n[SUCCESS] INSMOD/FINSMOD Canary Passed.")
        return 0
    else:
        print("\n[FAILURE] INSMOD/FINSMOD Canary Failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_sod_canary_test())

