# -*- coding: utf-8 -*-
"""
SMOD Logic Core
==============
Single source of truth for SMOD (Stability-Based):
- Reference (pure Python) versions for correctness checks and logs
- Numba-accelerated versions used by workers
"""

import numpy as np
from numba import njit

# Type codes
TYPE_T = 0   # Terminated
TYPE_UC = 1  # Unterminated with Cancels
TYPE_UU = 2  # Unterminated Uncanceled

TYPE_NAME = {
    TYPE_T:  "T",
    TYPE_UC: "UC",
    TYPE_UU: "UU",
}

# ------------------------------------------------------------------------------
# Reference (pure Python) - canonical semantics for validation & logs
# ------------------------------------------------------------------------------

def smod_pair_python(path_a, path_b):
    """
    Pure Python SMOD core for one pair of paths.
    Returns:
        type_code       : int (0=T, 1=UC, 2=UU)
        total_score_unc : float
        terminated      : bool
        levels_data     : list of dicts
    """
    if len(path_a) == 0 or len(path_b) == 0:
        return 0, 0.0, False, []

    # Early same-cell termination semantics
    if (path_a[0][0] == path_b[0][0]) and (path_a[0][1] == path_b[0][1]):
        return 0, 0.0, True, []

    n = min(len(path_a), len(path_b))
    a_rows, a_cols = set(), set()
    b_rows, b_cols = set(), set()
    levels_data = []
    terminated = False

    # 1. Simulation Phase
    for level in range(n):
        rA, cA = int(path_a[level][0]), int(path_a[level][1])
        rB, cB = int(path_b[level][0]), int(path_b[level][1])

        a_rows.add(rA); a_cols.add(cA)
        b_rows.add(rB); b_cols.add(cB)

        term_A = (rA in b_rows) and (cA in b_cols)
        term_B = (rB in a_rows) and (cB in a_cols)

        levels_data.append({
            "level": level, "rA": rA, "cA": cA, "rB": rB, "cB": cB,
        })

        if term_A or term_B:
            terminated = True
            break

    # 2. Scoring Phase
    total_score_unc = 0.0
    has_any_cancellation = False

    for d in levels_data:
        lvl = d["level"]
        rA, cA = d["rA"], d["cA"]
        rB, cB = d["rB"], d["cB"]
        
        canc_Ar = (rA in b_rows)
        canc_Ac = (cA in b_cols)
        canc_Br = (rB in a_rows)
        canc_Bc = (cB in a_cols)

        canc_count = int(canc_Ar) + int(canc_Ac) + int(canc_Br) + int(canc_Bc)
        unc_count  = 4 - canc_count
        contrib    = unc_count * lvl

        if canc_count > 0:
            has_any_cancellation = True

        d.update({
            "canc_Ar": canc_Ar, "canc_Ac": canc_Ac, 
            "canc_Br": canc_Br, "canc_Bc": canc_Bc,
            "canc_count": canc_count, 
            "uncanc_count": unc_count, 
            "unc_contrib": contrib
        })
        total_score_unc += contrib

    # 3. Determine Type
    if terminated:
        tcode = TYPE_T
    elif has_any_cancellation:
        tcode = TYPE_UC
    else:
        tcode = TYPE_UU

    return tcode, total_score_unc, terminated, levels_data


# ------------------------------------------------------------------------------
# Numba-accelerated (used by workers/canary)
# ------------------------------------------------------------------------------

@njit
def smod_pair_numba(path_a, path_b):
    """
    Returns: (type_code, total_score_unc)
    0 = T, 1 = UC, 2 = UU
    """
    len_a = path_a.shape[0]
    len_b = path_b.shape[0]

    if len_a == 0 or len_b == 0:
        return 0, 0.0

    # Early same-cell check
    if (path_a[0, 0] == path_b[0, 0]) and (path_a[0, 1] == path_b[0, 1]):
        return 0, 0.0

    n = len_a if len_a < len_b else len_b

    a_rows = set()
    a_cols = set()
    b_rows = set()
    b_cols = set()

    terminated = False
    has_any_cancellation = False
    final_level = 0

    # 1. Simulation Loop
    for level in range(n):
        final_level = level

        rA = path_a[level, 0]
        cA = path_a[level, 1]
        rB = path_b[level, 0]
        cB = path_b[level, 1]

        a_rows.add(rA)
        a_cols.add(cA)
        b_rows.add(rB)
        b_cols.add(cB)

        term_A = (rA in b_rows) and (cA in b_cols)
        term_B = (rB in a_rows) and (cB in a_cols)

        if term_A or term_B:
            terminated = True
            break
            
    # 2. Scoring Loop
    total_score = 0.0
    for level in range(final_level + 1):
        rA = path_a[level, 0]
        cA = path_a[level, 1]
        rB = path_b[level, 0]
        cB = path_b[level, 1]

        canc_Ar = rA in b_rows
        canc_Ac = cA in b_cols
        canc_Br = rB in a_rows
        canc_Bc = cB in a_cols
        
        canc = 0
        if canc_Ar: canc += 1
        if canc_Ac: canc += 1
        if canc_Br: canc += 1
        if canc_Bc: canc += 1
        
        if canc > 0:
            has_any_cancellation = True

        unc = 4 - canc
        total_score += unc * level

    # 3. Determine Type
    if terminated:
        tcode = 0 # TYPE_T
    elif has_any_cancellation:
        tcode = 1 # TYPE_UC
    else:
        tcode = 2 # TYPE_UU

    return tcode, total_score
