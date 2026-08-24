"""
Phase 4, Step 2: Regression Tracker
- Compares two eval runs side-by-side
- Flags: new failures, new passes, score deltas above threshold, category-level shifts
- This is the early warning system for model degradation
"""
import sqlite3
import pandas as pd
from datetime import datetime

SCORE_DELTA_THRESHOLD = 1  # Flag if score changes by more than this

def get_run_results(conn, run_id):
    """Load all results for a given run as a dict keyed by eval_case_id."""
    df = pd.read_sql_query('''
        SELECT eval_case_id, category, difficulty, score, passed,
               must_contain_pass, must_not_contain_pass, reasoning
        FROM eval_results WHERE run_id = ?
    ''', conn, params=(run_id,))
    return df.set_index("eval_case_id")

def get_latest_two_runs(conn):
    """Return the two most recent completed eval run IDs."""
    rows = conn.execute('''
        SELECT run_id, model_name, completed_at FROM eval_runs
        WHERE completed_at IS NOT NULL
        ORDER BY completed_at DESC LIMIT 2
    ''').fetchall()
    return rows

def run_regression_check(run_id_new=None, run_id_old=None):
    conn = sqlite3.connect("logs.db")

    # Auto-detect the two most recent runs if not specified
    if not run_id_new or not run_id_old:
        runs = get_latest_two_runs(conn)
        if len(runs) < 2:
            print("Need at least 2 completed eval runs to compare. Run eval_runner.py twice.")
            conn.close()
            return
        run_id_new, model_new, ts_new = runs[0]
        run_id_old, model_old, ts_old = runs[1]
    else:
        model_new = model_old = "unknown"

    print(f"\n{'='*65}")
    print(f"REGRESSION REPORT")
    print(f"  NEW run: {run_id_new} ({model_new})")
    print(f"  OLD run: {run_id_old} ({model_old})")
    print(f"{'='*65}")

    new_df = get_run_results(conn, run_id_new)
    old_df = get_run_results(conn, run_id_old)

    # Find common test cases
    common_ids = new_df.index.intersection(old_df.index)
    new_only    = new_df.index.difference(old_df.index)
    old_only    = old_df.index.difference(new_df.index)

    print(f"\nCoverage: {len(common_ids)} shared cases | {len(new_only)} new | {len(old_only)} removed")

    new_failures  = []
    new_passes    = []
    score_changes = []

    for case_id in common_ids:
        n = new_df.loc[case_id]
        o = old_df.loc[case_id]

        was_passing = bool(o["passed"])
        is_passing  = bool(n["passed"])
        score_delta = int(n["score"]) - int(o["score"])

        if was_passing and not is_passing:
            new_failures.append({
                "eval_id":   case_id,
                "category":  n["category"],
                "difficulty":n["difficulty"],
                "old_score": o["score"],
                "new_score": n["score"],
                "reasoning": n["reasoning"]
            })
        elif not was_passing and is_passing:
            new_passes.append({
                "eval_id":   case_id,
                "category":  n["category"],
                "difficulty":n["difficulty"],
                "old_score": o["score"],
                "new_score": n["score"],
            })
        elif abs(score_delta) > SCORE_DELTA_THRESHOLD:
            score_changes.append({
                "eval_id":   case_id,
                "category":  n["category"],
                "difficulty":n["difficulty"],
                "old_score": o["score"],
                "new_score": n["score"],
                "delta":     score_delta,
            })

    # Print new failures (most critical)
    if new_failures:
        print(f"\n[REGRESSIONS] {len(new_failures)} test case(s) NOW FAILING (used to pass):")
        for r in new_failures:
            print(f"  Eval #{r['eval_id']} [{r['category']}/{r['difficulty']}] "
                  f"score {r['old_score']} -> {r['new_score']}: {r['reasoning'][:80]}")
    else:
        print("\n[REGRESSIONS] None detected.")

    # Print new passes (improvements)
    if new_passes:
        print(f"\n[IMPROVEMENTS] {len(new_passes)} test case(s) NOW PASSING (used to fail):")
        for r in new_passes:
            print(f"  Eval #{r['eval_id']} [{r['category']}/{r['difficulty']}] "
                  f"score {r['old_score']} -> {r['new_score']}")
    else:
        print("\n[IMPROVEMENTS] None detected.")

    # Score changes above threshold
    if score_changes:
        print(f"\n[SCORE SHIFTS] {len(score_changes)} case(s) changed score by >{SCORE_DELTA_THRESHOLD}:")
        for r in score_changes:
            direction = "up" if r["delta"] > 0 else "DOWN"
            print(f"  Eval #{r['eval_id']} [{r['category']}/{r['difficulty']}] "
                  f"score {r['old_score']} -> {r['new_score']} ({direction})")

    # Category-level performance shift
    print("\n[CATEGORY PERFORMANCE SHIFT]")
    print(f"  {'Category':<20} {'Old Pass%':>10} {'New Pass%':>10} {'Change':>8}")
    print(f"  {'-'*50}")

    all_cats = set(new_df["category"].unique()) | set(old_df["category"].unique())
    for cat in sorted(all_cats):
        old_cat = old_df[old_df["category"] == cat]
        new_cat = new_df[new_df["category"] == cat]
        old_rate = old_cat["passed"].mean() if not old_cat.empty else None
        new_rate = new_cat["passed"].mean() if not new_cat.empty else None
        if old_rate is not None and new_rate is not None:
            delta = new_rate - old_rate
            flag  = " << WARNING" if delta < -0.1 else (" << IMPROVED" if delta > 0.1 else "")
            print(f"  {cat:<20} {old_rate:>9.0%} {new_rate:>10.0%} {delta:>+8.0%}{flag}")

    # Save regression report to DB
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS regression_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id_new TEXT,
        run_id_old TEXT,
        new_failures INTEGER,
        new_passes INTEGER,
        score_shifts INTEGER,
        created_at TEXT
    )''')
    c.execute('''INSERT INTO regression_reports
        (run_id_new, run_id_old, new_failures, new_passes, score_shifts, created_at)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (run_id_new, run_id_old, len(new_failures), len(new_passes),
         len(score_changes), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"\n{'='*65}\n")

if __name__ == "__main__":
    run_regression_check()
