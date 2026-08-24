"""
Phase 4, Step 1: Eval Harness
- Takes any model endpoint (Ollama or OpenRouter)
- Runs the full eval dataset against it
- Scores each response against golden answers + rubrics + must_contain / must_not_contain
- Produces structured pass/fail reports per category, difficulty, and metric dimension
"""
import sqlite3
import pandas as pd
import requests
import json
import os
import time
from datetime import datetime

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OLLAMA_URL         = "http://localhost:11434/api/chat"

# LLM-as-judge system prompt for scoring a model response
SCORER_PROMPT = """You are an expert AI evaluator scoring a model response for a customer support chatbot.

You will be given:
- The original user prompt
- The golden (ideal) answer or rubric
- The actual model response to score
- The must_contain assertions (things that MUST be in the response)
- The must_not_contain assertions (hallucination traps that must NOT appear)

Output ONLY valid JSON:
{
  "score": <integer 1-5>,
  "passed": <boolean>,
  "must_contain_pass": <boolean>,
  "must_not_contain_pass": <boolean>,
  "reasoning": "<brief explanation>"
}

Rules:
- passed = true only if score >= 4 AND must_contain_pass AND must_not_contain_pass
- must_contain_pass = true if all must_contain items are addressed in the response
- must_not_contain_pass = true if none of the must_not_contain items appear
"""

def get_model_response(prompt, system_prompt, model_config):
    """Call the model under evaluation. model_config = {"type": "ollama"|"openrouter", "model": "..."}"""
    if model_config["type"] == "ollama":
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt}
                ],
                "stream": False
            }, timeout=120)
            res.raise_for_status()
            return res.json()["message"]["content"]
        except Exception as e:
            return f"[MODEL ERROR: {e}]"

    elif model_config["type"] == "openrouter":
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model_config["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt}
                    ]
                }, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[MODEL ERROR: {e}]"
    return "[UNSUPPORTED MODEL TYPE]"

def score_response(user_prompt, model_response, golden_answer, rubric_score_5,
                   must_contain_json, must_not_contain_json, expected_behavior, scorer_config):
    """Use an LLM judge to score the model's response."""
    must_contain     = json.loads(must_contain_json)     if must_contain_json     else []
    must_not_contain = json.loads(must_not_contain_json) if must_not_contain_json else []

    reference = golden_answer or rubric_score_5 or "No reference provided."

    judge_user = f"""User Prompt: {user_prompt}

Golden Reference / Rubric (score 5): {reference}

Model Response to Score: {model_response}

Must Contain: {json.dumps(must_contain)}
Must NOT Contain: {json.dumps(must_not_contain)}
Expected Behavior: {expected_behavior}
"""
    if scorer_config["type"] == "openrouter":
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": scorer_config["model"],
                    "messages": [
                        {"role": "system", "content": SCORER_PROMPT},
                        {"role": "user",   "content": judge_user}
                    ],
                    "response_format": {"type": "json_object"}
                }, timeout=60)
            if res.status_code == 429:
                time.sleep(12)
                return {"score": 3, "passed": False, "must_contain_pass": False,
                        "must_not_contain_pass": False, "reasoning": "Rate limited"}
            res.raise_for_status()
            parsed = json.loads(res.json()["choices"][0]["message"]["content"])
            return {
                "score":                 parsed.get("score") or 3,
                "passed":                bool(parsed.get("passed", False)),
                "must_contain_pass":     bool(parsed.get("must_contain_pass", False)),
                "must_not_contain_pass": bool(parsed.get("must_not_contain_pass", True)),
                "reasoning":             parsed.get("reasoning") or "",
            }
        except Exception as e:
            return {"score": 3, "passed": False, "must_contain_pass": False,
                    "must_not_contain_pass": False, "reasoning": str(e)}
    return {"score": 3, "passed": False, "must_contain_pass": False,
            "must_not_contain_pass": False, "reasoning": "Unsupported scorer type"}

def setup_eval_tables(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            model_name TEXT,
            model_type TEXT,
            started_at TEXT,
            completed_at TEXT,
            total_cases INTEGER,
            passed INTEGER,
            failed INTEGER,
            pass_rate REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            eval_case_id INTEGER,
            category TEXT,
            difficulty TEXT,
            expected_behavior TEXT,
            model_response TEXT,
            score INTEGER,
            passed INTEGER,
            must_contain_pass INTEGER,
            must_not_contain_pass INTEGER,
            reasoning TEXT,
            FOREIGN KEY(run_id) REFERENCES eval_runs(run_id)
        )
    ''')
    conn.commit()

def run_eval(model_config, scorer_config=None, run_id=None, limit=None):
    """
    Run the full eval dataset against a model.
    model_config = {"type": "ollama"|"openrouter", "model": "model-name"}
    scorer_config = same format (defaults to Nemotron on OpenRouter)
    """
    if scorer_config is None:
        scorer_config = {"type": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b:free"}
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    conn = sqlite3.connect('logs.db')
    setup_eval_tables(conn)

    # Load eval cases that are approved + dedup-accepted
    query = '''
        SELECT
            e.id        AS eval_id,
            e.category,
            e.difficulty,
            i.user_prompt,
            g.golden_answer,
            g.rubric_score_5,
            m.must_contain,
            m.must_not_contain,
            m.expected_behavior
        FROM eval_dataset e
        JOIN normalized_logs    i  ON e.interaction_id   = i.id
        JOIN confidence_scores  cs ON e.id               = cs.eval_id
        LEFT JOIN golden_answers   g  ON e.id            = g.eval_id
        LEFT JOIN multi_labels     m  ON e.id            = m.eval_id
        LEFT JOIN dedup_log        d  ON e.id            = d.candidate_eval_id
        WHERE cs.status = 'approved'
          AND (d.action = 'accepted' OR d.id IS NULL)
    '''
    df = pd.read_sql_query(query, conn)
    if limit:
        df = df.head(limit)

    if df.empty:
        print("No approved eval cases found. Run the pipeline first.")
        conn.close()
        return None

    system_prompt = "You are a helpful customer support AI assistant."
    print(f"\nEval Run: {run_id}")
    print(f"Model:    {model_config['model']} ({model_config['type']})")
    print(f"Cases:    {len(df)}")
    print("-" * 60)

    # Register run
    c = conn.cursor()
    c.execute('''INSERT OR IGNORE INTO eval_runs
        (run_id, model_name, model_type, started_at, total_cases)
        VALUES (?, ?, ?, ?, ?)''',
        (run_id, model_config["model"], model_config["type"],
         datetime.now().isoformat(), len(df)))
    conn.commit()

    results = []
    for idx, row in df.iterrows():
        print(f"  [{idx+1}/{len(df)}] {row['category']}/{row['difficulty']} | {row['user_prompt'][:60]}...")

        # Step 1: Get the model's response
        model_resp = get_model_response(row["user_prompt"], system_prompt, model_config)

        # Step 2: Score the response
        score_result = score_response(
            row["user_prompt"], model_resp,
            row.get("golden_answer"), row.get("rubric_score_5"),
            row.get("must_contain"), row.get("must_not_contain"),
            row.get("expected_behavior", "should_answer"),
            scorer_config
        )

        passed = score_result["passed"]
        print(f"          Score={score_result['score']} Passed={'PASS' if passed else 'FAIL'} | {score_result['reasoning'][:60]}")

        c.execute('''INSERT INTO eval_results
            (run_id, eval_case_id, category, difficulty, expected_behavior,
             model_response, score, passed, must_contain_pass, must_not_contain_pass, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            run_id, row["eval_id"], row["category"], row["difficulty"],
            row.get("expected_behavior"), model_resp,
            score_result["score"], int(passed),
            int(score_result["must_contain_pass"]),
            int(score_result["must_not_contain_pass"]),
            score_result["reasoning"]
        ))
        conn.commit()
        results.append({"category": row["category"], "difficulty": row["difficulty"], **score_result})
        time.sleep(8)  # Rate limit between scoring calls

    # Summary
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    rate   = passed / total if total > 0 else 0

    c.execute('''UPDATE eval_runs SET
        completed_at=?, passed=?, failed=?, pass_rate=?
        WHERE run_id=?''',
        (datetime.now().isoformat(), passed, failed, rate, run_id))
    conn.commit()

    # Category breakdown
    result_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE — {run_id}")
    print(f"Overall: {passed}/{total} passed ({rate:.0%})")
    print(f"{'='*60}")
    if not result_df.empty:
        breakdown = result_df.groupby(["category", "difficulty"])["passed"].agg(["sum","count"])
        breakdown["pass_rate"] = breakdown["sum"] / breakdown["count"]
        print(breakdown.to_string())
    print(f"{'='*60}\n")

    conn.close()
    return run_id


if __name__ == "__main__":
    # Default: test against Nemotron (same free model we use for labeling)
    model  = {"type": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b:free"}
    scorer = {"type": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b:free"}
    run_eval(model, scorer)
