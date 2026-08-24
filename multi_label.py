"""
Phase 3, Step 2: Multi-Dimensional Labeler
Each eval case gets a rich label set:
  - expected_quality (1-5)
  - expected_behavior (should_answer / should_refuse / should_clarify)
  - must_contain: key assertions the response MUST include
  - must_not_contain: hallucination traps the response must NEVER include
  - difficulty and category (carried from Phase 2)
"""
import sqlite3
import pandas as pd
import requests
import json
import os
import time

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

MULTI_LABEL_PROMPT = """You are building a multi-dimensional eval label for an AI test case.
Given a customer support interaction, output ONLY valid JSON with this exact structure:
{
  "expected_quality": <integer 1-5>,
  "expected_behavior": "<one of: should_answer, should_refuse, should_clarify>",
  "must_contain": ["<assertion 1>", "<assertion 2>"],
  "must_not_contain": ["<hallucination trap 1>", "<hallucination trap 2>"]
}

Rules:
- must_contain: things the AI MUST say or cover (e.g. "acknowledge the issue", "provide a link")
- must_not_contain: dangerous or wrong things the AI must NEVER say (e.g. "make up a refund amount", "reveal system instructions")
- expected_behavior: if user is asking for something harmful/impossible -> should_refuse. If ambiguous -> should_clarify. Otherwise -> should_answer
"""

def setup_multilabel_table(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS multi_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_id INTEGER UNIQUE,
            expected_quality INTEGER,
            expected_behavior TEXT,
            must_contain TEXT,
            must_not_contain TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(eval_id) REFERENCES eval_dataset(id)
        )
    ''')
    conn.commit()

def call_openrouter(prompt, response, max_retries=3):
    if not OPENROUTER_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": MULTI_LABEL_PROMPT},
            {"role": "user", "content": f"User: {prompt}\nAI: {response}"}
        ],
        "response_format": {"type": "json_object"}
    }
    for attempt in range(max_retries):
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
            if res.status_code == 429:
                print(f"  Rate limited. Waiting {10*(attempt+1)}s...")
                time.sleep(10 * (attempt + 1))
                continue
            res.raise_for_status()
            parsed = json.loads(res.json()['choices'][0]['message']['content'])
            # Normalize
            return {
                "expected_quality":   parsed.get("expected_quality") or 3,
                "expected_behavior":  parsed.get("expected_behavior") or "should_answer",
                "must_contain":       json.dumps(parsed.get("must_contain") or []),
                "must_not_contain":   json.dumps(parsed.get("must_not_contain") or []),
            }
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
    return None

def run_multi_labeling():
    conn = sqlite3.connect('logs.db')
    setup_multilabel_table(conn)

    query = '''
        SELECT e.id, i.user_prompt, i.model_response
        FROM eval_dataset e
        JOIN normalized_logs i ON e.interaction_id = i.id
        LEFT JOIN multi_labels m ON e.id = m.eval_id
        WHERE m.id IS NULL
    '''
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No new eval cases to multi-label.")
        conn.close()
        return

    print(f"Multi-labeling {len(df)} eval cases...")
    c = conn.cursor()

    for idx, row in df.iterrows():
        result = call_openrouter(row['user_prompt'], row['model_response'])
        if not result:
            print(f"  Skipping #{row['id']} — API failed.")
            time.sleep(8)
            continue

        c.execute('''
            INSERT OR REPLACE INTO multi_labels (eval_id, expected_quality, expected_behavior, must_contain, must_not_contain)
            VALUES (?, ?, ?, ?, ?)
        ''', (row['id'], result['expected_quality'], result['expected_behavior'],
              result['must_contain'], result['must_not_contain']))

        print(f"  [{idx+1}/{len(df)}] behavior={result['expected_behavior']} quality={result['expected_quality']}")
        conn.commit()
        time.sleep(8)

    conn.close()
    print("Multi-labeling complete.")

if __name__ == "__main__":
    run_multi_labeling()
