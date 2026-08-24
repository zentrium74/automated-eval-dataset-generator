"""
Phase 3, Step 4: Deduplication & Coverage Tracker
- Before any new test case is added, check cosine similarity against all existing ones
- Near-duplicate (similarity > 0.92) -> skip
- Tracks per-cluster coverage: which categories are over/under-represented
- Outputs coverage report to guide future data generation
"""
import sqlite3
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.92

model = None  # Lazy-loaded

def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def setup_dedup_table(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dedup_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_eval_id INTEGER,
            duplicate_of_eval_id INTEGER,
            similarity_score REAL,
            action TEXT,  -- 'accepted' or 'rejected_duplicate'
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()

def get_embeddings(texts):
    m = get_model()
    return m.encode(texts, show_progress_bar=False)

def run_deduplication():
    conn = sqlite3.connect('logs.db')
    setup_dedup_table(conn)

    # Get all approved eval cases
    approved_query = '''
        SELECT e.id, i.user_prompt
        FROM eval_dataset e
        JOIN normalized_logs i ON e.interaction_id = i.id
        JOIN confidence_scores cs ON e.id = cs.eval_id
        WHERE cs.status = 'approved'
    '''
    approved_df = pd.read_sql_query(approved_query, conn)

    # Get candidates not yet dedup-checked
    candidate_query = '''
        SELECT e.id, i.user_prompt
        FROM eval_dataset e
        JOIN normalized_logs i ON e.interaction_id = i.id
        JOIN confidence_scores cs ON e.id = cs.eval_id
        LEFT JOIN dedup_log d ON e.id = d.candidate_eval_id
        WHERE cs.status = 'approved' AND d.id IS NULL
    '''
    candidates_df = pd.read_sql_query(candidate_query, conn)

    if candidates_df.empty:
        print("No new candidates to deduplicate.")
        compute_coverage(conn)
        conn.close()
        return

    print(f"Deduplicating {len(candidates_df)} candidates against {len(approved_df)} existing entries...")
    c = conn.cursor()

    # Pre-embed all approved entries and all candidates
    approved_ids     = approved_df['id'].tolist()
    approved_prompts = approved_df['user_prompt'].tolist()
    cand_prompts     = candidates_df['user_prompt'].tolist()

    approved_embeddings = get_embeddings(approved_prompts) if approved_prompts else np.array([])
    cand_embeddings     = get_embeddings(cand_prompts)

    # Track which candidates were accepted so far (to check against each other too)
    accepted_so_far_embeddings = []
    accepted_so_far_ids        = []

    for i, (_, cand_row) in enumerate(candidates_df.iterrows()):
        cand_emb = cand_embeddings[i].reshape(1, -1)
        cand_id  = int(cand_row['id'])

        # Build comparison pool = approved entries (excluding self) + already-accepted candidates
        comparison_embeddings = []
        comparison_ids        = []

        for j, aid in enumerate(approved_ids):
            if aid != cand_id:  # Exclude self
                comparison_embeddings.append(approved_embeddings[j])
                comparison_ids.append(aid)

        for j, aid in enumerate(accepted_so_far_ids):
            comparison_embeddings.append(accepted_so_far_embeddings[j])
            comparison_ids.append(aid)

        if comparison_embeddings:
            pool = np.array(comparison_embeddings)
            sims = cosine_similarity(cand_emb, pool)[0]
            max_sim_idx = int(np.argmax(sims))
            max_sim     = float(sims[max_sim_idx])
            dup_of_id   = comparison_ids[max_sim_idx]
        else:
            max_sim   = 0.0
            dup_of_id = None

        if max_sim > SIMILARITY_THRESHOLD:
            action = "rejected_duplicate"
            print(f"  DUPLICATE (sim={max_sim:.3f}): eval #{cand_id} is near-duplicate of #{dup_of_id}")
        else:
            action = "accepted"
            dup_of_id = None
            accepted_so_far_embeddings.append(cand_embeddings[i])
            accepted_so_far_ids.append(cand_id)
            print(f"  ACCEPTED  (max_sim={max_sim:.3f}): eval #{cand_id} is unique")

        c.execute('''
            INSERT INTO dedup_log (candidate_eval_id, duplicate_of_eval_id, similarity_score, action)
            VALUES (?, ?, ?, ?)
        ''', (cand_id, dup_of_id, max_sim, action))
        conn.commit()

    compute_coverage(conn)
    conn.close()
    print("Deduplication complete.")

def compute_coverage(conn):
    """Print a coverage report: which categories have enough examples and which don't."""
    print("\n--- COVERAGE REPORT ---")
    try:
        coverage_query = '''
            SELECT e.category, e.difficulty, COUNT(*) as count,
                   AVG(cs.agreement_ratio) as avg_confidence
            FROM eval_dataset e
            JOIN confidence_scores cs ON e.id = cs.eval_id
            LEFT JOIN dedup_log d ON e.id = d.candidate_eval_id
            WHERE cs.status = 'approved' AND (d.action = 'accepted' OR d.id IS NULL)
            GROUP BY e.category, e.difficulty
            ORDER BY count ASC
        '''
        df = pd.read_sql_query(coverage_query, conn)
        if df.empty:
            print("No approved test cases yet.")
            return

        print(f"{'Category':<25} {'Difficulty':<12} {'Count':>5} {'Avg Confidence':>15}")
        print("-" * 60)
        for _, row in df.iterrows():
            flag = " << NEEDS MORE EXAMPLES" if row['count'] < 3 else ""
            print(f"{str(row['category']):<25} {str(row['difficulty']):<12} {int(row['count']):>5} {row['avg_confidence']:>14.0%}{flag}")

        # Save coverage to DB for dashboard
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS coverage_report (category TEXT, difficulty TEXT, count INTEGER, avg_confidence REAL, updated_at TEXT)')
        c.execute('DELETE FROM coverage_report')
        for _, row in df.iterrows():
            c.execute('INSERT INTO coverage_report VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)',
                      (row['category'], row['difficulty'], int(row['count']), row['avg_confidence']))
        conn.commit()
    except Exception as e:
        print(f"Coverage error: {e}")
    print("--- END REPORT ---\n")

if __name__ == "__main__":
    run_deduplication()
