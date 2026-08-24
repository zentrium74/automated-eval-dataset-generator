import sqlite3
import pandas as pd
import re

def setup_ingestion_tables(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS normalized_logs (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            user_prompt TEXT,
            model_response TEXT,
            latency_ms INTEGER,
            user_feedback TEXT,
            cluster_id INTEGER,
            is_sampled INTEGER DEFAULT 0
        )
    ''')
    conn.commit()

def redact_pii(text):
    if not text: return text
    # Redact Emails
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', text)
    # Redact Phone numbers (simple format)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
    return text

def run_ingestion_and_sampling():
    print("Running Ingestion & Sampling Pipeline...")
    conn = sqlite3.connect('logs.db')
    setup_ingestion_tables(conn)
    
    # Read raw logs that haven't been ingested
    query = """
        SELECT * FROM raw_logs 
        WHERE id NOT IN (SELECT id FROM normalized_logs)
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No new raw logs to ingest.")
        conn.close()
        return

    # 1. PII Redaction & Normalization
    df['user_prompt'] = df['user_prompt'].apply(redact_pii)
    df['model_response'] = df['model_response'].apply(redact_pii)
    
    # 2. Sampling Strategy (Signal-boosted sampling)
    # We want to oversample retries, thumbs_down, and high latency
    df['priority_score'] = 1
    df.loc[df['user_feedback'].isin(['thumbs_down', 'retry']), 'priority_score'] = 5
    df.loc[df['latency_ms'] > 1500, 'priority_score'] += 2
    
    # Sample based on weights
    sampled_df = df.sample(frac=0.5, weights='priority_score', random_state=42, replace=True).drop_duplicates(subset=['id'])
    
    c = conn.cursor()
    for _, row in df.iterrows():
        is_sampled = 1 if row['id'] in sampled_df['id'].values else 0
        c.execute('''
            INSERT INTO normalized_logs (id, timestamp, user_prompt, model_response, latency_ms, user_feedback, is_sampled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['id'], row['timestamp'], row['user_prompt'], row['model_response'], row['latency_ms'], row['user_feedback'], is_sampled))
        
    conn.commit()
    conn.close()
    print(f"Ingested {len(df)} logs. Sampled {len(sampled_df)} logs for evaluation.")

if __name__ == "__main__":
    run_ingestion_and_sampling()
