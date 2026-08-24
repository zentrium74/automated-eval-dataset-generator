import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan

def run_clustering():
    conn = sqlite3.connect('logs.db')
    
    # Only cluster sampled logs that haven't been clustered yet
    df = pd.read_sql_query("SELECT id, user_prompt, model_response FROM normalized_logs WHERE is_sampled = 1 AND cluster_id IS NULL", conn)
    
    if df.empty:
        print("No new sampled logs to cluster.")
        conn.close()
        return

    print(f"Clustering {len(df)} sampled interactions...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['user_prompt'].tolist(), show_progress_bar=False)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    df['cluster_id'] = clusterer.fit_predict(embeddings)
    
    c = conn.cursor()
    for index, row in df.iterrows():
        c.execute("UPDATE normalized_logs SET cluster_id = ? WHERE id = ?", (int(row['cluster_id']), row['id']))
        
    conn.commit()
    conn.close()
    
    outliers = len(df[df['cluster_id'] == -1])
    print(f"Clustering complete. Found {outliers} anomalies/novel requests (cluster -1).")

if __name__ == "__main__":
    run_clustering()
