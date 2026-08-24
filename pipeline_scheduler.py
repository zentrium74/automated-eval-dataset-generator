import time
from apscheduler.schedulers.background import BackgroundScheduler
import data_generator
import ingestion
import cluster_logs
import label_evals
import golden_answer_generator
import multi_label
import confidence_router
import deduplication
import coverage_generator

def run_pipeline():
    print("\n" + "="*50)
    print("  PIPELINE RUN STARTING")
    print("="*50)

    # --- COVERAGE GAP FILL (runs first) ---
    print("\n[Coverage] Checking for underrepresented categories...")
    coverage_generator.run_targeted_generation()

    # --- PHASE 1: Ingest & Sample ---
    print("\n[Phase 1] Generating & ingesting logs...")
    data_generator.generate_raw_logs(num_logs=30)
    ingestion.run_ingestion_and_sampling()

    # --- PHASE 2: Cluster & Label ---
    print("\n[Phase 2] Clustering & evaluating edge cases...")
    cluster_logs.run_clustering()
    label_evals.label_anomalies()

    # --- PHASE 3: Auto-Labeling Pipeline ---
    print("\n[Phase 3] Generating golden answers...")
    golden_answer_generator.generate_golden_answers()

    print("\n[Phase 3] Running multi-dimensional labeling...")
    multi_label.run_multi_labeling()

    print("\n[Phase 3] Running confidence-based routing...")
    confidence_router.run_confidence_routing()

    print("\n[Phase 3] Running deduplication & coverage check...")
    deduplication.run_deduplication()

    print("\n" + "="*50)
    print("  PIPELINE RUN COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    # Run every 5 minutes for demo purposes
    scheduler.add_job(run_pipeline, 'interval', minutes=5)
    scheduler.start()

    print("Full Pipeline Scheduler started (Phases 1-3). Press Ctrl+C to exit.")
    run_pipeline()

    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
