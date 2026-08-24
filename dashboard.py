import streamlit as st
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta

st.set_page_config(page_title="Eval Curation Dashboard", layout="wide", page_icon="🔬")
st.title("Human-in-the-Loop Eval Curation")
st.markdown("Browse, edit, and approve auto-generated eval cases to ensure dataset quality.")

# --- DATABASE HELPERS ---
def run_query(query, params=()):
    conn = sqlite3.connect('logs.db')
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def execute_db(query, params=()):
    conn = sqlite3.connect('logs.db')
    conn.execute(query, params)
    conn.commit()
    conn.close()

def approve_review(eval_id):
    execute_db("UPDATE confidence_scores SET status = 'approved' WHERE eval_id = ?", (eval_id,))
    execute_db("UPDATE human_review_queue SET reviewed = 1 WHERE eval_id = ?", (eval_id,))
    st.toast(f"Eval #{eval_id} approved!")

def reject_review(eval_id):
    execute_db("UPDATE confidence_scores SET status = 'rejected' WHERE eval_id = ?", (eval_id,))
    execute_db("UPDATE human_review_queue SET reviewed = 1 WHERE eval_id = ?", (eval_id,))
    st.toast(f"Eval #{eval_id} rejected!")

def update_eval_notes(eval_id, notes):
    # Ensure notes column exists
    try:
        execute_db("ALTER TABLE eval_dataset ADD COLUMN manual_notes TEXT")
    except:
        pass # Column exists
    execute_db("UPDATE eval_dataset SET manual_notes = ? WHERE id = ?", (notes, eval_id))
    st.toast("Notes saved!")

# --- LOAD DATA ---
evals_query = '''
    SELECT e.id, e.category, e.quality_score, e.difficulty, e.is_positive_case, e.labeled_by, e.manual_notes,
           i.user_prompt, i.model_response, i.timestamp,
           cs.status, cs.agreement_ratio,
           g.golden_answer, g.rubric_score_5,
           m.expected_behavior, m.must_contain, m.must_not_contain
    FROM eval_dataset e
    JOIN normalized_logs i ON e.interaction_id = i.id
    LEFT JOIN confidence_scores cs ON e.id = cs.eval_id
    LEFT JOIN golden_answers g ON e.id = g.eval_id
    LEFT JOIN multi_labels m ON e.id = m.eval_id
'''
try:
    execute_db("ALTER TABLE eval_dataset ADD COLUMN manual_notes TEXT")
except:
    pass
evals_df = run_query(evals_query)

queue_query = '''
    SELECT hr.eval_id, hr.reason, hr.reviewed, hr.created_at,
           i.user_prompt, i.model_response,
           cs.majority_category, cs.majority_behavior, cs.agreement_ratio
    FROM human_review_queue hr
    JOIN eval_dataset e ON hr.eval_id = e.id
    JOIN normalized_logs i ON e.interaction_id = i.id
    JOIN confidence_scores cs ON e.id = cs.eval_id
    WHERE hr.reviewed = 0
'''
queue_df = run_query(queue_query)

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs([
    "Dataset Explorer (Approved)", 
    f"Human Review Queue ({len(queue_df)})", 
    "Pipeline & Growth Metrics"
])

with tab1:
    st.subheader("Dataset Explorer")
    if not evals_df.empty:
        approved_df = evals_df[evals_df['status'] == 'approved'].copy()
        
        # Filters
        c1, c2, c3 = st.columns(3)
        cat_filter = c1.multiselect("Filter by Category", approved_df['category'].unique())
        diff_filter = c2.multiselect("Filter by Difficulty", approved_df['difficulty'].unique())
        qual_filter = c3.slider("Min Quality Score", 1, 5, 1)

        filtered_df = approved_df
        if cat_filter: filtered_df = filtered_df[filtered_df['category'].isin(cat_filter)]
        if diff_filter: filtered_df = filtered_df[filtered_df['difficulty'].isin(diff_filter)]
        filtered_df = filtered_df[filtered_df['quality_score'] >= qual_filter]

        st.markdown(f"**Showing {len(filtered_df)} approved test cases.**")

        for _, row in filtered_df.iterrows():
            with st.expander(f"Eval #{row['id']} | {row['category']} ({row['difficulty']}) | Score: {row['quality_score']}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Original Interaction**")
                    st.info(f"**User:** {row['user_prompt']}")
                    st.success(f"**Model:** {row['model_response']}")
                    
                with col_b:
                    st.markdown("**Eval Labels**")
                    st.write(f"- **Expected Behavior:** {row['expected_behavior']}")
                    st.write(f"- **Golden Answer:** {row['golden_answer'] or 'N/A'}")
                    st.write(f"- **Rubric (5):** {row['rubric_score_5'] or 'N/A'}")
                    
                    st.markdown("**Assertions**")
                    must_c = json.loads(row['must_contain']) if pd.notna(row['must_contain']) else []
                    must_nc = json.loads(row['must_not_contain']) if pd.notna(row['must_not_contain']) else []
                    st.write("✅ **Must Contain:** " + ", ".join(must_c) if must_c else "None")
                    st.write("❌ **Must NOT Contain:** " + ", ".join(must_nc) if must_nc else "None")

                # Manual Notes Editor
                current_notes = row.get('manual_notes', '')
                if pd.isna(current_notes): current_notes = ''
                
                notes = st.text_area("Reviewer Notes & Annotations", value=current_notes, key=f"note_{row['id']}")
                if st.button("Save Notes", key=f"btn_note_{row['id']}"):
                    update_eval_notes(row['id'], notes)
                    st.rerun()

with tab2:
    st.subheader("Human Review Queue (Low Confidence Auto-Labels)")
    if not queue_df.empty:
        st.warning(f"You have {len(queue_df)} auto-labeled cases requiring human verification.")
        
        for _, row in queue_df.iterrows():
            eid = row['eval_id']
            with st.container():
                st.markdown(f"### Eval #{eid}")
                st.error(f"**Flag Reason:** {row['reason']} (Agreement: {row['agreement_ratio']:.0%})")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**User Prompt:** {row['user_prompt']}")
                    st.success(f"**Model Response:** {row['model_response']}")
                with c2:
                    st.markdown("**Proposed Auto-Labels (Needs Verification):**")
                    st.write(f"- Category: **{row['majority_category']}**")
                    st.write(f"- Expected Behavior: **{row['majority_behavior']}**")
                    
                    st.markdown("**Actions**")
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("✅ Approve As-Is", key=f"app_{eid}", use_container_width=True):
                            approve_review(eid)
                            st.rerun()
                    with col_act2:
                        if st.button("🗑️ Reject (Discard)", key=f"rej_{eid}", use_container_width=True):
                            reject_review(eid)
                            st.rerun()
                st.divider()
    else:
        st.success("All caught up! The review queue is empty.")

with tab3:
    st.subheader("Pipeline Health & Growth Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Generated Evals", len(evals_df))
    app_count = len(evals_df[evals_df['status'] == 'approved']) if 'status' in evals_df.columns else 0
    c2.metric("Total Approved", app_count)
    rej_count = len(evals_df[evals_df['status'] == 'rejected']) if 'status' in evals_df.columns else 0
    c3.metric("Total Rejected", rej_count)
    
    st.subheader("Coverage Matrix")
    if not evals_df.empty:
        heatmap_data = pd.crosstab(evals_df['category'], evals_df['difficulty'])
        st.dataframe(heatmap_data.style.background_gradient(cmap='Blues'))
