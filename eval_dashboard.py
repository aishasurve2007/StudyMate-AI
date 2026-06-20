"""
StudyMate evaluation dashboard.

Shows the metrics produced by evaluate.py in a clean Streamlit view — the
headline numbers, a per-question table, and a faithfulness chart. Run it with:

    streamlit run eval_dashboard.py

(Run evaluate.py first so there are results in data/eval/ to display.)
"""
import glob
import json
import statistics

import pandas as pd
import streamlit as st

st.set_page_config(page_title="StudyMate — RAG Evaluation", layout="wide")
st.title("StudyMate AI — RAG Evaluation")
st.caption("Measured on a fixed test set with deterministic generation (temperature = 0). "
           "Single run — expect ~1–2 point variation between runs.")

files = sorted(glob.glob("data/eval/*.json"))
if not files:
    st.warning("No results found. Run evaluate.py first.")
    st.stop()

default = "data/eval/top5.json"
path = st.selectbox("Results file", files,
                    index=files.index(default) if default in files else 0)

data = json.load(open(path, encoding="utf-8"))
summary, rows = data["summary"], data["rows"]
settings = data.get("settings", {})

st.markdown(
    f"**Config:** retrieve = {settings.get('retrieve', '?')}, "
    f"top = {settings.get('top', '?')} &nbsp;•&nbsp; "
    f"**Questions:** {summary['questions']}"
)

# headline metrics
median_latency = round(statistics.median(r["total_seconds"] for r in rows), 2)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Faithfulness", f"{summary['avg_support_score']}%")
c2.metric("Hallucination", f"{summary['avg_hallucination_risk']}%")
c3.metric("Retrieval hit-rate", f"{summary['retrieval_hit_rate']}%")
c4.metric("Answer recall", f"{summary['avg_answer_recall']}%")
c5.metric("Median latency", f"{median_latency}s")

df = pd.DataFrame(rows)

# faithfulness by question
st.subheader("Faithfulness by question")
chart = df.set_index("question")[["support_score", "hallucination_risk"]]
st.bar_chart(chart, height=380)

# per-question detail
st.subheader("Per-question results")
view = df[["question", "retrieval_hit", "support_score",
           "hallucination_risk", "answer_recall", "total_seconds"]].copy()
view.columns = ["Question", "Retrieved", "Support %", "Halluc %", "Recall", "Latency (s)"]


def highlight_low(row):
    color = "background-color: #fff3cd" if row["Support %"] < 60 else ""
    return [color] * len(row)


st.dataframe(view.style.apply(highlight_low, axis=1), use_container_width=True, height=560)

st.caption(
    "Faithfulness = fraction of answer claims grounded in the retrieved source. "
    "Hallucination = fraction of claims the source contradicts. "
    "Rows under 60% support are highlighted for review."
)