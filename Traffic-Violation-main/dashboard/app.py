"""Simple local dashboard for the violation pipeline: pick an image, run the models + OCR on it,
and browse everything that's ever been logged to violations.db.

  .venv\\Scripts\\python.exe -m streamlit run dashboard\\app.py

Two tabs: "Run on an image" (upload a file or point at one on disk, see the annotated detections and
the violation call) and "Violation log" (every processed image, newest first, with simple filters and
counts). Both tabs read/write the same pipeline.aggregator.Pipeline and violations.db that
pipeline/aggregator.py's command line uses, so a snapshot logged from either place shows up in both.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from aggregator import DB_PATH, Pipeline  # noqa: E402

st.set_page_config(page_title="Violation Monitor", layout="wide")
st.title("Two-Wheeler Violation Monitor")


@st.cache_resource(show_spinner="Loading models (first run only — a few seconds)...")
def get_pipeline(helmet_model, triple_model, conf):
    return Pipeline(helmet_model=helmet_model, triple_model=triple_model, conf=conf)


def load_log():
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations ORDER BY id DESC", con)
    con.close()
    return df


with st.sidebar:
    st.header("Settings")
    helmet_model = st.selectbox("Helmet / plate model", ["yolo26", "option_b"], index=0)
    triple_model = st.selectbox("Triple-riding model", ["yolo26", "option_b"], index=0)
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    st.caption("yolo26 scored higher than option_b on both tasks' test split — see analysis/summary.txt.")

tab_run, tab_log = st.tabs(["Run on an image", "Violation log"])

with tab_run:
    st.write("Upload a snapshot, or give a path to one already on disk, then run both models and plate OCR on it.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    manual_path = st.text_input("...or a file path on this machine")
    speed = st.number_input("Measured speed (km/h) — optional", min_value=0.0, value=0.0, step=1.0)

    image_path = None
    if uploaded is not None:
        upload_dir = PROJECT_DIR / "violations" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / uploaded.name
        image_path.write_bytes(uploaded.getbuffer())
    elif manual_path:
        candidate = Path(manual_path)
        if candidate.exists():
            image_path = candidate
        else:
            st.error(f"not found: {candidate}")

    if image_path and st.button("Run inference", type="primary"):
        pipeline = get_pipeline(helmet_model, triple_model, conf)
        with st.spinner("Running models + OCR..."):
            record = pipeline.process(image_path, speed_kmph=(speed or None))

        col_image, col_details = st.columns([2, 1])
        with col_image:
            st.image(record["evidence_image"], caption="Coloured boxes = predictions", use_container_width=True)
        with col_details:
            if record["violation_types"]:
                st.error("VIOLATION: " + ", ".join(record["violation_types"]))
            else:
                st.success("No violation found")
            st.write(f"**Helmet/plate model:** {record['helmet_message']}")
            st.write(f"**Triple-riding model:** {record['triple_message']}")
            st.write(f"**Plate text:** {record['plate_number'] or '(none read)'}")
            if record["plate_number"] and not record["plate_valid_format"]:
                st.caption("Does not match the Indian plate format — may be a misread.")
            if record["speed_kmph"] is not None:
                st.write(f"**Speed:** {record['speed_kmph']:.1f} km/h")
            st.caption(f"Logged as row {record['id']} in violations.db")

with tab_log:
    df = load_log()
    if df.empty:
        st.info("No images processed yet — run one in the first tab.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total checked", len(df))
        c2.metric("No violation", int((df["violation_types"] == "").sum()))
        c3.metric("No-helmet flags", int(df["violation_types"].str.contains("no_helmet").sum()))
        c4.metric("Triple-riding flags", int(df["violation_types"].str.contains("triple_riding").sum()))

        only_violations = st.checkbox("Show violations only")
        view = df[df["violation_types"] != ""] if only_violations else df
        st.dataframe(
            view[["id", "timestamp", "plate_number", "violation_types", "speed_kmph", "raw_image_path"]],
            use_container_width=True, hide_index=True,
        )

        selected = st.selectbox("View evidence image for row id", [""] + view["id"].astype(str).tolist())
        if selected:
            row = view[view["id"] == int(selected)].iloc[0]
            if row["annotated_image_path"] and Path(row["annotated_image_path"]).exists():
                st.image(row["annotated_image_path"], use_container_width=True)
            else:
                st.warning("Evidence image file is missing on disk for this row.")
