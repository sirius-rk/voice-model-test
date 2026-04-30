from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from stt_tts_benchmark.database import delete_results, export_results_csv, fetch_results


def render_history(
    *,
    database_path: Path,
    export_dir: Path,
    page_type: str,
    title: str,
    project_root: Path,
    audio_column: str,
) -> None:
    st.subheader(title)
    rows = fetch_results(database_path, page_type=page_type, limit=100)
    if not rows:
        st.info("No history yet.")
        return

    table_rows = [_with_select_flag(row) for row in rows]
    edited = st.data_editor(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        disabled=[column for column in table_rows[0] if column != "select"],
        column_config={
            "select": st.column_config.CheckboxColumn("Delete", default=False),
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "input_text": st.column_config.TextColumn("Input text", width="large"),
            "transcript": st.column_config.TextColumn("Transcript", width="large"),
            "output_audio_path": st.column_config.TextColumn("Output audio", width="large"),
            "input_audio_path": st.column_config.TextColumn("Input audio", width="large"),
            "error_message": st.column_config.TextColumn("Error", width="large"),
        },
        key=f"{page_type}_history_editor",
    )

    selected_ids = [int(row["id"]) for row in edited.to_dict("records") if row.get("select")]
    selected_rows = [row for row in rows if int(row["id"]) in selected_ids]

    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("Delete selected", disabled=not selected_ids, key=f"{page_type}_delete_selected"):
        deleted_files = _delete_associated_audio(selected_rows, project_root, audio_column)
        delete_results(database_path, selected_ids)
        st.success(f"Deleted {len(selected_ids)} row(s) and {deleted_files} audio file(s).")
        st.rerun()

    if action_cols[1].button("Export CSV", key=f"{page_type}_export_csv"):
        csv_path = export_results_csv(database_path, export_dir, page_type=page_type)
        st.success(f"Exported: {csv_path}")

    playable_rows = [row for row in rows if _audio_path(row, audio_column).is_file()]
    if playable_rows:
        labels = [_audio_label(row, audio_column) for row in playable_rows]
        selected_label = st.selectbox("Playback history audio", labels, key=f"{page_type}_playback_select")
        selected_index = labels.index(selected_label)
        selected_audio = _audio_path(playable_rows[selected_index], audio_column)
        st.audio(str(selected_audio))
    else:
        st.caption("No playable audio files found in history.")


def _with_select_flag(row: dict[str, object]) -> dict[str, object]:
    compact = {
        "select": False,
        "id": row["id"],
        "created_at": row["created_at"],
        "model_name": row["model_name"],
        "status": row["status"],
        "latency_seconds": row["latency_seconds"],
        "audio_duration_seconds": row["audio_duration_seconds"],
        "realtime_factor": row["realtime_factor"],
        "wer": row["wer"],
        "cer": row["cer"],
        "input_text": row["input_text"],
        "transcript": row["transcript"],
        "output_audio_path": row["output_audio_path"],
        "input_audio_path": row["input_audio_path"],
        "notes": row["notes"],
        "error_message": row["error_message"],
    }
    return compact


def _delete_associated_audio(rows: list[dict[str, object]], project_root: Path, audio_column: str) -> int:
    deleted = 0
    for row in rows:
        audio_path = _audio_path(row, audio_column)
        if not audio_path.is_file() or not _is_inside_project(audio_path, project_root):
            continue
        audio_path.unlink()
        deleted += 1
    return deleted


def _audio_path(row: dict[str, object], audio_column: str) -> Path:
    value = str(row.get(audio_column) or "")
    return Path(value)


def _audio_label(row: dict[str, object], audio_column: str) -> str:
    path = _audio_path(row, audio_column)
    return f"#{row['id']} {row['model_name']} - {path.name}"


def _is_inside_project(path: Path, project_root: Path) -> bool:
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False
    return True
