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
    key_prefix: str | None = None,
) -> None:
    key_prefix = key_prefix or page_type
    st.subheader(title)
    all_rows = fetch_results(database_path, page_type=page_type, limit=100)
    if not all_rows:
        st.info("No history yet.")
        return

    search_text = st.text_input(
        "Search filename or transcript",
        key=f"{key_prefix}_history_search",
        placeholder="Type part of a filename, transcript, or model name",
    ).strip().casefold()
    rows = [row for row in all_rows if _matches_search(row, audio_column, search_text)]
    if not rows:
        st.info("No history matches the search.")
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
            "input_audio_name": st.column_config.TextColumn("Input filename", width="large"),
            "transcript": st.column_config.TextColumn("Transcript", width="large"),
            "output_audio_path": None,
            "input_audio_path": None,
            "error_message": st.column_config.TextColumn("Error", width="large"),
        },
        key=f"{key_prefix}_history_editor",
    )

    selected_ids = [int(row["id"]) for row in edited.to_dict("records") if row.get("select")]
    selected_rows = [row for row in rows if int(row["id"]) in selected_ids]

    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("Delete selected", disabled=not selected_ids, key=f"{key_prefix}_delete_selected"):
        deleted_files = _delete_associated_audio(selected_rows, project_root, audio_column)
        delete_results(database_path, selected_ids)
        st.success(f"Deleted {len(selected_ids)} row(s) and {deleted_files} audio file(s).")
        st.rerun()

    if action_cols[1].button("Export CSV", key=f"{key_prefix}_export_csv"):
        csv_path = export_results_csv(database_path, export_dir, page_type=page_type)
        st.success(f"Exported: {csv_path}")

    st.markdown("#### Audio and transcript")
    for row in rows:
        audio_path = _audio_path(row, audio_column)
        audio_name = _audio_display_name(row, audio_column)
        model_name = str(row.get("model_name") or "Unknown model")
        audio_label = "Uploaded file" if audio_column == "input_audio_path" else "Generated audio"
        with st.expander(f"#{row['id']} · {audio_name} · {model_name}"):
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"{audio_label}: {audio_name}")
            meta_cols[1].caption(f"Status: {row.get('status') or 'unknown'}")
            meta_cols[2].caption(f"Created: {row.get('created_at') or 'unknown'}")

            if page_type == "tts" and str(row.get("input_text") or "").strip():
                st.markdown("**Input text**")
                st.write(str(row["input_text"]))

            st.markdown("**Transcript**")
            transcript = str(row.get("transcript") or "").strip()
            if transcript:
                st.write(transcript)
            elif row.get("error_message"):
                st.error(str(row["error_message"]))
            else:
                st.caption("No transcript recorded.")

            if audio_path and audio_path.is_file():
                st.audio(str(audio_path))
            elif audio_path:
                st.warning(f"Audio file not found: {audio_path}")
            else:
                st.caption("No audio file recorded.")


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
        "input_audio_name": row["input_audio_name"],
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
        if not audio_path or not audio_path.is_file() or not _is_inside_project(audio_path, project_root):
            continue
        audio_path.unlink()
        deleted += 1
    return deleted


def _audio_path(row: dict[str, object], audio_column: str) -> Path | None:
    value = str(row.get(audio_column) or "").strip()
    return Path(value) if value else None


def _audio_display_name(row: dict[str, object], audio_column: str) -> str:
    if audio_column == "input_audio_path":
        stored_name = str(row.get("input_audio_name") or "").strip()
        if stored_name:
            return stored_name
    path = _audio_path(row, audio_column)
    return path.name if path else "No audio file"


def _matches_search(row: dict[str, object], audio_column: str, search_text: str) -> bool:
    if not search_text:
        return True
    searchable = " ".join(
        (
            _audio_display_name(row, audio_column),
            str(row.get("transcript") or ""),
            str(row.get("model_name") or ""),
        )
    ).casefold()
    return search_text in searchable


def _is_inside_project(path: Path, project_root: Path) -> bool:
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return False
    return True
