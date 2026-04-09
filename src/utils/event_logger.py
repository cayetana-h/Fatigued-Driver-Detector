from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CsvEventLogger:
    FIELDNAMES = [
        "timestamp_utc",
        "frame_idx",
        "source",
        "state",
        "status",
        "event",
        "detail",
        "face_visible",
        "face_missing_frames",
        "eye_closed_frames",
        "yawn_frames",
        "yawn_count",
        "head_drop_frames",
        "eye_ratio",
        "mouth_ratio",
        "head_drop",
        "reason",
    ]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log(
        self,
        *,
        frame_idx: int,
        source: str,
        state: str,
        status: str,
        event: str,
        detail: str = "",
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        snapshot = snapshot or {}
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_idx": frame_idx,
            "source": source,
            "state": state,
            "status": status,
            "event": event,
            "detail": detail,
            "face_visible": snapshot.get("face_visible", ""),
            "face_missing_frames": snapshot.get("face_missing_frames", ""),
            "eye_closed_frames": snapshot.get("eye_closed_frames", ""),
            "yawn_frames": snapshot.get("yawn_frames", ""),
            "yawn_count": snapshot.get("yawn_count", ""),
            "head_drop_frames": snapshot.get("head_drop_frames", ""),
            "eye_ratio": snapshot.get("eye_ratio", ""),
            "mouth_ratio": snapshot.get("mouth_ratio", ""),
            "head_drop": snapshot.get("head_drop", ""),
            "reason": snapshot.get("reason", ""),
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()