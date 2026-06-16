"""SQLite-хранилище истории диалога."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from eva_agent.dialog.models import (
    DialogMeaning,
    DialogMessage,
    MessageRole,
    Session,
    SessionStatus,
)
from eva_agent.domain.plan import DialogStatus
from eva_agent.settings import settings

_STORE: DialogStore | None = None
_STORE_PATH: str | None = None
_STORE_GUARD = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_db_path() -> str:
    return str(Path.cwd() / "var" / "dialog.db")


def _configured_db_path(db_path: str | None = None) -> str:
    if db_path:
        return db_path
    env_path = os.environ.get("EVA_DIALOG_DB", "").strip()
    if env_path:
        return env_path
    if settings.eva_dialog_db.strip():
        return settings.eva_dialog_db.strip()
    return _default_db_path()


def get_store() -> DialogStore:
    """Ленивый singleton, пересоздается при смене EVA_DIALOG_DB."""

    global _STORE, _STORE_PATH
    path = _configured_db_path()
    with _STORE_GUARD:
        if _STORE is not None and path == _STORE_PATH:
            return _STORE
        if _STORE is not None:
            _STORE.close()
        _STORE = DialogStore(path)
        _STORE_PATH = path
        return _STORE


def reset_store() -> None:
    """Закрыть singleton. Используется тестами при подмене пути БД."""

    global _STORE, _STORE_PATH
    with _STORE_GUARD:
        if _STORE is not None:
            _STORE.close()
        _STORE = None
        _STORE_PATH = None


class DialogStore:
    """Потокобезопасный доступ к SQLite через одно соединение и lock."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = _configured_db_path(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def get_or_create_session(self, session_id: str) -> Session:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                return _session_from_row(row)
            created = _now()
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sessions(session_id, created_at, updated_at, status, last_intent, turn)
                    VALUES (?, ?, ?, 'active', NULL, 0)
                    """,
                    (session_id, created, created),
                )
            return Session(
                session_id=session_id,
                created_at=created,
                updated_at=created,
                status="active",
                turn=0,
            )

    def update_session(
        self,
        session_id: str,
        *,
        status: SessionStatus | None = None,
        last_intent: str | None = None,
        bump_turn: bool = False,
    ) -> Session:
        with self._lock:
            current = self.get_or_create_session(session_id)
            updated_at = _now()
            next_turn = current.turn + (1 if bump_turn else 0)
            next_status = status or current.status
            next_intent = last_intent if last_intent is not None else current.last_intent
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?, status = ?, last_intent = ?, turn = ?
                    WHERE session_id = ?
                    """,
                    (updated_at, next_status, next_intent, next_turn, session_id),
                )
            return Session(
                session_id=current.session_id,
                created_at=current.created_at,
                updated_at=updated_at,
                status=next_status,
                last_intent=next_intent,
                turn=next_turn,
            )

    def append_message(self, session_id: str, role: MessageRole, text: str) -> DialogMessage:
        with self._lock:
            self.get_or_create_session(session_id)
            ts = _now()
            with self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO messages(session_id, role, text, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, text, ts),
                )
            return DialogMessage(
                id=cursor.lastrowid,
                session_id=session_id,
                role=role,
                text=text,
                ts=ts,
            )

    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[DialogMessage]:
        with self._lock:
            if limit is None:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
            return [_message_from_row(row) for row in rows]

    def add_snapshot(self, session_id: str, snapshot: DialogMeaning) -> DialogMeaning:
        with self._lock:
            self.get_or_create_session(session_id)
            ts = snapshot.ts or _now()
            todo_list = _json_dumps(snapshot.todo_list)
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO meaning_snapshots(
                        session_id, turn, summary, open_question, accumulated_meaning,
                        reasoning, todo_list, dialog_status, ts
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        snapshot.turn,
                        snapshot.summary,
                        snapshot.open_question,
                        snapshot.accumulated_meaning,
                        snapshot.reasoning,
                        todo_list,
                        snapshot.dialog_status,
                        ts,
                    ),
                )
            return snapshot.model_copy(update={"session_id": session_id, "ts": ts})

    def latest_snapshot(self, session_id: str) -> DialogMeaning | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM meaning_snapshots
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return _snapshot_from_row(row) if row is not None else None

    def list_snapshots(self, session_id: str, *, limit: int | None = None) -> list[DialogMeaning]:
        with self._lock:
            if limit is None:
                rows = self._conn.execute(
                    "SELECT * FROM meaning_snapshots WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM meaning_snapshots
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
            return [_snapshot_from_row(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            with self._conn:
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_intent TEXT NULL,
                        turn INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        ts TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS ix_messages_session
                    ON messages(session_id, id);

                    CREATE TABLE IF NOT EXISTS meaning_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn INTEGER NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        open_question TEXT NULL,
                        accumulated_meaning TEXT NOT NULL DEFAULT '',
                        reasoning TEXT NOT NULL DEFAULT '',
                        todo_list TEXT NULL,
                        dialog_status TEXT NULL,
                        ts TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS ix_snap_session
                    ON meaning_snapshots(session_id, turn);
                    """
                )


def _json_dumps(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else None


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        session_id=str(row["session_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        status=cast(SessionStatus, str(row["status"])),
        last_intent=cast(str | None, row["last_intent"]),
        turn=int(row["turn"]),
    )


def _message_from_row(row: sqlite3.Row) -> DialogMessage:
    return DialogMessage(
        id=int(row["id"]),
        session_id=str(row["session_id"]),
        role=cast(MessageRole, str(row["role"])),
        text=str(row["text"]),
        ts=str(row["ts"]),
    )


def _snapshot_from_row(row: sqlite3.Row) -> DialogMeaning:
    return DialogMeaning(
        session_id=str(row["session_id"]),
        turn=int(row["turn"]),
        summary=str(row["summary"]),
        open_question=cast(str | None, row["open_question"]),
        accumulated_meaning=str(row["accumulated_meaning"]),
        reasoning=str(row["reasoning"]),
        todo_list=_json_loads(row["todo_list"]),
        dialog_status=cast(DialogStatus | None, row["dialog_status"]),
        ts=str(row["ts"]),
    )

