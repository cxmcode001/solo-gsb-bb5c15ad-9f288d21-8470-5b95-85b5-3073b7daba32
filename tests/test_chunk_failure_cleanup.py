"""PUT /sessions/{id}/chunks 落位后失败的清理与可重传性。

临时分片 replace 进 parts 目录之后，若 chunks 登记、chunk_events 写入、
bytes_received 累加/认领消费或事务提交失败：
  * 事务回滚——chunks、chunk_events、bytes_received 与认领令牌保持原状；
  * 清理仍在会话锁内，只删除本次刚落位的目标文件与临时文件，不留无登记
    的孤儿分片；
  * 移动前失败不得误删 parts 目录里已存在的同路径文件；
  * 同区间随后重传可以正常登记并完成文件，响应形态不变。
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app import db, storage


def _create(client: TestClient, data: bytes, name: str = "data.bin") -> str:
    resp = client.post(
        "/api/v1/sessions",
        json={
            "filename": name,
            "total_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _put(
    client: TestClient,
    sid: str,
    start: int,
    end: int,
    content: bytes,
    claim: str | None = None,
):
    headers = {"X-Chunk-Claim": claim} if claim is not None else None
    return client.put(
        f"/api/v1/sessions/{sid}/chunks?start={start}&end={end}",
        content=content,
        headers=headers,
    )


def _chunk_rows(sid: str) -> int:
    return db.get_conn().execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE session_id = ?", (sid,)
    ).fetchone()["n"]


def _event_seqs(sid: str) -> list[int]:
    return [
        r["seq"]
        for r in db.get_conn().execute(
            "SELECT seq FROM chunk_events WHERE session_id = ? ORDER BY seq",
            (sid,),
        ).fetchall()
    ]


def _claim_row(sid: str, claim_id: str):
    return db.get_conn().execute(
        "SELECT * FROM chunk_claims WHERE id = ? AND session_id = ?",
        (claim_id, sid),
    ).fetchone()


class InjectedDBError(RuntimeError):
    pass


class FailingConn:
    """连接代理：按谓词在首个匹配的 SQL 上注入故障。"""

    def __init__(self, conn, sql_predicate):
        self._conn = conn
        self._predicate = sql_predicate

    def execute(self, sql, *params):
        if self._predicate(sql):
            raise InjectedDBError(f"injected failure: {sql[:40]}")
        return self._conn.execute(sql, *params)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@contextmanager
def flaky_immediate(sql_predicate, state=None, fail_commit_number=0):
    conn = db.get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield FailingConn(conn, sql_predicate)
    except Exception:
        conn.rollback()
        raise
    else:
        if fail_commit_number:
            state["commits"] += 1
            if state["commits"] == fail_commit_number:
                # 模拟提交阶段失败：先回滚（与 SQLite 提交失败后的连接状态一致），
                # 保证调用方的 except 清理与后续重试都在干净连接上进行。
                conn.rollback()
                raise InjectedDBError("injected commit failure")
        conn.commit()


# 落位后的三个 SQL 失败点：chunks 登记、事件写入、字节数/认领 UPDATE。
_FAIL_AFTER_MOVE_PREDICATES = {
    "insert_chunk": lambda sql: sql.startswith("INSERT INTO chunks"),
    "insert_event": lambda sql: sql.startswith("INSERT INTO chunk_events"),
    "update_session": lambda sql: sql.startswith(
        "UPDATE upload_sessions"
    )
    and "bytes_received" in sql,
}


@pytest.mark.parametrize("failure_point", list(_FAIL_AFTER_MOVE_PREDICATES))
def test_failure_after_file_move_cleans_part_and_is_retryable(
    client: TestClient, monkeypatch, failure_point: str
) -> None:
    data = bytes((i * 17 + 3) % 256 for i in range(3000))
    sid = _create(client, data)

    # 相邻分片先落一片：失败清理不得波及其文件与登记。
    assert _put(client, sid, 2000, 3000, data[2000:3000]).status_code == 200

    monkeypatch.setattr(
        "app.main.db.immediate",
        lambda: flaky_immediate(
            _FAIL_AFTER_MOVE_PREDICATES[failure_point], {"commits": 0}
        ),
    )
    with pytest.raises(InjectedDBError):
        _put(client, sid, 0, 2000, data[:2000])
    monkeypatch.undo()

    # 没有无登记的孤儿分片，也没有残留临时文件；相邻分片原封不动。
    assert not storage.part_path(sid, 0, 2000).exists()
    assert storage.part_path(sid, 2000, 3000).read_bytes() == data[2000:3000]
    assert list(storage.parts_dir(sid).iterdir()) == [storage.part_path(sid, 2000, 3000)]
    assert list(storage.tmp_dir().glob("*.part")) == []

    # 事务已回滚：chunks / chunk_events / bytes_received 全部保持失败前原状。
    assert _chunk_rows(sid) == 1
    assert _event_seqs(sid) == [1]
    status = client.get(f"/api/v1/sessions/{sid}").json()
    assert status["bytes_received"] == 1000
    assert {"start": 0, "end": 2000} in status["missing_ranges"]

    # 同区间重传正常登记并完成文件；失败的那次不占事件序号。
    r = _put(client, sid, 0, 2000, data[:2000])
    assert r.status_code == 200, r.text
    assert r.json()["received"] == {"start": 0, "end": 2000}
    assert r.json()["bytes_received"] == 3000
    assert r.json()["fully_received"] is True
    assert storage.part_path(sid, 0, 2000).read_bytes() == data[:2000]
    assert _event_seqs(sid) == [1, 2]

    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


def test_commit_failure_after_file_move_cleans_part_and_is_retryable(
    client: TestClient, monkeypatch
) -> None:
    data = b"c" * 2500
    sid = _create(client, data)

    # 第 2 个 IMMEDIATE 事务即 _commit_chunk 的登记事务提交时失败
    #（第 1 个是 _prepare_chunk 的 TTL 滑动）。
    state = {"commits": 0}
    monkeypatch.setattr(
        "app.main.db.immediate",
        lambda: flaky_immediate(lambda sql: False, state, fail_commit_number=2),
    )
    with pytest.raises(InjectedDBError):
        _put(client, sid, 0, 2500, data)
    monkeypatch.undo()

    assert not storage.part_path(sid, 0, 2500).exists()
    assert not list(storage.parts_dir(sid).iterdir())
    assert list(storage.tmp_dir().glob("*.part")) == []
    assert _chunk_rows(sid) == 0
    assert _event_seqs(sid) == []
    assert client.get(f"/api/v1/sessions/{sid}").json()["bytes_received"] == 0

    r = _put(client, sid, 0, 2500, data)
    assert r.status_code == 200, r.text
    assert r.json()["fully_received"] is True
    assert _event_seqs(sid) == [1]
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


def test_failure_before_move_keeps_existing_same_path_file(
    client: TestClient, monkeypatch
) -> None:
    data = b"m" * 1000
    sid = _create(client, data)

    # 模拟 parts 目录里已存在同路径文件（历史遗留的未登记分片）。
    sentinel = storage.part_path(sid, 0, 1000)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"pre-existing-content")

    # 在 replace 之前（首个事务内 SELECT 会话时）注入故障：
    # 清理只能删临时文件，绝不能动这个已存在的同路径文件。
    monkeypatch.setattr(
        "app.main.db.immediate",
        lambda: flaky_immediate(
            lambda sql: sql.startswith("SELECT * FROM upload_sessions"),
            {"commits": 0},
        ),
    )
    with pytest.raises(InjectedDBError):
        _put(client, sid, 0, 1000, data)
    monkeypatch.undo()

    assert sentinel.read_bytes() == b"pre-existing-content"
    assert list(storage.tmp_dir().glob("*.part")) == []
    assert _chunk_rows(sid) == 0
    assert client.get(f"/api/v1/sessions/{sid}").json()["bytes_received"] == 0

    # 同区间重传：replace 覆盖旧内容后正常登记完成。
    r = _put(client, sid, 0, 1000, data)
    assert r.status_code == 200, r.text
    assert sentinel.read_bytes() == data
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data


def test_failure_after_move_keeps_claim_token_for_retry(
    client: TestClient, monkeypatch
) -> None:
    data = bytes((i * 29 + 11) % 256 for i in range(1500))
    sid = _create(client, data)

    claimed = client.post(
        f"/api/v1/sessions/{sid}/chunk-claims",
        json={"start": 0, "end": 1500},
    )
    assert claimed.status_code == 201, claimed.text
    claim_id = claimed.json()["claim_id"]
    token = claimed.json()["claim"]

    monkeypatch.setattr(
        "app.main.db.immediate",
        lambda: flaky_immediate(
            _FAIL_AFTER_MOVE_PREDICATES["insert_event"], {"commits": 0}
        ),
    )
    with pytest.raises(InjectedDBError):
        _put(client, sid, 0, 1500, data, claim=token)
    monkeypatch.undo()

    # 落位后失败：令牌未被消费（DELETE 随事务回滚），落地文件已清理。
    assert _claim_row(sid, claim_id) is not None
    assert not storage.part_path(sid, 0, 1500).exists()
    assert _chunk_rows(sid) == 0
    assert client.get(f"/api/v1/sessions/{sid}").json()["bytes_received"] == 0

    # 同一令牌重传被接受并消费，随后完成。
    r = _put(client, sid, 0, 1500, data, claim=token)
    assert r.status_code == 200, r.text
    assert _claim_row(sid, claim_id) is None
    assert r.json()["fully_received"] is True
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data
