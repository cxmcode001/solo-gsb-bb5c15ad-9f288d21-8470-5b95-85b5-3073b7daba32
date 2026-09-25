"""PUT /chunks 落位后失败的清理回归：

临时分片 replace 进 parts 目录后，chunks 登记 / chunk_events 写入 /
事务提交任一步失败时，必须在会话锁内删除**本次落位**的目标文件；
移动之前的失败不得删除 parts 目录中已存在的文件。失败后 chunks、
chunk_events、bytes_received 与认领令牌全部保持原状，同区间重传
可正常登记并完成文件。
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app import db, main, storage


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


def _session_row(sid: str):
    return db.get_conn().execute(
        "SELECT * FROM upload_sessions WHERE id = ?", (sid,)
    ).fetchone()


def test_failure_after_placement_removes_orphan_and_retry_completes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = bytes((i * 13 + 5) % 256 for i in range(1000))
    sid = _create(client, data)

    # 认领 [0,1000)：失败后令牌必须保持可用，重传仍能消费它。
    claim = client.post(
        f"/api/v1/sessions/{sid}/chunk-claims",
        json={"start": 0, "end": 1000},
    )
    assert claim.status_code == 201, claim.text
    credential = claim.json()["claim"]

    # 模拟“文件已落位、事件写入失败”：chunks INSERT 之后、同一事务内
    # _record_chunk_event 抛错，整个事务回滚。
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated chunk_events write failure")

    monkeypatch.setattr(main, "_record_chunk_event", boom)
    with pytest.raises(RuntimeError, match="simulated chunk_events"):
        client.put(
            f"/api/v1/sessions/{sid}/chunks?start=0&end=1000",
            content=data,
            headers={"X-Chunk-Claim": credential},
        )
    monkeypatch.undo()

    # 磁盘：本次落位的 part 文件与接收临时文件都不得残留（无无登记分片）。
    assert not storage.part_path(sid, 0, 1000).exists()
    assert not list(storage.tmp_dir().glob("recv-*.part"))

    # 数据库：chunks / chunk_events / bytes_received 全部保持原状。
    rows = db.get_conn().execute(
        "SELECT * FROM chunks WHERE session_id = ?", (sid,)
    ).fetchall()
    assert rows == []
    events = db.get_conn().execute(
        "SELECT * FROM chunk_events WHERE session_id = ?", (sid,)
    ).fetchall()
    assert events == []
    row = _session_row(sid)
    assert row["status"] == "uploading"
    assert row["bytes_received"] == 0

    # 认领令牌原样保留，仍可用于同区间重传。
    claim_rows = db.get_conn().execute(
        "SELECT id FROM chunk_claims WHERE session_id = ?", (sid,)
    ).fetchall()
    assert len(claim_rows) == 1

    # 同区间重传正常登记（成功消费令牌）并完成文件。
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=1000",
        content=data,
        headers={"X-Chunk-Claim": credential},
    )
    assert r.status_code == 200, r.text
    assert r.json()["received"] == {"start": 0, "end": 1000}
    assert storage.part_path(sid, 0, 1000).is_file()
    assert (
        db.get_conn()
        .execute("SELECT COUNT(*) AS n FROM chunk_claims WHERE session_id = ?", (sid,))
        .fetchone()["n"]
        == 0
    )

    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    dl = client.get(f"/api/v1/sessions/{sid}/download")
    assert dl.status_code == 200
    assert dl.content == data


def test_failure_before_move_keeps_existing_part_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = bytes((i * 17 + 1) % 256 for i in range(1000))
    sid = _create(client, data)

    # 先正常登记一片 [0,100)。
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=0&end=100", content=data[:100]
    )
    assert r.status_code == 200, r.text

    # 目标区间 [100,200) 的正式路径预先放一个哨兵文件：模拟落位前路径上
    # 已存在的文件，移动前失败绝不能把它删掉。
    final = storage.part_path(sid, 100, 200)
    final.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b"pre-existing-sentinel"
    final.write_bytes(sentinel)

    # _check_chunk_claim 在 _prepare_chunk（第 1 次）通过、在提交事务内
    # replace 之前的复核（第 2 次）抛错——属于“移动前失败”。
    calls = {"n": 0}
    real_check = main._check_chunk_claim

    def fail_inside_tx(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated pre-move check failure")
        return real_check(*args, **kwargs)

    monkeypatch.setattr(main, "_check_chunk_claim", fail_inside_tx)
    with pytest.raises(RuntimeError, match="simulated pre-move"):
        client.put(
            f"/api/v1/sessions/{sid}/chunks?start=100&end=200",
            content=data[100:200],
        )
    monkeypatch.undo()

    # 已登记分片与哨兵文件都原封不动；接收临时文件也已清理。
    assert storage.part_path(sid, 0, 100).read_bytes() == data[:100]
    assert final.read_bytes() == sentinel
    assert not list(storage.tmp_dir().glob("recv-*.part"))

    # 数据库只有此前登记的一片，字节数与事件不变。
    rows = db.get_conn().execute(
        "SELECT start_offset, end_offset FROM chunks WHERE session_id = ? "
        "ORDER BY start_offset",
        (sid,),
    ).fetchall()
    assert [(r["start_offset"], r["end_offset"]) for r in rows] == [(0, 100)]
    row = _session_row(sid)
    assert row["bytes_received"] == 100

    # 移除哨兵后同区间重传可正常登记，会话最终能完成。
    final.unlink()
    r = client.put(
        f"/api/v1/sessions/{sid}/chunks?start=100&end=1000",
        content=data[100:1000],
    )
    assert r.status_code == 200, r.text
    done = client.post(f"/api/v1/sessions/{sid}/complete")
    assert done.status_code == 200, done.text
    assert client.get(f"/api/v1/sessions/{sid}/download").content == data
