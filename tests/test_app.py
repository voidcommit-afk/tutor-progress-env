from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_and_metadata():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    meta = client.get("/metadata")
    assert meta.status_code == 200
    body = meta.json()
    assert "name" in body and "description" in body


def test_session_reset_and_step():
    session = client.get("/session/new")
    assert session.status_code == 200
    session_id = session.json()["session_id"]

    reset = client.post("/reset", json={"session_id": session_id, "task_id": "easy_1"})
    assert reset.status_code == 200
    reset_body = reset.json()
    assert reset_body["task_id"] == "easy_1"
    assert reset_body["session_id"] == session_id

    tool = client.post("/step", json={"session_id": session_id, "type": "tool", "tool_name": "extract_concepts"})
    assert tool.status_code == 200
    tool_body = tool.json()
    assert tool_body["done"] is False
    assert "tool_output" in tool_body["info"]

    final = client.post(
        "/step",
        json={
            "session_id": session_id,
            "type": "final_answer",
            "content": "Summary: confusion.\nDiagnosis: concept gap.\nPlan: targeted practice.\nConstraints: none.",
        },
    )
    assert final.status_code == 200
    final_body = final.json()
    assert final_body["done"] is True
    assert final_body["observation"]["session_id"] == session_id


def test_schema_and_mcp_available():
    schema = client.get("/schema")
    assert schema.status_code == 200
    schema_body = schema.json()
    assert "action" in schema_body and "observation" in schema_body and "state" in schema_body

    mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert mcp.status_code == 200
    mcp_body = mcp.json()
    assert mcp_body["jsonrpc"] == "2.0"
