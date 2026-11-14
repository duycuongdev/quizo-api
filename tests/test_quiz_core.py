import pytest
from httpx import AsyncClient


async def _register_and_headers(client: AsyncClient, email: str, name: str) -> dict[str, str]:
    res = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPassword123", "full_name": name},
    )
    assert res.status_code == 201
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_categories_seeded_with_8_modules(client: AsyncClient) -> None:
    res = await client.get("/api/v1/categories")
    assert res.status_code == 200
    cats = res.json()
    assert len(cats) == 8
    slugs = {c["slug"] for c in cats}
    assert slugs == {
        "python-fundamentals",
        "math-for-ai",
        "machine-learning",
        "deep-learning",
        "nlp-llm",
        "computer-vision",
        "mlops",
        "ai-agents",
    }
    for cat in cats:
        assert cat["question_count"] >= 6

    detail_res = await client.get("/api/v1/categories/nlp-llm")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["slug"] == "nlp-llm"
    assert len(detail["quizzes"]) >= 1

    not_found = await client.get("/api/v1/categories/non-existent")
    assert not_found.status_code == 404

    # Test creating custom category
    create_res = await client.post(
        "/api/v1/categories",
        json={"name": "Kiến trúc Transformer & LLM", "description": "Tùy chỉnh"},
    )
    assert create_res.status_code == 201
    custom_cat = create_res.json()
    assert custom_cat["name"] == "Kiến trúc Transformer & LLM"
    assert "kien-truc-transformer" in custom_cat["slug"]


@pytest.mark.asyncio
async def test_question_crud_bulk_and_idor_protection(client: AsyncClient) -> None:
    headers_a = await _register_and_headers(client, "usera@example.com", "User A")
    headers_b = await _register_and_headers(client, "userb@example.com", "User B")

    # User A creates a private manual question
    create_res = await client.post(
        "/api/v1/questions",
        headers=headers_a,
        json={
            "category_slug": "ai-agents",
            "content": "What does MCP stand for in AI Agent architectures?",
            "code_snippet": "mcp_client.call_tool('search', {'q': 'rag'})",
            "code_language": "python",
            "explanation": "Model Context Protocol standardizes LLM tool/resource connections.",
            "difficulty": "easy",
            "is_public": False,
            "options": [
                {"content": "Model Context Protocol", "is_correct": True},
                {"content": "Multi Compute Pipeline", "is_correct": False},
                {"content": "Memory Cache Proxy", "is_correct": False},
                {"content": "Model Compression Plan", "is_correct": False},
            ],
        },
    )
    assert create_res.status_code == 201
    q_data = create_res.json()
    q_id = q_data["id"]
    assert q_data["is_owner"] is True

    # User B MUST NOT see User A's private question
    list_b = await client.get("/api/v1/questions?search=Model+Context+Protocol+standardizes", headers=headers_b)
    assert list_b.status_code == 200
    assert all(item["id"] != q_id for item in list_b.json()["items"])

    # User B MUST NOT update or delete User A's question (IDOR)
    put_b = await client.put(
        f"/api/v1/questions/{q_id}",
        headers=headers_b,
        json={"content": "Hacked question content"},
    )
    assert put_b.status_code == 403

    del_b = await client.delete(f"/api/v1/questions/{q_id}", headers=headers_b)
    assert del_b.status_code == 403

    # User B MUST NOT create a quiz referencing User A's private question
    quiz_b = await client.post(
        "/api/v1/quizzes",
        headers=headers_b,
        json={"title": "Steal Q", "question_ids": [q_id]},
    )
    assert quiz_b.status_code == 403

    # User A cannot edit a seeded system question (user_id is None)
    seed_list = await client.get("/api/v1/questions?source=seed&limit=1", headers=headers_a)
    seed_q_id = seed_list.json()["items"][0]["id"]
    edit_seed = await client.put(
        f"/api/v1/questions/{seed_q_id}",
        headers=headers_a,
        json={"content": "Modified seed question"},
    )
    assert edit_seed.status_code == 403

    # User A updates & deletes own question
    put_a = await client.put(
        f"/api/v1/questions/{q_id}",
        headers=headers_a,
        json={"difficulty": "medium"},
    )
    assert put_a.status_code == 200
    assert put_a.json()["difficulty"] == "medium"

    # Bulk create by User A
    bulk_res = await client.post(
        "/api/v1/questions/bulk",
        headers=headers_a,
        json={
            "questions": [
                {
                    "category_slug": "mlops",
                    "content": "Bulk Q1: What is PagedAttention used for?",
                    "explanation": "Efficient KV cache memory management in vLLM.",
                    "difficulty": "medium",
                    "options": [
                        {"content": "KV Cache paging", "is_correct": True},
                        {"content": "Image resizing", "is_correct": False},
                    ],
                },
                {
                    "category_slug": "mlops",
                    "content": "Bulk Q2: What is Training-Serving Skew?",
                    "explanation": "Mismatch between offline and online feature computation.",
                    "difficulty": "easy",
                    "options": [
                        {"content": "Feature mismatch between train and serve", "is_correct": True},
                        {"content": "GPU clock drift", "is_correct": False},
                    ],
                },
            ]
        },
    )
    assert bulk_res.status_code == 201
    assert bulk_res.json()["created_count"] == 2

    del_a = await client.delete(f"/api/v1/questions/{q_id}", headers=headers_a)
    assert del_a.status_code == 200


@pytest.mark.asyncio
async def test_quiz_anti_cheat_grading_and_attempt_idor(client: AsyncClient) -> None:
    headers_a = await _register_and_headers(client, "quiza@example.com", "Quiz A")
    headers_b = await _register_and_headers(client, "quizb@example.com", "Quiz B")

    # Fetch public quizzes
    quizzes_res = await client.get("/api/v1/quizzes?category_slug=python-fundamentals")
    assert quizzes_res.status_code == 200
    public_quiz_id = quizzes_res.json()[0]["id"]

    # Anti-cheat: GET /quizzes/{id} MUST NOT expose is_correct or explanation
    take_res = await client.get(f"/api/v1/quizzes/{public_quiz_id}")
    assert take_res.status_code == 200
    take_data = take_res.json()
    assert len(take_data["questions"]) == 6
    for q in take_data["questions"]:
        assert "explanation" not in q
        for opt in q["options"]:
            assert "is_correct" not in opt

    # Also test private quiz IDOR
    priv_quiz_res = await client.post(
        "/api/v1/quizzes",
        headers=headers_a,
        json={
            "title": "A Private Quiz",
            "category_slug": "deep-learning",
            "question_count": 3,
            "is_public": False,
        },
    )
    assert priv_quiz_res.status_code == 201
    priv_quiz_id = priv_quiz_res.json()["id"]

    assert (await client.get(f"/api/v1/quizzes/{priv_quiz_id}")).status_code == 401
    assert (await client.get(f"/api/v1/quizzes/{priv_quiz_id}", headers=headers_b)).status_code == 403

    # Look up real answers via question bank only to simulate User A picking answers
    q_bank = await client.get("/api/v1/questions?category_slug=python-fundamentals", headers=headers_a)
    correct_map = {
        item["id"]: next(o["id"] for o in item["options"] if o["is_correct"])
        for item in q_bank.json()["items"]
    }

    # Tampered option ID from another question must be rejected
    first_q = take_data["questions"][0]
    second_q_opt = take_data["questions"][1]["options"][0]["id"]
    tampered_res = await client.post(
        f"/api/v1/quizzes/{public_quiz_id}/submit",
        headers=headers_a,
        json={
            "answers": [{"question_id": first_q["id"], "selected_option_id": second_q_opt}],
            "time_spent_seconds": 45,
        },
    )
    assert tampered_res.status_code == 400

    # Submit valid answers (first 3 correct, rest wrong)
    answers = []
    for idx, q in enumerate(take_data["questions"]):
        corr_id = correct_map[q["id"]]
        if idx < 3:
            chosen = corr_id
        else:
            chosen = next(o["id"] for o in q["options"] if o["id"] != corr_id)
        answers.append({"question_id": q["id"], "selected_option_id": chosen})

    submit_res = await client.post(
        f"/api/v1/quizzes/{public_quiz_id}/submit",
        headers=headers_a,
        json={"answers": answers, "time_spent_seconds": 120},
    )
    assert submit_res.status_code == 200
    result = submit_res.json()
    assert result["score"] == 3
    assert result["total_questions"] == 6
    assert result["percentage"] == 50.0
    assert len(result["results"]) == 6
    assert all("explanation" in r and r["correct_option_id"] for r in result["results"])

    attempt_id = result["attempt_id"]

    # Attempt history for User A vs User B
    hist_a = await client.get("/api/v1/quizzes/attempts/history", headers=headers_a)
    assert hist_a.status_code == 200
    assert hist_a.json()["total"] == 1

    # IDOR check on attempt detail
    detail_b = await client.get(f"/api/v1/quizzes/attempts/{attempt_id}", headers=headers_b)
    assert detail_b.status_code == 403

    detail_a = await client.get(f"/api/v1/quizzes/attempts/{attempt_id}", headers=headers_a)
    assert detail_a.status_code == 200
    assert detail_a.json()["attempt_id"] == attempt_id
