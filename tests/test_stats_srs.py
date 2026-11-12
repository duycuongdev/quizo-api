from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.srs import UserQuestionSRS
from app.services.srs_service import compute_sm2_next_state


def test_sm2_algorithm_progression_and_reset() -> None:
    now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)

    # First correct: rep=1, interval=1
    reps, interval, ef, next_dt = compute_sm2_next_state(0, 0, 2.5, True, now=now)
    assert reps == 1
    assert interval == 1
    assert ef == 2.5
    assert next_dt == now + timedelta(days=1)

    # Second correct: rep=2, interval=6
    reps, interval, ef, next_dt = compute_sm2_next_state(reps, interval, ef, True, now=now)
    assert reps == 2
    assert interval == 6
    assert next_dt == now + timedelta(days=6)

    # Third correct: rep=3, interval=15
    reps, interval, ef, _ = compute_sm2_next_state(reps, interval, ef, True, now=now)
    assert reps == 3
    assert interval == 15

    # Incorrect answer resets repetitions to 0 and interval to 1
    reps, interval, ef, _ = compute_sm2_next_state(reps, interval, ef, False, now=now)
    assert reps == 0
    assert interval == 1
    assert ef >= 1.3


@pytest.mark.asyncio
async def test_stats_overview_and_srs_due_review(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    reg = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "srs@example.com",
            "password": "StrongPassword123",
            "full_name": "SRS Learner",
        },
    )
    assert reg.status_code == 201
    user_id = UUID(reg.json()["user"]["id"])
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Initial overview is empty
    empty_stats = await client.get("/api/v1/stats/overview", headers=headers)
    assert empty_stats.status_code == 200
    assert empty_stats.json()["total_attempts"] == 0
    assert empty_stats.json()["current_streak_days"] == 0
    assert len(empty_stats.json()["category_breakdown"]) == 8
    assert len(empty_stats.json()["daily_activity"]) == 30

    # Take & submit a quiz
    quizzes = (await client.get("/api/v1/quizzes?category_slug=machine-learning")).json()
    quiz_id = quizzes[0]["id"]
    take = (await client.get(f"/api/v1/quizzes/{quiz_id}")).json()

    bank = (await client.get("/api/v1/questions?category_slug=machine-learning", headers=headers)).json()
    correct_map = {
        q["id"]: next(o["id"] for o in q["options"] if o["is_correct"])
        for q in bank["items"]
    }

    answers = [
        {
            "question_id": q["id"],
            "selected_option_id": correct_map[q["id"]]
            if idx < 4
            else next(o["id"] for o in q["options"] if o["id"] != correct_map[q["id"]]),
        }
        for idx, q in enumerate(take["questions"])
    ]
    sub_res = await client.post(
        f"/api/v1/quizzes/{quiz_id}/submit",
        headers=headers,
        json={"answers": answers, "time_spent_seconds": 90},
    )
    assert sub_res.status_code == 200

    stats = (await client.get("/api/v1/stats/overview", headers=headers)).json()
    assert stats["total_attempts"] == 1
    assert stats["total_questions_answered"] == 6
    assert stats["correct_answers"] == 4
    assert stats["overall_accuracy"] == pytest.approx(66.67, abs=0.1)
    assert stats["current_streak_days"] == 1
    assert stats["total_time_spent_seconds"] == 90
    assert len(stats["accuracy_trend"]) == 1

    # Make 2 SRS items due now
    due_q_ids = [UUID(take["questions"][0]["id"]), UUID(take["questions"][1]["id"])]
    await db_session.execute(
        update(UserQuestionSRS)
        .where(
            UserQuestionSRS.user_id == user_id,
            UserQuestionSRS.question_id.in_(due_q_ids),
        )
        .values(next_review_at=datetime.now(UTC) - timedelta(hours=1))
    )
    await db_session.commit()

    due_res = await client.get("/api/v1/stats/srs/due", headers=headers)
    assert due_res.status_code == 200
    due_data = due_res.json()
    assert due_data["due_count"] == 2
    assert len(due_data["items"]) == 2
    # Anti-cheat: no is_correct or explanation in due items
    for item in due_data["items"]:
        assert "explanation" not in item
        for opt in item["options"]:
            assert "is_correct" not in opt

    # Review first due question with correct answer
    first_due_q_id = due_data["items"][0]["question_id"]
    rev_res = await client.post(
        "/api/v1/stats/srs/review",
        headers=headers,
        json={
            "question_id": first_due_q_id,
            "selected_option_id": correct_map[first_due_q_id],
        },
    )
    assert rev_res.status_code == 200
    rev_data = rev_res.json()
    assert rev_data["is_correct"] is True
    assert rev_data["interval_days"] >= 1
    assert "explanation" in rev_data

    due_after = (await client.get("/api/v1/stats/srs/due", headers=headers)).json()
    assert due_after["due_count"] == 1
