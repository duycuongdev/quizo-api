from unittest.mock import patch

import jwt
import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.core.rate_limit import limiter


@pytest.mark.asyncio
async def test_register_login_me_and_password_policy(client: AsyncClient) -> None:
    # Weak password rejected
    weak_res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "alice@example.com",
            "password": "weak",
            "full_name": "Alice",
        },
    )
    assert weak_res.status_code == 422

    # Valid registration
    reg_res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "Alice@Example.com",
            "password": "StrongPassword123",
            "full_name": "Alice Nguyen",
        },
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["token_type"] == "bearer"
    assert reg_data["user"]["email"] == "alice@example.com"
    assert reg_data["user"]["auth_provider"] == "local"
    assert "hashed_password" not in reg_data["user"]

    # Duplicate email rejected (case-insensitive)
    dup_res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "alice@example.com",
            "password": "StrongPassword123",
            "full_name": "Alice Two",
        },
    )
    assert dup_res.status_code == 409

    # Wrong password rejected
    bad_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "WrongPassword123"},
    )
    assert bad_login.status_code == 401

    # Login success
    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "StrongPassword123"},
    )
    assert login_res.status_code == 200
    tokens = login_res.json()

    # /me works with access token
    me_res = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "alice@example.com"

    # Refresh token MUST NOT be accepted as access token
    refresh_as_access = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['refresh_token']}"},
    )
    assert refresh_as_access.status_code == 401

    # Unsigned alg=none JWT rejected
    forged = jwt.encode(
        {"sub": reg_data["user"]["id"], "type": "access", "jti": "forged", "exp": 9999999999},
        "",
        algorithm="none",
    )
    forged_res = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {forged}"},
    )
    assert forged_res.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_rotation_and_reuse_revocation(client: AsyncClient) -> None:
    reg_res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "bob@example.com",
            "password": "StrongPassword123",
            "full_name": "Bob Tran",
        },
    )
    assert reg_res.status_code == 201
    r1 = reg_res.json()["refresh_token"]

    # Rotate r1 -> r2
    rot_res = await client.post("/api/v1/auth/refresh", json={"refresh_token": r1})
    assert rot_res.status_code == 200
    r2 = rot_res.json()["refresh_token"]
    a2 = rot_res.json()["access_token"]
    assert r2 != r1

    # Reusing r1 must fail AND revoke r2 (reuse detection)
    reuse_res = await client.post("/api/v1/auth/refresh", json={"refresh_token": r1})
    assert reuse_res.status_code == 401

    r2_after_reuse = await client.post("/api/v1/auth/refresh", json={"refresh_token": r2})
    assert r2_after_reuse.status_code == 401

    # Logout with fresh login revokes token
    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": "StrongPassword123"},
    )
    assert login_res.status_code == 200
    a3 = login_res.json()["access_token"]
    r3 = login_res.json()["refresh_token"]

    logout_res = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": r3},
        headers={"Authorization": f"Bearer {a3}"},
    )
    assert logout_res.status_code == 200

    use_after_logout = await client.post("/api/v1/auth/refresh", json={"refresh_token": r3})
    assert use_after_logout.status_code == 401
    assert a2 != ""


@pytest.mark.asyncio
async def test_google_oauth_verification(client: AsyncClient) -> None:
    fake_claims = {
        "iss": "https://accounts.google.com",
        "sub": "google-sub-12345",
        "email": "guser@example.com",
        "email_verified": True,
        "name": "Google User",
        "picture": "https://example.com/avatar.png",
        "aud": settings.GOOGLE_CLIENT_ID or "test-client-id",
    }

    with patch.object(settings, "GOOGLE_CLIENT_ID", "test-client-id"):
        with patch(
            "app.services.auth_service.google_id_token.verify_oauth2_token",
            return_value=fake_claims,
        ):
            res = await client.post(
                "/api/v1/auth/google",
                json={"id_token": "valid-mock-google-id-token"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["user"]["email"] == "guser@example.com"
            assert data["user"]["auth_provider"] == "google"

        # Unverified email rejected
        unverified_claims = {**fake_claims, "email_verified": False}
        with patch(
            "app.services.auth_service.google_id_token.verify_oauth2_token",
            return_value=unverified_claims,
        ):
            res_unverified = await client.post(
                "/api/v1/auth/google",
                json={"credential": "valid-mock-google-id-token"},
            )
            assert res_unverified.status_code == 401

        # Invalid token raises ValueError
        with patch(
            "app.services.auth_service.google_id_token.verify_oauth2_token",
            side_effect=ValueError("Invalid token"),
        ):
            res_invalid = await client.post(
                "/api/v1/auth/google",
                json={"id_token": "bad-mock-google-id-token"},
            )
            assert res_invalid.status_code == 401


@pytest.mark.asyncio
async def test_auth_rate_limiting(client: AsyncClient) -> None:
    limiter.enabled = True
    limiter.reset()
    try:
        statuses = []
        for _ in range(12):
            res = await client.post(
                "/api/v1/auth/login",
                json={"email": "nobody@example.com", "password": "Password123"},
            )
            statuses.append(res.status_code)
        assert 429 in statuses
    finally:
        limiter.enabled = False
        limiter.reset()
