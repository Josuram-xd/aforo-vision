import time
import uuid

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from src.stream.auth import OVERLAY_ROLES, STREAM_ROLES, AuthError, SupabaseAuth

PROJECT = "https://proj.supabase.test"
ISSUER = f"{PROJECT}/auth/v1"
PUBLISHABLE = "sb_publishable_test-value-123"
USER_ID = str(uuid.uuid4())

PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


class _Jwks:
    """Stands in for PyJWKClient: always hands out the test public key."""

    def get_signing_key_from_jwt(self, token):
        class Key:
            key = PRIVATE_KEY.public_key()

        return Key()


def _token(key=PRIVATE_KEY, **overrides):
    claims = {"sub": USER_ID, "role": "authenticated", "aud": "authenticated", "iss": ISSUER,
              "iat": int(time.time()), "exp": int(time.time()) + 3600}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": "test"})


def _auth(codes=("viewer",), status=200, seen=None, clock=time.monotonic, active=True, **kwargs):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=[{"roles": {"code": c}, "profile": {"is_active": active}} for c in codes])

    return SupabaseAuth(PROJECT, PUBLISHABLE, jwks_client=_Jwks(), clock=clock,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def _bearer(token):
    return f"Bearer {token}"


def test_valid_token_returns_the_user_and_role():
    user = _auth(("teacher",)).authenticate(_bearer(_token()))
    assert (user.user_id, user.role) == (USER_ID, "teacher")


def test_role_lookup_uses_the_users_own_jwt_and_filters_by_sub():
    seen = []
    token = _token()
    _auth(seen=seen).authenticate(_bearer(token))
    request = seen[0]
    assert request.url.path == "/rest/v1/user_roles"
    assert request.url.params["user_id"] == f"eq.{USER_ID}"
    assert request.url.params["select"] == "roles(code),profile:profiles!user_roles_user_id_fkey(is_active)"
    assert request.headers["authorization"] == f"Bearer {token}" and request.headers["apikey"] == PUBLISHABLE


def test_deactivated_account_is_403_even_with_a_role():
    with pytest.raises(AuthError) as info:
        _auth(("admin",), active=False).authenticate(_bearer(_token()))
    assert info.value.status == 403


def test_missing_profile_is_treated_as_deactivated():
    def handler(request):
        return httpx.Response(200, json=[{"roles": {"code": "admin"}}])

    auth = SupabaseAuth(PROJECT, PUBLISHABLE, jwks_client=_Jwks(),
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(AuthError) as info:
        auth.authenticate(_bearer(_token()))
    assert info.value.status == 403


def test_highest_role_wins_and_unknown_codes_are_ignored():
    assert _auth(("viewer", "admin", "hacker")).authenticate(_bearer(_token())).role == "admin"


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Basic abc", "token-without-scheme"])
def test_missing_or_malformed_header_is_401(header):
    with pytest.raises(AuthError) as info:
        _auth().authenticate(header)
    assert info.value.status == 401


def test_garbage_token_is_401():
    with pytest.raises(AuthError) as info:
        _auth().authenticate("Bearer not.a.jwt")
    assert info.value.status == 401


def test_expired_token_is_401():
    with pytest.raises(AuthError) as info:
        _auth().authenticate(_bearer(_token(exp=int(time.time()) - 10)))
    assert info.value.status == 401


def test_token_signed_with_another_key_is_401():
    with pytest.raises(AuthError) as info:
        _auth().authenticate(_bearer(_token(key=OTHER_KEY)))
    assert info.value.status == 401


@pytest.mark.parametrize("overrides", [
    {"iss": "https://evil.test/auth/v1"}, {"aud": "other"}, {"exp": None}, {"sub": None},
    {"sub": "1; drop table"}, {"role": "anon"}, {"role": "service_role"},
])
def test_wrong_claims_are_401(overrides):
    with pytest.raises(AuthError) as info:
        _auth().authenticate(_bearer(_token(**overrides)))
    assert info.value.status == 401


def test_hs256_and_unsigned_tokens_are_refused():
    claims = {"sub": USER_ID, "role": "authenticated", "aud": "authenticated", "iss": ISSUER, "exp": int(time.time()) + 60}
    for token in (jwt.encode(claims, "x" * 64, algorithm="HS256"), jwt.encode(claims, None, algorithm="none")):
        with pytest.raises(AuthError) as info:
            _auth().authenticate(_bearer(token))
        assert info.value.status == 401


def test_user_without_a_role_is_403():
    with pytest.raises(AuthError) as info:
        _auth(()).authenticate(_bearer(_token()))
    assert info.value.status == 403


def test_viewer_cannot_open_the_overlay_but_admin_can():
    token = _bearer(_token())
    assert _auth(("viewer",)).authorize(token, STREAM_ROLES).role == "viewer"
    with pytest.raises(AuthError) as info:
        _auth(("viewer",)).authorize(token, OVERLAY_ROLES)
    assert info.value.status == 403
    with pytest.raises(AuthError):
        _auth(("teacher",)).authorize(token, OVERLAY_ROLES)
    assert _auth(("admin",)).authorize(token, OVERLAY_ROLES).role == "admin"


def test_role_lookup_failures_are_503_and_rejection_is_401():
    def down(request):
        raise httpx.ConnectError("unreachable")

    unreachable = SupabaseAuth(PROJECT, PUBLISHABLE, jwks_client=_Jwks(), http_client=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(AuthError) as info:
        unreachable.authenticate(_bearer(_token()))
    assert info.value.status == 503
    with pytest.raises(AuthError) as info:
        _auth(status=500).authenticate(_bearer(_token()))
    assert info.value.status == 503
    with pytest.raises(AuthError) as info:
        _auth(status=401).authenticate(_bearer(_token()))
    assert info.value.status == 401


def test_roles_are_cached_briefly_then_looked_up_again():
    seen, now = [], [100.0]
    auth = _auth(seen=seen, clock=lambda: now[0], role_cache_seconds=60)
    header = _bearer(_token())
    auth.authenticate(header)
    auth.authenticate(header)
    assert len(seen) == 1
    now[0] += 61
    auth.authenticate(header)
    assert len(seen) == 2


def test_jwks_outage_is_503():
    class Down:
        def get_signing_key_from_jwt(self, token):
            raise jwt.PyJWKClientConnectionError("unreachable")

    auth = SupabaseAuth(PROJECT, PUBLISHABLE, jwks_client=Down())
    with pytest.raises(AuthError) as info:
        auth.authenticate(_bearer(_token()))
    assert info.value.status == 503


def test_tokens_and_keys_are_never_logged(caplog):
    token = _token(exp=int(time.time()) - 10)
    with caplog.at_level("DEBUG"):
        for auth in (_auth(), _auth(status=500)):
            for header in (_bearer(token), _bearer(_token())):
                try:
                    auth.authenticate(header)
                except AuthError:
                    pass
    assert token not in caplog.text and PUBLISHABLE not in caplog.text


def test_constructor_requires_a_key_and_a_url():
    with pytest.raises(ValueError):
        SupabaseAuth(PROJECT, "")
    with pytest.raises(ValueError):
        SupabaseAuth("ftp://x", PUBLISHABLE)
