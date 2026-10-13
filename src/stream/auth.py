"""Authentication and roles for the local stream server (ARCHITECTURE.md 4.10 and 10.4).

Viewers log in with Supabase Auth; the stream server receives `Authorization: Bearer <access token>`.

1. The token is verified locally against the project's JWKS (signature, issuer, audience, expiry, `sub`).
   The keys are cached by PyJWKClient, so no request to Supabase is needed per frame or per connection.
2. The role is NOT in the token (it only carries `sub` and `role=authenticated`). It is read from
   `user_roles` joined to `roles.code` with a REST call made WITH THE USER'S OWN JWT, so row level
   security decides what is visible (a user can read their own rows) and this server holds no
   service-role key. The same request embeds `profiles.is_active`: a deactivated account can still read
   its own role rows, so it is rejected here (403) like the database does. The result is cached for a
   short time (default 60 s), which also bounds how long a revoked role or deactivation takes to apply.

Roles: admin > teacher > viewer. viewer/teacher/admin may see plain streams; only admin sees the overlay.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx
import jwt

logger = logging.getLogger(__name__)

ROLE_ORDER = ("viewer", "teacher", "admin")  # lowest to highest
STREAM_ROLES = frozenset(ROLE_ORDER)
OVERLAY_ROLES = frozenset({"admin"})
ALGORITHMS = ("ES256", "RS256")  # asymmetric only: never accept HS256 or "none"
ROLE_CACHE_SECONDS = 60.0
# user_roles has two foreign keys to profiles (user_id, granted_by): name the one to embed.
ROLES_SELECT = "roles(code),profile:profiles!user_roles_user_id_fkey(is_active)"


class AuthError(Exception):
    """Carries the HTTP status the stream server must answer with (401, 403 or 503)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: str


class SupabaseAuth:
    def __init__(
        self,
        project_url: str,
        supabase_key: str,
        jwks_client=None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 5.0,
        role_cache_seconds: float = ROLE_CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not project_url.startswith(("http://", "https://")):
            raise ValueError("project_url must start with http:// or https://")
        if not supabase_key:
            raise ValueError("the Supabase publishable key is required (set AFORO_SUPABASE_KEY)")
        base = project_url.rstrip("/")
        self._issuer = f"{base}/auth/v1"
        self._roles_url = f"{base}/rest/v1/user_roles"
        self._supabase_key = supabase_key
        self._jwks = jwks_client or jwt.PyJWKClient(f"{self._issuer}/.well-known/jwks.json", cache_keys=True, lifespan=600)
        self._http = http_client or httpx.Client(timeout=timeout_seconds)
        self._role_cache_seconds = role_cache_seconds
        self._clock = clock
        self._role_cache: dict[tuple[str, str], tuple[float, str]] = {}

    def authenticate(self, authorization: str | None) -> AuthenticatedUser:
        """Validate the header and return the user with their role. Raises AuthError(401/403/503)."""
        token = _bearer(authorization)
        user_id = self._verify(token)
        return AuthenticatedUser(user_id=user_id, role=self._role_of(user_id, token))

    def authorize(self, authorization: str | None, allowed_roles: Iterable[str]) -> AuthenticatedUser:
        """authenticate() plus a role check: 403 when the user's role is not in `allowed_roles`."""
        user = self.authenticate(authorization)
        if user.role not in set(allowed_roles):
            raise AuthError(403, "role not allowed for this resource")
        return user

    def _verify(self, token: str) -> str:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(ALGORITHMS),
                audience="authenticated",
                issuer=self._issuer,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
        except jwt.PyJWKClientConnectionError as error:
            logger.error("could not fetch the JWKS (%s)", type(error).__name__)
            raise AuthError(503, "authentication service unavailable") from error
        except jwt.PyJWTError as error:  # bad signature, expired, wrong issuer/audience, malformed, unknown kid
            logger.info("token rejected: %s", type(error).__name__)
            raise AuthError(401, "invalid or expired token") from error
        subject = claims["sub"]
        try:
            uuid.UUID(str(subject))
        except ValueError as error:  # the id goes into a query string: only a UUID is acceptable
            raise AuthError(401, "invalid token subject") from error
        if claims.get("role") != "authenticated":  # reject the anon key and service tokens
            raise AuthError(401, "not a user token")
        return str(subject)

    def _role_of(self, user_id: str, token: str) -> str:
        cache_key = (user_id, token[-16:])
        cached = self._role_cache.get(cache_key)
        if cached is not None and self._clock() < cached[0]:
            return cached[1]
        try:
            response = self._http.get(
                self._roles_url,
                params={"select": ROLES_SELECT, "user_id": f"eq.{user_id}"},
                headers={"apikey": self._supabase_key, "Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as error:
            logger.error("could not read user_roles (%s)", type(error).__name__)
            raise AuthError(503, "role lookup unavailable") from error
        if response.status_code in (401, 403):
            raise AuthError(401, "token rejected by Supabase")
        if response.status_code != 200:
            logger.error("user_roles answered %d", response.status_code)
            raise AuthError(503, "role lookup unavailable")
        try:
            rows = response.json()
        except ValueError:
            raise AuthError(503, "role lookup unavailable") from None
        if not _is_active(rows):
            raise AuthError(403, "account deactivated")
        role = _highest_role(rows)
        if role is None:
            raise AuthError(403, "user has no role")
        self._role_cache[cache_key] = (self._clock() + self._role_cache_seconds, role)
        return role


def _bearer(header: str | None) -> str:
    if not header:
        raise AuthError(401, "missing token")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise AuthError(401, "expected a Bearer token")
    return token


def _is_active(rows) -> bool:
    """Every row embeds the owner's profile; anything but is_active == True (missing, false) is inactive."""
    if not isinstance(rows, list) or not rows:
        return True  # no role rows: _highest_role answers 403 "user has no role"
    for row in rows:
        profile = row.get("profile") if isinstance(row, dict) else None
        if not isinstance(profile, dict) or profile.get("is_active") is not True:
            return False
    return True


def _highest_role(rows) -> str | None:
    """rows look like [{"roles": {"code": "admin"}}, ...]; unknown codes are ignored."""
    best = -1
    if not isinstance(rows, list):
        return None
    for row in rows:
        roles = row.get("roles") if isinstance(row, dict) else None
        code = roles.get("code") if isinstance(roles, dict) else None
        if code in ROLE_ORDER:
            best = max(best, ROLE_ORDER.index(code))
    return ROLE_ORDER[best] if best >= 0 else None
