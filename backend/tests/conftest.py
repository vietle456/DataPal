import os
import sys
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

# ---------------------------------------------------------------------------
# Must be set BEFORE any `app.*` import — database.py reads DATABASE_URL at
# module level. load_dotenv() uses override=False, so these values win.
# ---------------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app
from app.models.artifact import Artifact
from app.models.base import Base
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.upload import Upload
from app.models.user import User

# Ensure `backend` directory is included in sys.path when running tests
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


# Pre-computed bcrypt hash for "password123" — avoids runtime hashing which
# triggers a passlib/bcrypt version incompatibility in this environment.
_HASHED_PW = "$2b$12$voPMIj2H/Q4CKL/zbMiGNe/elSVh6rpNZOkEpHyRbZIUc5mHOjGSq"

# ---------------------------------------------------------------------------
# In-memory SQLite engine shared across the test session
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
    poolclass=StaticPool,  # one shared connection so concurrent sessions see the same :memory: DB
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_test_tables():
    """Create all DB tables once per test session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional test session that is rolled back after each test."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient wired to the FastAPI app with the test DB session."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


async def create_user(
    db: AsyncSession, username: str = "testuser", password: str = "password123"
) -> User:
    user = User(id=str(uuid.uuid4()), username=username, password=_HASHED_PW)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


def auth_headers(user_id: str) -> dict[str, str]:
    token = create_access_token(subject=user_id)
    return {"Authorization": f"Bearer {token}"}


async def create_conversation(
    db: AsyncSession, user_id: str, title: str = "Test Conv"
) -> Conversation:
    conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def create_message(
    db: AsyncSession, conversation_id: str, role: str = "user", content: str = "Hello"
) -> Message:
    msg = Message(id=str(uuid.uuid4()), conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def create_upload(
    db: AsyncSession,
    conversation_id: str,
    name: str = "file.csv",
    storage_path: str = "/tmp/file.csv",
) -> Upload:
    upload = Upload(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        name=name,
        file_type="text/csv",
        storage_path=storage_path,
        is_selected=False,
    )
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload


async def create_artifact(
    db: AsyncSession,
    conversation_id: str,
    name: str = "result.json",
    file_type: str = "application/json",
) -> Artifact:
    artifact = Artifact(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        name=name,
        file_type=file_type,
    )
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)
    return artifact


@pytest.fixture(autouse=True)
def patch_message_session_factory(monkeypatch):
    """Route the streaming reply's independent session to the test database."""
    monkeypatch.setattr("app.api.v1.message.AsyncSessionLocal", TestSessionLocal)
