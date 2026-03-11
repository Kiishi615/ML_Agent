from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import ForeignKey, String, Text, create_engine
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship)

from config import load_config

AppConfig= load_config()


DB__DIR = AppConfig.database.directory
DB_NAME = AppConfig.database.name

DATABASE_DIR = Path(DB__DIR)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = AppConfig.database.connection_string

engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass

def utc_now():
    return datetime.now(timezone.utc)



class Dataset(Base):
    __tablename__ = "datasets"  

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(200), default="Untitled")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan",
        order_by="DatasetVersion.version_number"
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, title='{self.filename}')>"

class DatasetVersion(Base):
    __tablename__ = "dataset_versions" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version_number: Mapped[int] = mapped_column(nullable= False)
    row_count: Mapped[int] = mapped_column(nullable= False)
    column_count: Mapped[int] = mapped_column(nullable= False)
    columns_json: Mapped[str] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(default=utc_now)

    dataset: Mapped["Dataset"] = relationship(
        back_populates="versions",
    )
    sessions: Mapped[list["AnalysisSession"]] = relationship(
        back_populates="dataset_version",
    )

    def __repr__(self) -> str:
        return f"<DatasetVersion(id={self.id},  version_number = {self.version_number})>"

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    user_request: Mapped[str] = mapped_column(Text, nullable= True)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    completed_at: Mapped[datetime] = mapped_column(default=utc_now, nullable= True)
    total_tokens: Mapped[int] = mapped_column(default=0)
    total_cost: Mapped[float] = mapped_column(default= 0.0)
    
    dataset_version: Mapped["DatasetVersion"] = relationship(
        back_populates="sessions",
    )

    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="session"
    )
    column_analyses : Mapped[list["ColumnAnalysis"]] = relationship(
        back_populates="session"
    )

    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id},  completed_at = {self.completed_at})>"

class ToolCall(Base):
    __tablename__ = "tool_calls" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("analysis_sessions.id"))
    tool_name: Mapped[str] = mapped_column(String(100))
    tool_input: Mapped[str] = mapped_column(Text)
    tool_output: Mapped[str] = mapped_column(Text)
    called_at: Mapped[datetime] = mapped_column(default=utc_now, nullable= True)
    session: Mapped["AnalysisSession"] = relationship(back_populates="tool_calls")
    def __repr__(self) -> str:
        return f"<ToolCall(id={self.id},  tool_name = {self.tool_name})>"

class ColumnAnalysis(Base):
    __tablename__ = "column_analyses" 

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("analysis_sessions.id"))
    column_name: Mapped[str] = mapped_column(String(200))
    dtype: Mapped[str] = mapped_column(String(50))
    null_count: Mapped[int | None] = mapped_column()
    unique_count: Mapped[int | None] = mapped_column()
    mean_value: Mapped[float | None] = mapped_column()
    has_outliers: Mapped[bool] = mapped_column(default= False)
    
    session: Mapped["AnalysisSession"] = relationship(back_populates="column_analyses")
    def __repr__(self) -> str:
        return f"<ColumnAnalysis(id={self.id},  column_name = {self.column_name})>"

def create_tables() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)

def get_session() -> Session:
    return Session(bind=engine)



def create_or_get_dataset(filename: str = "Untitled") -> Dataset:
    session = get_session()
    try:
        dataset = session.query(Dataset).filter(Dataset.filename == filename).first()
        if dataset:
            return dataset
        else:
            dataset = Dataset(filename=filename)
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_version(dataset_id: int, file_hash: str, version_number: int, row_count: int, column_count: int, columns_json: str) -> DatasetVersion:
    """Creates a new version record for an uploaded file."""
    session = get_session()
    try:
        version = DatasetVersion(
            dataset_id=dataset_id,
            file_hash=file_hash,
            version_number=version_number,
            row_count=row_count,
            column_count=column_count,
            columns_json=columns_json
        )
        session.add(version)
        session.commit()
        session.refresh(version)
        return version
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def start_session(version_id: int, user_request: str) -> AnalysisSession:
    """Starts a new analysis chat session."""
    session = get_session()
    try:
        new_session = AnalysisSession(
            version_id=version_id,
            user_request=user_request
        )
        session.add(new_session)
        session.commit()
        session.refresh(new_session)
        return new_session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_tool_call(session_id: int, tool_name: str, tool_input: str, tool_output: str) -> ToolCall:
    """Records exactly what the agent did."""
    session = get_session()
    try:
        call = ToolCall(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output
        )
        session.add(call)
        session.commit()
        session.refresh(call)
        return call
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def complete_session(session_id: int, total_tokens: int, total_cost: float) -> None:
    session = get_session()
    try:
        analysis_session = session.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if analysis_session:
            analysis_session.completed_at = utc_now()
            analysis_session.total_tokens = total_tokens
            analysis_session.total_cost = total_cost
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# # ── READ ─────────────────────────────────────────────────

# def get_all_parents() -> list[Parent]:
#     """Get all active parents, newest first."""
#     session = get_session()
#     try:
#         return (
#             session.query(Parent)
#             .filter(Parent.is_active == True)
#             .order_by(Parent.created_at.desc())
#             .all()
#         )
#     finally:
#         session.close()


# def get_parent_by_id(parent_id: int) -> Parent | None:
#     """Get one parent by ID, or None if not found."""
#     session = get_session()
#     try:
#         return (
#             session.query(Parent)
#             .filter(Parent.id == parent_id)
#             .first()
#         )
#     finally:
#         session.close()


# # ── UPDATE ───────────────────────────────────────────────

# def update_parent_title(parent_id: int, new_title: str) -> bool:
#     """Update a parent's title. Returns True if found."""
#     session = get_session()
#     try:
#         parent = (
#             session.query(Parent)
#             .filter(Parent.id == parent_id)
#             .first()
#         )
#         if not parent:
#             return False
#         parent.title = new_title
#         session.commit()
#         return True
#     except Exception:
#         session.rollback()
#         raise
#     finally:
#         session.close()


# # ── DELETE ───────────────────────────────────────────────

# def delete_parent(parent_id: int) -> bool:
#     """Soft delete a parent. Returns True if found."""
#     session = get_session()
#     try:
#         parent = (
#             session.query(Parent)
#             .filter(Parent.id == parent_id)
#             .first()
#         )
#         if not parent:
#             return False
#         parent.is_active = False
#         session.commit()
#         return True
#     except Exception:
#         session.rollback()
#         raise
#     finally:
#         session.close()


# # ═══════════════════════════════════════════════════════════
# # TESTS — DELETE THIS SECTION WHEN YOU'RE DONE TESTING
# # ═══════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import os

#     # Fresh start
#     if DATABASE_PATH.exists():
#         os.remove(DATABASE_PATH)

#     create_tables()
#     print("Tables created")

#     # Create
#     p = create_parent("Test Parent")
#     print(f"Created: {p}")

#     c1 = add_child(p.id, "First child")
#     c2 = add_child(p.id, "Second child")
#     print(f"Added: {c1}")
#     print(f"Added: {c2}")

#     # Read
#     all_parents = get_all_parents()
#     print(f"All parents: {len(all_parents)}")

#     found = get_parent_by_id(p.id)
#     print(f"Found by ID: {found}")

#     # Update
#     update_parent_title(p.id, "Updated Title")
#     found = get_parent_by_id(p.id)
#     print(f"After update: {found}")

#     # Delete
#     delete_parent(p.id)
#     all_parents = get_all_parents()
#     print(f"After delete: {len(all_parents)} active parents")

#     print("\nALL TESTS PASSED")