from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from geologic_memory.db import LocalGeometricDB


def storage_db(tmp_path):
    db = object.__new__(LocalGeometricDB)
    db.cache_dir = str(tmp_path)
    return db


def test_user_storage_paths_are_isolated(tmp_path):
    db = storage_db(tmp_path)
    alice = Path(db._session_path("conversation-1", "alice"))
    bob = Path(db._session_path("conversation-1", "bob"))

    assert alice == tmp_path / "users" / "alice" / "conversation-1.pkl"
    assert bob == tmp_path / "users" / "bob" / "conversation-1.pkl"
    assert alice != bob


def test_user_storage_rejects_path_traversal(tmp_path):
    db = storage_db(tmp_path)
    with pytest.raises(ValueError, match="user_id"):
        db._session_path("memory", "../other-user")
