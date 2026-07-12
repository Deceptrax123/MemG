from types import SimpleNamespace

import pytest

from mcp.api import create_service


def test_status_does_not_load_models(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    status = service.status()
    assert status["loaded"] is False
    assert status["cache_dir"] == str(tmp_path)


def test_configure_maps_public_option_names(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    status = service.configure_memory(weights_path="model.pt", embedding_model="custom/model", top_k=3)
    assert status["weights_path"] == "model.pt"
    assert status["embedding_model"] == "custom/model"


def test_unknown_configuration_is_rejected(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    with pytest.raises(TypeError, match="Unknown configuration"):
        service.configure_memory(typo=True)


def test_ingest_validates_before_loading_models(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="overlap_size"):
        service.ingest_turns([{"role": "user", "content": "hello"}], window_size=2, overlap_size=2)


def test_query_validates_before_loading_models(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="non-empty"):
        service.retrieve_facts(" ")


def test_user_memory_load_clears_previous_graph(monkeypatch, tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))

    class DB:
        nodes_dict = {"old": object()}

        def clear(self):
            self.nodes_dict = {}

        def load_session(self, session_id, user_id=None):
            assert session_id == "shared-memory"
            assert user_id == "alice"
            return False

    service.system = SimpleNamespace(db=DB())
    result = service.load_user_memory("alice")
    assert result["loaded"] is False
    assert result["nodes"] == 0
    assert service.active_user_id == "alice"


def test_user_memory_cannot_be_saved_under_another_user(tmp_path):
    service = create_service(device="cpu", cache_dir=str(tmp_path))
    service.active_user_id = "alice"
    with pytest.raises(RuntimeError, match="Load this user's memory"):
        service.save_user_memory("bob")
