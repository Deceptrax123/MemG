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
