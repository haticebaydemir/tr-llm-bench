import pytest


@pytest.fixture(autouse=True)
def _sha_cache_in_tmp(tmp_path, monkeypatch):
    """Testler kullanicinin ~/.cache klasorune hash onbellegi yazmasin (Faz 7q)."""
    monkeypatch.setenv("TRBENCH_CACHE", str(tmp_path / "trbench-cache"))
