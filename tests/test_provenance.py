from trbench.provenance import file_sha256, provenance


def test_file_sha256_cached_and_invalidated(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"abc")
    h = file_sha256(f)
    assert h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert (tmp_path / "w.gguf.sha256").exists() and file_sha256(f) == h
    f.write_bytes(b"abcd")                       # boyut degisti -> yeniden hesap
    assert file_sha256(f) != h
    assert file_sha256(tmp_path / "yok") is None


def test_provenance_fields():
    p = provenance()
    assert {"git_commit", "git_dirty", "python", "platform"} <= p.keys()
