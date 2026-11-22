import os

from trbench.provenance import experiment_id, file_sha256, provenance


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
    assert {"git_commit", "git_dirty", "python", "platform", "cpu", "ram_gb", "kernel"} <= p.keys()


def test_file_sha256_same_size_same_second_change(tmp_path):
    """Ayni boyut, ayni saniye icinde degisen dosya: nanosaniye damgasi yeni hash'i zorlar."""
    f = tmp_path / "w.gguf"
    f.write_bytes(b"abc")
    base = 1_700_000_000 * 10**9
    os.utime(f, ns=(base, base + 100))
    h = file_sha256(f)
    f.write_bytes(b"abd")
    os.utime(f, ns=(base, base + 200))           # int(saniye) ayni kaldi
    assert file_sha256(f) != h


def test_experiment_id_stable_and_sensitive():
    a = experiment_id({"eval_sha": "x", "weights": "w", "git_commit": "c"})
    assert a == experiment_id({"git_commit": "c", "weights": "w", "eval_sha": "x"})   # sira onemsiz
    assert a != experiment_id({"eval_sha": "y", "weights": "w", "git_commit": "c"})
