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


def _repo(tmp_path):
    import subprocess
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    for f in ("trbench/x.py", "results/r.jsonl", "README.md"):
        (tmp_path / f).parent.mkdir(exist_ok=True)
        (tmp_path / f).write_text("a\n")
    g("add", ".")
    g("commit", "-qm", "i")
    return tmp_path


def test_results_write_does_not_dirty_runs(tmp_path):
    from trbench.provenance import code_diff_sha, git_commit, require_clean
    root = _repo(tmp_path)
    (root / "results/r.jsonl").write_text("a\nb\n")          # bir onceki kosunun yazdigi satir
    assert git_commit(root)[1] is False
    assert code_diff_sha(root) is None and require_clean(False, root) is None


def test_code_change_blocks_run_unless_allowed(tmp_path):
    import pytest

    from trbench.provenance import code_diff_sha, git_commit, require_clean
    root = _repo(tmp_path)
    (root / "trbench/x.py").write_text("b\n")
    assert git_commit(root)[1] is True
    with pytest.raises(SystemExit):
        require_clean(False, root)
    assert require_clean(True, root) == code_diff_sha(root) is not None


def test_readme_change_is_dirty_but_not_blocking(tmp_path):
    from trbench.provenance import git_commit, require_clean
    root = _repo(tmp_path)
    (root / "README.md").write_text("b\n")
    assert git_commit(root)[1] is True and require_clean(False, root) is None


def test_untracked_code_file_blocks_run(tmp_path):
    import pytest

    from trbench.provenance import code_diff_sha, require_clean
    root = _repo(tmp_path)
    (root / "trbench/yeni.py").write_text("x = 1\n")          # commit edilmemis yeni dosya
    d1 = code_diff_sha(root)
    assert d1 is not None
    with pytest.raises(SystemExit):
        require_clean(False, root)
    (root / "trbench/yeni.py").write_text("x = 2\n")          # icerik degisince ozet de degisir
    assert code_diff_sha(root) != d1


def test_untracked_results_file_does_not_block(tmp_path):
    from trbench.provenance import require_clean
    root = _repo(tmp_path)
    (root / "results/yeni.jsonl").write_text("{}\n")
    assert require_clean(False, root) is None
