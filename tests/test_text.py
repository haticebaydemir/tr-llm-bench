from trbench.metrics.text import count_words


def test_basic():
    assert count_words("Merhaba dünya") == 2


def test_turkish_apostrophe():
    assert count_words("Ankara'ya gittim.") == 2


def test_curly_apostrophe():
    assert count_words("İstanbul\u2019da yaşıyorum") == 2


def test_turkish_chars():
    assert count_words("Çığ, şöyle güzel İstanbul") == 4


def test_punctuation_and_numbers():
    assert count_words("Fiyat: 1.250 TL, değil mi?") == 5


def test_empty():
    assert count_words("") == 0
