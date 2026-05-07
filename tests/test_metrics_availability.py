import builtins

from eval.metrics import compute_bertscore


def test_bertscore_unavailable_does_not_fake_score(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bert_score":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = compute_bertscore(["예측"], ["정답"])

    assert result["available"] is False
    assert result["average"] is None
    assert result["per_example"] == []
