import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ashiria_train.features import FEATURE_DIM  # noqa: E402
from ashiria_train.model import SignNet  # noqa: E402


def small_model(classes: int = 7) -> SignNet:
    model = SignNet(FEATURE_DIM, classes, dim=64, stages=1, conv_per_stage=1, heads=2)
    model.set_normalization(np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))
    return model.eval()


def test_forward_shape():
    out = small_model()(torch.zeros(3, 32, FEATURE_DIM))
    assert out.shape == (3, 7)


def test_onnx_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from ashiria_train.export import _to_onnx

    model = small_model()
    x = torch.randn(1, 32, FEATURE_DIM)
    file = tmp_path / "m.onnx"
    _to_onnx(model, torch.zeros(1, 32, FEATURE_DIM), file)
    got = ort.InferenceSession(str(file), providers=["CPUExecutionProvider"]).run(["logits"], {"features": x.numpy()})[0]
    np.testing.assert_allclose(got, model(x).detach().numpy(), atol=1e-4)
