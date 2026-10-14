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


# Operators the exported SignNet uses, all checked against Unity Sentis 2.6's supported-operator list
# (14 Oct 2026). A new operator means re-checking Sentis before shipping the model.
SENTIS_CHECKED = {
    "Add", "Cast", "Concat", "Constant", "Conv", "Div", "Gather", "Gemm", "Identity", "LayerNormalization",
    "MatMul", "Mod", "Mul", "ReduceMean", "Reshape", "Shape", "Sigmoid", "Slice", "Softmax", "Sqrt",
    "Squeeze", "Sub", "Transpose", "Unsqueeze",
}


def test_onnx_uses_only_sentis_checked_ops(tmp_path):
    onnx = pytest.importorskip("onnx")
    from ashiria_train.export import OPSET, _to_onnx

    model = SignNet(FEATURE_DIM, 72)
    model.set_normalization(np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))
    file = tmp_path / "m.onnx"
    _to_onnx(model.eval(), torch.zeros(1, 32, FEATURE_DIM), file)
    graph = onnx.load(str(file)).graph
    assert {n.domain for n in graph.node} <= {""}, "contrib/custom operator domains don't load in Sentis"
    assert {n.op_type for n in graph.node} <= SENTIS_CHECKED
    assert OPSET <= 25  # Sentis imports opsets 7–25


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
