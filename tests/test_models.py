import torch
from echoprime_transfer.models import EFRegressionHead, trainable_parameter_count

def test_regression_head_shape():
    model = EFRegressionHead(embedding_dim=512, hidden_dim=32)
    y = model(torch.randn(4,512))
    assert y.shape == (4,)
    assert trainable_parameter_count(model) > 0
