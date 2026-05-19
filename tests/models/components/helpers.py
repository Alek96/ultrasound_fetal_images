import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_output_shapes(model, img, batch, output_size):
    """Forward pass must return (features, logits) with the right shapes."""
    model.eval()
    with torch.no_grad():
        features, logits = model(img)
    assert logits.shape == (batch, output_size), f"logits shape {logits.shape} != ({batch}, {output_size})"
    assert features.ndim == 2 and features.shape[0] == batch, f"features shape {features.shape} unexpected"
