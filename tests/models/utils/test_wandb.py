"""Unit tests for src/models/utils/wandb."""

from unittest.mock import patch

import pytest
import torch

from src.models.utils.wandb import wandb_confusion_matrix


def test_calls_plot_table_with_correct_fields():
    cm = torch.tensor([[5, 1], [2, 8]])
    with patch("wandb.plot_table") as mock_plot, patch("wandb.Table"):
        wandb_confusion_matrix(cm, ["a", "b"], title="my_title")
        mock_plot.assert_called_once()
        fields = mock_plot.call_args[1]["fields"]
        assert "Actual" in fields
        assert "Predicted" in fields
        assert "nPredictions" in fields


def test_title_forwarded_correctly():
    cm = torch.eye(2, dtype=torch.int)
    with patch("wandb.plot_table") as mock_plot, patch("wandb.Table"):
        wandb_confusion_matrix(cm, ["a", "b"], title="test_title")
        string_fields = mock_plot.call_args[1]["string_fields"]
        assert string_fields["title"] == "test_title"


def test_default_title_is_empty_string():
    cm = torch.eye(3, dtype=torch.int)
    with patch("wandb.plot_table") as mock_plot, patch("wandb.Table"):
        wandb_confusion_matrix(cm, ["a", "b", "c"])
        string_fields = mock_plot.call_args[1]["string_fields"]
        assert string_fields["title"] == ""


@pytest.mark.parametrize("n", [2, 3, 5])
def test_table_data_has_n_squared_rows(n: int):
    """wandb.Table must receive exactly n² rows for an n×n confusion matrix."""
    cm = torch.randint(0, 10, (n, n))
    class_names = [f"c{i}" for i in range(n)]
    captured = []
    with patch("wandb.plot_table"), patch("wandb.Table", side_effect=lambda columns, data: captured.append(data)):
        wandb_confusion_matrix(cm, class_names)
    assert len(captured[0]) == n * n


@pytest.mark.parametrize("n", [2, 3])
def test_table_values_match_confusion_matrix(n: int):
    """Each cell value in the table must equal the corresponding cm entry."""
    cm = torch.arange(n * n, dtype=torch.int).reshape(n, n)
    class_names = [f"c{i}" for i in range(n)]
    captured = []
    with patch("wandb.plot_table"), patch("wandb.Table", side_effect=lambda columns, data: captured.append(data)):
        wandb_confusion_matrix(cm, class_names)
    data = captured[0]
    for row in data:
        actual_idx = class_names.index(row[0])
        pred_idx = class_names.index(row[1])
        assert row[2] == cm[actual_idx, pred_idx].item()
