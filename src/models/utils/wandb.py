from typing import Optional, Sequence

import torch
import wandb


def wandb_confusion_matrix(
    cm: torch.Tensor,
    class_names: Sequence[str],
    title: Optional[str] = None,
):
    cm = cm.cpu().numpy()
    data = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            data.append([class_names[i], class_names[j], cm[i, j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    title = title or ""
    return wandb.plot_table(
        "fetal-brain/multi-run_confusion_matrix",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": title},
    )
