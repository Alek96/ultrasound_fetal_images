from collections.abc import Sequence

import torch
import wandb


def wandb_confusion_matrix(
    cm: torch.Tensor,
    class_names: Sequence[str],
    title: str | None = None,
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
        vega_spec_name="fetal-brain/multi-run_confusion_matrix",
        data_table=wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields=fields,
        string_fields={"title": title},
    )
