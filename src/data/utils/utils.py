from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, Subset


def group_split(
    dataset: Dataset, test_size: float, groups: pd.Series, random_state: int = None
) -> Tuple[Subset, Subset]:
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(dataset, groups=groups)
    train_idx, test_idx = next(split)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)
