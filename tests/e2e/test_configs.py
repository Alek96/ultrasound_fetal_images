import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_brain_planes_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_brain_planes_train
    assert cfg_brain_planes_train.data
    assert cfg_brain_planes_train.model
    assert cfg_brain_planes_train.trainer

    HydraConfig().set_config(cfg_brain_planes_train)

    hydra.utils.instantiate(cfg_brain_planes_train.data)
    hydra.utils.instantiate(cfg_brain_planes_train.model)
    hydra.utils.instantiate(cfg_brain_planes_train.trainer)


def test_eval_config(cfg_brain_planes_eval: DictConfig) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    assert cfg_brain_planes_eval
    assert cfg_brain_planes_eval.data
    assert cfg_brain_planes_eval.model
    assert cfg_brain_planes_eval.trainer

    HydraConfig().set_config(cfg_brain_planes_eval)

    hydra.utils.instantiate(cfg_brain_planes_eval.data)
    hydra.utils.instantiate(cfg_brain_planes_eval.model)
    hydra.utils.instantiate(cfg_brain_planes_eval.trainer)
