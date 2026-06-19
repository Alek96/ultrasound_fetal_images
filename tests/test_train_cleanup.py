from pathlib import Path

from omegaconf import OmegaConf

from src.train import cleanup_log_directory


def _make_cfg(output_dir: Path, cleanup_mode=None):
    cfg = OmegaConf.create({"paths": {"output_dir": str(output_dir)}})
    if cleanup_mode is not None:
        cfg.cleanup_mode = cleanup_mode
    return cfg


def test_cleanup_log_directory_none_is_noop(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.ckpt").write_text("weights")
    (tmp_path / ".hydra").mkdir()

    cleanup_log_directory(_make_cfg(tmp_path, cleanup_mode="none"))

    assert (tmp_path / "checkpoint.ckpt").exists()
    assert (tmp_path / ".hydra").exists()


def test_cleanup_log_directory_model_weights_keeps_metadata(tmp_path: Path) -> None:
    (tmp_path / ".hydra").mkdir()
    (tmp_path / "tags.log").write_text("tags")
    (tmp_path / "config_tree.log").write_text("tree")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "last.ckpt").write_text("weights")
    (tmp_path / "metrics.json").write_text("metrics")

    cleanup_log_directory(_make_cfg(tmp_path, cleanup_mode="model_weights"))

    assert (tmp_path / ".hydra").exists()
    assert (tmp_path / "tags.log").exists()
    assert (tmp_path / "config_tree.log").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_cleanup_log_directory_all_removes_output_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.ckpt").write_text("weights")

    cleanup_log_directory(_make_cfg(tmp_path, cleanup_mode="all"))

    assert not tmp_path.exists()
