"""Unit tests for the pure CLI-config helpers in ``cli_utils``."""

from cuvis_ai_schemas.training.data import DataSplitConfig, SelectorKind

from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI


def test_parse_data_config_defaults() -> None:
    cli = AdaCLIPCLI()
    cfg = cli.parse_data_config()

    assert cfg["cu3s_file_path"] == "data/Lentils/Lentils_000.cu3s"
    assert cfg["annotation_json_path"] == "data/Lentils/Lentils_000.json"
    assert cfg["batch_size"] == 4
    assert cfg["processing_mode"] == "Reflectance"

    splits = cfg["splits"]
    assert isinstance(splits, DataSplitConfig)
    assert splits.leakage_check == "off"
    # default train/val/test ids parsed from the comma strings
    assert [s.ids for s in splits.train] == [[0, 2]]
    assert [s.ids for s in splits.val] == [[2, 4]]
    assert [s.ids for s in splits.test] == [[1, 3, 5]]
    assert splits.train[0].kind == SelectorKind.FILE_INDICES
    assert splits.train[0].source == "data/Lentils/Lentils_000.cu3s"


def test_parse_data_config_overrides() -> None:
    cli = AdaCLIPCLI()
    cfg = cli.parse_data_config(
        cu3s_file_path="data/custom.cu3s",
        annotation_json_path="data/custom.json",
        batch_size=8,
        processing_mode="Radiance",
        train_ids="0, 1",
        val_ids="2",
        test_ids="3, 4, 5",
    )

    assert cfg["cu3s_file_path"] == "data/custom.cu3s"
    assert cfg["annotation_json_path"] == "data/custom.json"
    assert cfg["batch_size"] == 8
    assert cfg["processing_mode"] == "Radiance"

    splits = cfg["splits"]
    assert [s.ids for s in splits.train] == [[0, 1]]
    assert [s.ids for s in splits.val] == [[2]]
    assert [s.ids for s in splits.test] == [[3, 4, 5]]
    assert splits.test[0].source == "data/custom.cu3s"


def test_parse_normal_class_ids() -> None:
    cli = AdaCLIPCLI()
    assert cli.parse_normal_class_ids("0, 2, 5") == [0, 2, 5]


def test_parse_target_wavelengths() -> None:
    cli = AdaCLIPCLI()
    assert cli.parse_target_wavelengths("660.0, 730, 850.5") == (660.0, 730.0, 850.5)
