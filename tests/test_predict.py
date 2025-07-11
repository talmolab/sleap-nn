from pathlib import Path
import sleap_io as sio
from sleap_nn.predict import main


def test_predict_main(
    minimal_instance,
    minimal_instance_ckpt,
    minimal_instance_centroid_ckpt,
    tmp_path,
):
    # test for centered instance + saving slp file
    main(
        [
            "--data_path",
            f"{minimal_instance}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_preds.slp",
            "--peak_threshold",
            f"{0.1}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_preds.slp").exists()

    gt = sio.load_slp(minimal_instance)
    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_preds.slp").as_posix())

    assert len(gt) == len(pred)

    # centorid+centered
    main(
        [
            "--data_path",
            f"{minimal_instance}",
            "--model_paths",
            f"{minimal_instance_centroid_ckpt}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_topdown_preds.slp",
            "--peak_threshold",
            f"{0.0}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_topdown_preds.slp").exists()

    gt = sio.load_slp(minimal_instance)
    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_topdown_preds.slp").as_posix())

    assert len(gt) == len(pred)

    # test with video
    main(
        [
            "--data_path",
            "./tests/assets/centered_pair_small.mp4",
            "--model_paths",
            f"{minimal_instance_centroid_ckpt}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_preds.slp",
            "--frames",
            "0-5",
            "--peak_threshold",
            f"{0.0}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_preds.slp").exists()

    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_preds.slp").as_posix())

    assert len(pred) == 6

    # test with video
    main(
        [
            "--data_path",
            "./tests/assets/centered_pair_small.mp4",
            "--model_paths",
            f"{minimal_instance_centroid_ckpt}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_preds.slp",
            "--frames",
            "0,1,2",
            "--peak_threshold",
            f"{0.0}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_preds.slp").exists()

    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_preds.slp").as_posix())

    assert len(pred) == 3

    # test with video
    main(
        [
            "--data_path",
            "./tests/assets/centered_pair_small.mp4",
            "--model_paths",
            f"{minimal_instance_centroid_ckpt}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_preds.slp",
            "--frames",
            "0-3",
            "--peak_threshold",
            f"{0.0}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_preds.slp").exists()

    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_preds.slp").as_posix())

    assert len(pred) == 4

    # test with video index
    main(
        [
            "--data_path",
            f"{minimal_instance}",
            "--model_paths",
            f"{minimal_instance_centroid_ckpt}",
            "--model_paths",
            f"{minimal_instance_ckpt}",
            "-o",
            f"{tmp_path}/minimal_inst_preds.slp",
            "--video_index",
            "0",
            "--frames",
            "0",
            "--peak_threshold",
            f"{0.0}",
            "--device",
            "cpu",
        ]
    )
    assert (Path(tmp_path) / "minimal_inst_preds.slp").exists()

    pred = sio.load_slp((Path(tmp_path) / "minimal_inst_preds.slp").as_posix())

    assert len(pred) == 1
