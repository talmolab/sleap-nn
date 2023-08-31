import torch

from sleap_nn.config.data import base_topdown_data_config
from sleap_nn.data.confidence_maps import ConfidenceMapGenerator
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.pipelines import SleapDataset, TopdownConfmapsPipeline
from sleap_nn.data.providers import LabelsReader


def test_sleap_dataset(minimal_instance):
    datapipe = LabelsReader.from_filename(filename=minimal_instance)
    datapipe = Normalizer(datapipe)
    datapipe = InstanceCentroidFinder(datapipe)
    datapipe = InstanceCropper(datapipe, (160, 160))
    datapipe = ConfidenceMapGenerator(datapipe, sigma=1.5, output_stride=2)
    datapipe = SleapDataset(datapipe)

    sample = next(iter(datapipe))
    assert len(sample) == 2
    assert sample[0].shape == (1, 160, 160)
    assert sample[1].shape == (2, 80, 80)


def test_topdownconfmapspipeline(minimal_instance):
    pipeline = TopdownConfmapsPipeline(data_config=base_topdown_data_config)
    datapipe = pipeline.make_base_pipeline(
        data_provider=LabelsReader, filename=minimal_instance
    )

    sample = next(iter(datapipe))
    assert len(sample) == 2
    assert sample[0].shape == (1, 160, 160)
    assert sample[1].shape == (2, 80, 80)