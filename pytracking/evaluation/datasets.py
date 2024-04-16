from collections import namedtuple
import importlib
from pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    ###### multi-modality
    gtot=DatasetInfo(module=pt % "gtot", class_name="GTOTDataset", kwargs=dict()),
    gtot_i=DatasetInfo(module=pt % "gtot_i", class_name="GTOT_iDataset", kwargs=dict()),
    rgbt210=DatasetInfo(module=pt % "rgbt210", class_name="RGBT210Dataset", kwargs=dict()),
    rgbt234=DatasetInfo(module=pt % "rgbt234", class_name="RGBT234Dataset", kwargs=dict()),
    lasher=DatasetInfo(module=pt % "lasher", class_name="LasHeRDataset", kwargs=dict()),
    lashertestingset=DatasetInfo(module=pt % "lashertestingset", class_name="LasHeRtestingSetDataset", kwargs=dict()),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset