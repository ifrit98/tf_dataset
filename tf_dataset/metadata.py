import os
import numpy as np
import pandas as pd

from .dataset_sql import signal_dataset
from .dataset_prepare_signal import dataset_prepare_signal_for_ml
from .dataset_slice_windows import dataset_signal_slice_windows
from .dataset_complex import dataset_signal_apply_analytic
from .dataset_normalize import dataset_signal_normalize
from .tf_utils import dataset_batch, dataset_set_shapes, is_empty

from pathlib import Path

def construct_metadata(data_dir, labels=None, label_colname='class', targets=None,
                       metadata=None, ext='.wav', create_targets=True):
    r"""Construct a metadata pandas dataframe for use with `signal_dataset`

    Args:
        data_dir: string filepath to top-level data directory.  
            (Nested dir support coming soon)

        labels: may be either a `np.ndarray` of sequential labels to associate with 
            lexicographically sorted filepaths in `data_dir`, OR a python `dict`, 
            mapping filepaths to labels.
            
            E.g.
                np.ndarray
                labels = np.asarray(['label123', 'label456'])
                
                OR

                dict
                labels =  {
                    'filepath123.wav': 'label123',
                    'filepath456.wav': 'label456',
                    ...,
                }
            
            If none supplied, `construct_metadata()` will use regex to extract
            labels between the final underscore `_` to file extension `.wav'.  
            
            If your file naming scheme does not follow this convention, 
            parsed `labels` will be meaningless.  
            
            E.g. 'data/signal_123_cargo.wav'. -> 'cargo'
            Where
                'cargo' is extracted and added to the labels column of the dataframe 
                for every example in `data_dir`.

        label_colname: string label to use as key for dataframe labels. Default: 'class'

        metadata: (optional) pandas dataframe of relavant information to add to the sql db.

        ext: string representation of file extension starting with '.'.  Default: '.wav'.
            This is how `construct_metadata()` will find data files from the `data_dir`.
            Can be arbitrary, so long as passing `process_db_entry()` along with subsequent
            calls to `training_dataset()` or `signal_dataset()`

        create_targets: boolean. Default: True.

    Returns:
        pd.DataFrame object containing metadata for creating a tf.data.Dataset object
    """
    if targets is not None and create_targets is True:
        import warnings
        warnings.warn(
            "Setting `create_targets` to False since `targets` is supplied.")
        create_targets = False
    data_dir = os.path.abspath(data_dir)
    data_files = list(Path(data_dir).rglob("*" + ext))
    if data_files == []:
        raise ValueError("No files found with ext {} at {}".format(ext, data_dir))
    df = pd.DataFrame.from_dict({'filepath': data_files})

    # Extract from `data_files`
    if isinstance(labels, list) or isinstance(labels, np.ndarray):
        df.loc[:, (label_colname)] = np.asarray(labels).astype(str)
    elif isinstance(labels, dict):
        labels2 = {
            'filepath': list(),
            label_colname: list()
        }
        for k,v in labels.items():
            labels2['filepath'].append(os.path.abspath(k))
            labels2[label_colname].append(v)
        df[label_colname] = pd.DataFrame(labels2)[label_colname]
    elif labels is None:
        import re
        extract = lambda s: re.search("\\D*{}".format(ext), str(s)).group(0)
        extract = np.vectorize(extract)
        classes = extract(df['filepath'])
        extract = lambda s: re.search("(?<=_)\\w*[^{}]".format(ext), s).group(0)
        extract = np.vectorize(extract)
        labels = np.asarray(extract(classes)).astype(str)
        df[label_colname] = labels
    else:
        raise ValueError("`labels` must be one of [`list`, `np.ndarray`, `dict`]")

    if label_colname in df.columns.values:
        df.loc[:, ('num_classes')] = [len(df[label_colname].unique())] * len(df[label_colname])
    if create_targets:
        CLASSES = df[label_colname].unique().astype(str)
        classes_dict = dict(zip(CLASSES, range(len(CLASSES))))
        targets = list(map(lambda x: classes_dict[x], df[label_colname]))
    if targets is not None:
        if len(df['filepath']) != len(targets):
            raise ValueError("Passed `targets` with different length than `df`.")
        df.loc[:, ('target')] = np.asarray(targets)

    return df
