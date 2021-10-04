import os
import json
import numpy as np
import pandas as pd
import sqlite3 as sq3
import soundfile as sf
import tensorflow as tf
from pathlib import Path
from .utils import read_sql, read_table, write_table, df_to_tftype
from .utils import is_function, timestamp, on_linux
from .tf_utils import as_tensor, as_integer_tensor, compact


def write_signal_dataset_precursor_database(df, 
                                            db_dir=".", 
                                            label_colname='class',
                                            overwrite=True, 
                                            targets=None, 
                                            overwrite_target=False,
                                            verbose=False):
    r"""Precursor Database to [signal_dataset()]

    Creates a new sqlite database from a metadata dataframe that is then used by
        [signal_dataset].

    Args:
        df: Metadata pd.DataFrame of signals produced when the signals database was
            created.

        db_dir: Path to write the new sqlite database to. If not provided will be
            a temp file.

        label_colname: Column string name to choose for labels. Default: 'class'

        targets: (optional) np.ndarray or python list of regressed values

        overwrite: T/F Overwrite db at `db_dir` if exists?

    Returns: 
        Path to new sqlite database to be used by `signal_dataset()`.
    """
    db_dir = os.path.abspath(db_dir)
    path = os.path.join(
        db_dir, 
        "signal_dataset_precursor_{}_{}_.sqlite".format(os.getpid(), timestamp()))
    if (os.path.exists(path)):
      if overwrite:
        os.unlink(path)
      else:
        raise ValueError(
            "db already exists `path`. Set `overwrite = True` or supply a different path")
    if label_colname in df.columns.values:
        classes = df[label_colname].unique()
        classes.sort() #flip [1,0] -> [0,1] !
        classes_dict = dict(zip(classes, range(len(classes))))
        if verbose:
            print("CLASSES_DICT:", classes_dict)
        statistics = pd.DataFrame.from_dict(
            {
                'var_nm': [label_colname] * len(classes_dict.keys()),
                'label': list(classes_dict.keys()),
                'idx': list(classes_dict.values()),
                'total': [len(df[label_colname])] * len(classes_dict.keys()),
                'freq': [len(df[label_colname][df[label_colname] == val]) for val in classes]
            }
        )
        write_sqlite({'statistics': statistics}, path=path)
    else:
        if targets is None:
            raise ValueError(
                "`label_colname` '{}' is not contained in the dataframe and `targets`\
                    ".format(label_colname) + 
                    " is `None`.\nMust supply `targets` if not categorical Ys.")
        classes, classes_dict = None, None

    targets = targets if targets is not None else list(
        map(
            lambda x: classes_dict[x], df[label_colname].values
            )
        )
    if verbose:
        print("targets in write_sqlite...", targets)
    if 'target' not in df.columns.values or overwrite_target:
        df.loc[:, ('target')] = np.asarray(targets)

    if df['filepath'].dtype.name == "object":
        df.loc[:, ('filepath')] = df['filepath'].astype(str)
    
    if df[label_colname].dtype.name == "object":
        df.loc[:, (label_colname)] = df[label_colname].astype(str)

    df.loc[:, ('num_classes')] = len(
        np.unique(
            list(
                map(
                    lambda x: np.argmax(x), df['target'])
                    )
                )
            )
    write_sqlite({'metadata': df}, path=path)
    df = read_table('metadata', path) # Adds index column, simplest workaround
    if verbose:
        print("df (after read) from inside write_sqlite...\n", df)
    write_sqlite({'var_tftypes': df_to_tftype(df)}, path=path)
    return path


def write_sqlite(tables, 
                 path, 
                 table_nm='metadata', 
                 overwrite=False, 
                 busy_timeout=60):
    x = tables
    if isinstance(x, dict):
        for tbl_nm, df in x.items():
            write_sqlite(
                df, path=path, table_nm=tbl_nm, 
                overwrite=overwrite, busy_timeout=busy_timeout)
        return path

    if not (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)): raise TypeError(
        "`x` must be a pandas DataFrame, Series, or python `dict` of (table_nm, df) pairs")
    if not os.path.exists(os.path.abspath(os.path.dirname(path))):
        os.mkdir(path)
        
    is_new_db = not os.path.exists(path)
    con = sq3.connect(path)
    con.execute("PRAGMA busy_timeout = {}".format(busy_timeout * 1000))
    
    if is_new_db:
        con.execute("PRAGMA journal_mode=WAL;")
        con.commit()
        con.close()
        con = sq3.connect(path)

    write_table(con, x.dropna(), table_nm)
    con.close()

    return path

def sqlite_dataset(filename, query, types):
    filename = os.path.abspath(filename)
    dataset = tf.data.experimental.SqlDataset(
        tf.constant("sqlite", dtype=tf.string),
        filename,
        query,
        tuple(types.values.ravel())
    )
    return dataset

def signal_dataset_precursor(path, query):
    if not os.path.exists(path):
        raise ValueError("`path` must contain a valid sqlite database ending in `.db`")
    names_out = read_sql(query, path=path).columns.values
    tf_types  = read_table('var_tftypes', path)
    names_in  = tf_types.columns.values.astype(str)[1:]
    tf_types  = tf_types[names_in[names_out in names_in].ravel()]

    tfdtypes = {
        'string': tf.string,
        'int32': tf.int32,
        'int64': tf.int64,
        'float32': tf.float32,
        'float64': tf.float64
    }
    types = tf_types.applymap(lambda x: tfdtypes[x])
    ds = sqlite_dataset(path, query, types)

    def _apply(*args):
        return dict(zip(types.columns.values, tuple(args)))
    return ds.map(_apply)

def as_signal_dataset_precursor(df, 
                                deterministic=False,
                                overwrite=True,
                                targets=None,
                                verbose=False):
    path = write_signal_dataset_precursor_database(
        df, overwrite=overwrite, targets=targets, verbose=verbose)
    query = "SELECT * FROM `metadata` ORDER BY " +\
        ("`_ROWID_`" if deterministic else "RANDOM()")
    return signal_dataset_precursor(path=path, query=query)

def signal_dataset(df, 
                   onehot_categoricals=False, 
                   infinite=False, 
                   use_soundfile=True,
                   keep_filepath=True, 
                   parallel_files=None, 
                   overwrite_db=True, 
                   targets=None,
                   as_complex=False,
                   signal_dtype='float32',
                   target_dtype='int32',
                   num_parallel_calls=4, 
                   process_db_entry=None, 
                   data_key='signal',
                   load_grams=False,
                   verbose=False):
    r"""Dataset of Signals (or abitrary data formats)
    Generate a tensorflow dataset from `x`.  Requires `x` to be a pandas DataFrame.

    Args:
        x Metadata DataFrame containing signal filepaths and categoricals.

        onehot_categoricals T/F: Onehot categorical feature values?

        infinite T/F: Infinitely shuffle and repeat tensorflow dataset?

        keep_filepath T/F: Keep the signal path as a feature for each
            sample in the dataset?
        
        parallel_files: Number of files to read/write in parallel.  If not None,
            uses `tf.data.Dataset.interleave()` to process entries.  Default
            behavior is `tf.data.Dataset.map()`.

        overwrite_db: bool, passed on to `as_signal_dataset_precursor()`

        process_db_entry: callable to process your data in an abitrary way.
            For example, this repo defaults to 1D signals, expected to be stored
            in int32 format as '.wav' (or similarly encoded lossless) files.

            If your data are 2D images stored as 1D binary strings for example, 
            then you can write something like:

            def my_process_db_img_example(x):
                img = tf.io.parse_tensor(x['filepath'])
                img = tf.io.decode_image(img)
                x['image'] = img
                return x

            See README for more details.

        data_key: string key to store loaded data in. 

    Returns:
        A tensorflow dataset with each sample containing up to `n` (usually 3) python
        dictionary entries the signal, feature values and associated metadata values.
    """
    if on_linux():
        target_dtype = 'float64'
    else:
        target_dtype = 'float32'

    def process_db_entry_soundfile(x):
        filename = x.get('filepath', None)
        if filename is None:
            raise ValueError('Signal filepath not provided')
        data, fs = tf.py_function(
            lambda fp: sf.read(fp.numpy()), inp=[filename], Tout=[signal_dtype, 'int32']
        )
        x['fs'] = as_integer_tensor(fs)
        x['fs'].set_shape(())
        if as_complex:
            x[data_key] = tf.complex(real=data[:, 0], imag=data[:, 1])
        else:
            x[data_key] = as_tensor(data, data.dtype.name)

        if not keep_filepath:
            x.pop('filepath')

        if onehot_categoricals:
            raise NotImplementedError(
                "Please one-hot your labels just before entering models")
        else:
            if x['target'].dtype == tf.string:
                x['target'] = tf.io.decode_raw(x['target'], target_dtype)
            x['target'] = as_integer_tensor(x['target'])
        return x

    def process_db_entry_npz_gram(x, key='arr_0'):
        filename = x.get('filepath', None)
        if filename is None:
            raise ValueError('Signal filepath not provided')
        data, fs = tf.py_function(
            lambda fp: np.load(
                fp.numpy())[key], inp=[filename], Tout=[signal_dtype, 'float64']
        )
        x['fs'] = as_integer_tensor(fs)
        x['fs'].set_shape(())
        x[data_key] = as_tensor(data, data.dtype.name)

        if not keep_filepath:
            x.pop('filepath')

        if onehot_categoricals:
            raise NotImplementedError(
                "Please one-hot your labels just before entering models")
        else:
            if x['target'].dtype == tf.string:
                x['target'] = tf.io.decode_raw(x['target'], target_dtype)
            x['target'] = as_integer_tensor(x['target'])
        return x

    def process_db_entry_base(x):
        base = tf.io.read_file(x['filepath'])
        base = tf.io.decode_raw(base, tf.int32)
        fs = base[6] #Byte indices 20-23 represent the sampling rate in .wav format
        signal = base[11:] #First 44 bytes are header information, signal begins at index 44
        x[data_key] = signal
        x['fs'] = fs
        if not keep_filepath:
            x.pop('filepath')
        if onehot_categoricals:
            x['target'] = tf.one_hot(
                as_integer_tensor(x['target']), 
                as_integer_tensor(x['num_classes']),
                dtype=tf.int64
            )
            x['target'].set_shape(x['num_classes'])
        return x

    ds = as_signal_dataset_precursor(
        df, overwrite=overwrite_db, targets=targets, verbose=verbose
    )

    if infinite:
        ds = ds.repeat()
    if process_db_entry is None:
        if use_soundfile:
            process_db_entry = process_db_entry_soundfile
        elif load_grams:
            process_db_entry = process_db_entry_npz_gram
        else:
            process_db_entry = process_db_entry_base
    elif not is_function(process_db_entry):
        raise ValueError("`process_db_entry` must be callable, and return a `dict`")

    if parallel_files is None:
        return ds.map(process_db_entry,
            num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int64))
    else:
        return ds.interleave(
            lambda x: (process_db_entry(x)),
            cycle_length = tf.convert_to_tensor(parallel_files, tf.int64),
            num_parallel_calls = tf.convert_to_tensor(num_parallel_calls, tf.int64)
        )
