import os
import pytest
import numpy as np
import pandas as pd
import sqlite3 as sq3
import tensorflow as tf
from numpy import all as np_all
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal, assert_equal
import json
import itertools
from tempfile import NamedTemporaryFile

from .dataset_sql import write_signal_dataset_precursor_database
from .dataset_sql import signal_dataset, signal_dataset_precursor
from .record_iterator import _build_json_dict, _write_to_tfrecord
from .slice_windows import dataset_signal_slice_windows
from .record_iterator import record_dataset, replay_dataset
from .tensorflow_dataset import training_dataset, dataset_from_dir, construct_metadata
from .normalize_gain import dataset_signal_normalize, dataset_signal_safe_rescale, tf_normalize_signal_gain, safe_rescale
from .to_complex import dataset_signal_apply_analytic, dataset_signal_apply_hilbert, tf_analytic, tf_hilbert
from .compact import dataset_compact
from .tf_utils import find_bit_depth, bit_normalize, shapes, within1, as_integer_tensor
from .record_iterator import JSON_FNAME

data_key = 'signal'

def minmax(x):
    mi = tf.reduce_min(x)
    ma = tf.reduce_max(x)
    return (mi.numpy(), ma.numpy())

def assert_in(x, y):
    assert x in y

TOL = 1e-4
DATADIR = "./data"
OUTPUT_DIR = "./records"
JSON_FNAME = "TF_RECORD_FEATURES_SPEC.json"
BATCH_SIZES = [1, 2, 4, 7, 15, 32, 99, 128]
WINDOW_SIZES = [256, 512, 1024, 4096, 8192, 32768, 100000]
TOTAL_SAMPLES = 32074556
SAMPLES = tf.constant([2613707, 14681003, 13793339, 986507])
DS_SIZES = {
    win_len: tf.reduce_sum(
        SAMPLES // win_len).numpy() for win_len in WINDOW_SIZES}

d = {
'filepath': [
    'data/1561965276.252_1561965582.106_24346_passenger.wav',
    'data/1561970414.69_1561970702.051_24346_passenger.wav',
    'data/1561971147.625_1561971202.077_24346_cargo.wav',
    'data/1561971697.731_1561971718.283_24346_tug.wav'],
'class': ['passenger', 'passenger', 'cargo', 'tug']
}
DF = pd.DataFrame.from_dict(d)
DF['filepath'] = DF['filepath'].apply(lambda x: os.path.abspath(x))
DF['target'] = [0, 0, 1, 2]
DF['num_classes'] = [4, 4, 4, 4]

@pytest.mark.filterwarnings("ignore:.*usage will be deprecated.*:DeprecationWarning")
def test_write_signal_dataset_precursor_database():
    path = write_signal_dataset_precursor_database(DF, overwrite=True)
    assert os.path.exists(path)
    conn = sq3.connect(path)
    df = pd.read_sql(
        "SELECT * FROM `{}` ORDER BY `_ROWID_`".format('metadata'), conn)
    list(map(lambda c: assert_array_equal(df[c], DF[c]), DF.columns))
    return True

def test_signal_dataset_precursor():
    path = write_signal_dataset_precursor_database(DF, overwrite=True)
    deterministic = True
    query = "SELECT * FROM `metadata` ORDER BY " +\
        ("`_ROWID_`" if deterministic else "RANDOM()")

    conn = sq3.connect(path)
    df = pd.read_sql(
        "SELECT * FROM `{}` ORDER BY `_ROWID_`".format('metadata'), conn)

    ds = signal_dataset_precursor(path, query)
    for x in ds:
        {
            assert_in(v.numpy().decode(), df[k].values) 
                if v.dtype == tf.string else assert_in(v.numpy(), df[k].values)
                    for k,v in x.items()
        }
    return True

def test_signal_dataset():
    ds = signal_dataset(DF, overwrite_db=True)
    def tf_load_signal(filepath):
        return tf.io.decode_raw(tf.io.read_file(filepath), tf.int32)        

    for x in ds:
        fp = x['filepath']
        signal = tf_load_signal(fp)
        assert_allclose(signal, x[data_key].numpy(), rtol=TOL)
    return True

def test_slice_windows():
    ds = signal_dataset(DF)
    for win_size in WINDOW_SIZES:
        ds1 = dataset_signal_slice_windows(ds, win_size)
        for i, x in ds1.enumerate():
            assert_equal((win_size,), x[data_key].shape)
        assert within1(i.numpy(), DS_SIZES[win_size])
    return True
    
def test_to_complex():
    ones = tf.ones([1000])
    ones_h = tf_hilbert(ones)
    assert_allclose(ones, abs(ones_h), rtol=TOL)
    return True
    # ds = signal_dataset(DF)
    # dsl = list(ds)
    # dsh = dataset_signal_apply_hilbert(ds)

    # for x in dsh:
    #     xs = x['signal'] # complex valued signal
    #     for y in dsl:
    #         if y['index'] == x['index']:
    #             ys = y['signal']
    #             break
    #     xs = as_integer_tensor(tf.math.real(xs))
    #     assert_allclose(xs.numpy(), ys.numpy(), rtol=1e1)

def test_normalize():
    ds = signal_dataset(DF)

    for nb in ds.take(1):
        x = nb[data_key]

    xs = safe_rescale(x)
    assert_allclose(minmax(xs), (-1.0, 1.0), rtol=TOL)

    xg = tf_normalize_signal_gain(x)
    assert_allclose(minmax(xg), (-1.0, 1.0), rtol=TOL)

    rds = dataset_signal_safe_rescale(ds)
    gds = dataset_signal_normalize(ds)

    for r in rds:
        assert_allclose(minmax(r[data_key]), (-1.0, 1.0), rtol=TOL)
    for g in gds:
        assert_allclose(minmax(g[data_key]), (-1.0, 1.0), rtol=TOL)
    
    return True

def test_batch_sizes():
    ds = signal_dataset(DF)
    for bs in BATCH_SIZES:
        dss = dataset_signal_slice_windows(ds, 32768)
        dsb = dss.batch(bs, drop_remainder=True)
        for x in dsb:
            assert x[data_key].shape[0] == bs

def test_tensorflow_dataset():
    ds = dataset_from_dir(DATADIR)
    assert ds.__class__ == tf.python.data.ops.dataset_ops.PrefetchDataset
    for x in ds:
        assert x[data_key].ndim == 3
        assert x[data_key].shape == (16, 1024, 12) # subject to change
        assert x[data_key].dtype == tf.float32 

def test_record_generator_reader():
    num_elems = 40
    elements_per_file = 4
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    ds = signal_dataset(DF)
    ds = dataset_signal_slice_windows(ds, 8192).repeat()

    record_dataset(ds, 
        output_dir=OUTPUT_DIR, 
        elements_per_file=elements_per_file, 
        num_elems=num_elems)

    #Reader code
    ds_write = ds
    ds_read = replay_dataset(OUTPUT_DIR)
    for x in ds_read.take(1):
        sx = shapes(x)
    for y in ds_write.take(1):
        sy = shapes(y)
    assert sx == sy
    return True

def test_construct_metadata():
    df = construct_metadata(DATADIR)
    for col in df.columns:
        assert np_all(df[col] == DF[col])

def test_dataset_compact():
    ds = signal_dataset(DF)
    x_key = 'signal'
    y_key = 'target'
    dsc = dataset_compact(ds, x_key, y_key)
    for x in dsc:
        signal = x[0]
        target = x[1]
        assert signal.shape[0] in SAMPLES
        assert target in range(len(DF['target']))

def test_cleanup():
    from shutil import rmtree
    rmtree(OUTPUT_DIR)

    ls = np.asarray(os.listdir())
    splits = list(map(os.path.splitext, ls))
    idx = list(map(lambda x: x[1] == '.sqlite', splits))
    list(map(os.unlink, ls[idx]))
