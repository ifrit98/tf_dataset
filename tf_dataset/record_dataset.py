import os
import json
import itertools
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile
from .tf_utils import grab, is_scalar, first, is_dataset_batched
from .tf_utils import dataset_set_shapes, dataset_batch, dataset_unbatch

from warnings import warn

JSON_FNAME = "TF_RECORD_FEATURES_SPEC.json"

TF_D_SWITCH = {
    "tf.string": tf.string,
    "tf.int32": tf.int32,
    "tf.int64": tf.int64, 
    "tf.float32": tf.float32,
    "tf.complex64": tf.complex64
}

NP_D_SWITCH = {
    'string': 'tf.string',
    'bytes': 'tf.string',
    'int32': 'tf.int32',
    'int64': 'tf.int64',
    'float32': 'tf.float32',
    'float64': 'tf.float64',
    'complex64': 'tf.complex64',
    'complex128': 'tf.complex128'
}

def np_to_tf_dtype(typename):
    if 'bytes' in typename:
        typename = 'bytes'
    return NP_D_SWITCH[typename]

def dtype_switch(typename, return_str=True):
    if 'bytes' in typename:
        return 'tf.string' if return_str else tf.string
    if 'string' in typename:
        return 'tf.string' if return_str else tf.string
    if 'int32' in typename:
        return 'tf.int64' if return_str else tf.int64
    if 'int64' in typename:
        return 'tf.int64' if return_str else tf.int64
    if 'float32' in typename:
        return 'tf.float32' if return_str else tf.float32
    if 'float64' in typename:
        return 'tf.float64' if return_str else tf.float64

def decode_dtype_name(v): # numpy b'string' (byte string) workaround
    if not is_scalar(v):
        v = first(v)
    if type(v) == bytes or type(v) == str:
        return 'string'
    v = np.asarray(v)
    return v.dtype.name

def as_example(record):
    _x = {}
    update_keys = set()
    for k,v in record.items():
        if type(v) == np.ndarray and v.dtype == np.complex64:
                update_keys.add(k) # no duplicates with sets
                bits = tf.bitcast(v, 'float32').numpy()
                _x[k+'_real'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(bits[:, 0]).numpy()]))
                _x[k+'_imag'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(bits[:, 1]).numpy()]))
                continue
        t = tf.convert_to_tensor(v, name=k)
        s = tf.io.serialize_tensor(t)
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.numpy()]))
        _x[k] = f

    x = tf.train.Features(feature=_x)
    return tf.train.Example(features=x).SerializeToString()

def _write_to_tfrecord(record, writer):
    writer.write(as_example(record))

def _build_json_dict(record):
    assert isinstance(record, dict)

    def _assert_is_flat_dict(_dict):
        for v in _dict.values():
            if type(v) == dict:
             raise TypeError("record is a nested dictionary")
        return True

    _assert_is_flat_dict(record)
    record_keys = record.keys()
    tensor_metadata = ['fnc', 'dtype', 'shape', 'previous_dtype']
    features = dict.fromkeys(record_keys)
  
    from copy import deepcopy
    for field in record_keys:
        item = deepcopy(record[field]) # to avoid pass by reference issue
        dtype_name = decode_dtype_name(item)

        features[field] = dict.fromkeys(tensor_metadata)
        if type(item) == bytes:
            item = np.asarray(item)
        if item.shape == () or item.shape == (1,) or item.ndim == 1:
            # TODO: update chained indexing df['a']['b'] ->> df.loc[:, ('a', 'b')]
            features[field]['fnc'] = 'FixedLen'
            # features.loc[:, (field, 'fnc')] = 'FixedLen'
        else:
            features[field]['fnc'] = 'FixedLenSequence'
            # features.loc[:, (field, 'fnc')] = 'FixedLenSequence'
        # get shape
        features[field]['shape'] = item.shape
        # features.loc[:, (field, 'shape')] = item.shape
        # get previous dtype (e.g. numpy bytes40 or something odd)
        features[field]['previous_dtype'] = np_to_tf_dtype(dtype_name)
        # features.loc[:, (field, 'previous_dtype')] = np_to_tf_dtype(dtype_name)

        if 'complex' in features[field]['previous_dtype']:
            dtype = 'tf.float32' # will be saved out as 2 bitcased arrays
            features[field + '_real'] = {
                'dtype': dtype, 'fnc': 'FixedLenSequence', 
                'shape': item.shape, 'previous_dtype': 'tf.complex64'}
            features[field + '_imag'] = {
                'dtype': dtype, 'fnc': 'FixedLenSequence',
                'shape': item.shape, 'previous_dtype': 'tf.complex64'}
        
        features[field]['dtype'] = dtype_switch(dtype_name)
        # features.loc[:, (field, 'dtype')] = dtype_switch(dtype_name)
    
    # remove original complex signal.  It can't be saved.
    for field in list(features.keys()):
        if '_imag' in field:
            features.pop(field[:-5])

    return features

def replay_dataset(record_dir, shuffle=False):
    path = os.path.abspath(record_dir)
    records = [os.path.join(
        path, f) for f in os.listdir(path) if f.endswith('.tfrecord')]
    record_ds = tf.data.TFRecordDataset(records)
    json_path = os.path.join(path, JSON_FNAME)

    with open(json_path) as json_file:
        config = json.load(json_file)

    path = os.path.dirname(os.path.abspath(json_path))
    path = os.path.join(path, 'batched')
    if os.path.exists(path):
        with open(path, 'r') as f:
            batch_size = int(f.read(1))
    else:
        batch_size = None

    features = {
        k: tf.io.FixedLenFeature([], 
                                 tf.string,
                                 default_value="") for k,v in config.items()}

    def parse_tensors(x):
        x = tf.io.parse_single_example(x, features=features)
        _x = dict.fromkeys(x)
        cast_to_cplx = []
        for k,v in x.items():
            mold = config[k]

            if mold['previous_dtype'] in ['tf.complex64', 'tf.complex128']:
                cast_to_cplx.append({k: config[k]})        

            dtype_sub = mold['dtype'][-2:]
            pdtype_sub = mold['previous_dtype'][-2:]

            # I felt rushed when originally doing this. 
            # (@bares) Can we do these string comparisons more elegantly?
            if dtype_sub in ['32', '64'] and pdtype_sub in ['32', '64']:
                if dtype_sub != pdtype_sub:
                    if mold['dtype'][:-2] == mold['previous_dtype'][:-2]:
                        mold['dtype'] = mold['dtype'][:-2] + pdtype_sub
            tensor = tf.io.parse_tensor(
                v,
                out_type=TF_D_SWITCH[mold['dtype']],
                name=k)
            tensor.set_shape(mold['shape'])
            _x[k] = tensor

        # Restore complex arrays and correct original key values
        if len(cast_to_cplx):
            k0 = list(cast_to_cplx[0].keys())[0]
            k1 = list(cast_to_cplx[1].keys())[0]
            og_key = k0[:-5]
            _x[og_key] = tf.complex(real=_x[k0], imag=_x[k1])
            [_x.pop(k) for k in [k0, k1]]

        return _x

    ds = record_ds.map(parse_tensors)
    return dataset_batch(ds, batch_size) if batch_size is not None else ds

def record_dataset(dataset, output_dir=".", elements_per_file=2, num_elems=None):
    r"""Record an iterator (tensorflow dataset) to disk by serializing to
        `tfrecord` format.

        Creates a json file "TF_RECORD_FEATURES_SPEC.json" that contains metadata
        about the dataset so that we can tell tensorflow how to reconstruct the 
        data when reading.  Stored information is: 
            [element name, element shape, element dtype]

        Grabs `elements_per_file` elements from the dataset and writes 
        num_elems//elements_per_file number of files out.

        E.g.: (If divsible)
            elements_per_file = 10
            num_elements = 50
            batches = [10, 10, 10, 10, 10]
            num_files = len(batches) # 5

            If not divisible:
            elements_per_file = 10
            num_elements = 33
            batches = [10, 10, 10, 3]
            num_files = len(batches) # 4

            if num_elems = None (default):
            elements_per_file = 10
            batches = [10,10,10,10,...] # lazy infinte list

        Args:
            dataset: tensorflow dataset object

            output_dir: string path to where binary tfrecords will be written.

            elements_per_file: how many elems should there be in a single tfrecord.

            num_elems: int number of elems to write from the dataset.

        Returns:
            output_dir
    """
    
    # check if json feature dict already exists in dir
    output_dir = os.path.abspath(output_dir)
    json_path  = os.path.join(output_dir, JSON_FNAME)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if is_dataset_batched(dataset):
        nb = grab(dataset)
        batch_size = nb[list(nb.keys())[0]].shape[0]
        del nb
        path = os.path.dirname(os.path.abspath(json_path))
        path = os.path.join(path, 'batched')
        with open(path, 'w') as f:
            f.write(str(batch_size))
        dataset = dataset_unbatch(dataset)
        # Dataset will be rebatched with correct batch size upon reloading...
        # warn('Dataset is being unbatched for use with `record_dataset().')
        # warn('It will be returned to original batch size upon reloading...')

    # Ensure shapes are known at runtime
    dataset = dataset_set_shapes(dataset)

    # Convert to a numpy iterator for saving
    it = dataset.as_numpy_iterator()

    # load existing json if exists
    if os.path.isfile(json_path):
        with open(json_path) as f:
            json_dict = json.load(f)
    else:
        batch = next(it)
        it = itertools.chain([batch], it)
        json_dict = _build_json_dict(batch)
        with open(os.path.join(output_dir, JSON_FNAME), "w") as json_file:
            json.dump(json_dict, json_file)

    if num_elems is not None:
        elements_per_file = itertools.chain(
            itertools.repeat(elements_per_file, num_elems//elements_per_file),
            [num_elems % elements_per_file])
    else: # Write out complete dataset by creating an infinite generator
        epf = int(elements_per_file)
        def inf_gen():
            while True:
                yield epf
        elements_per_file = inf_gen()

    file_no = 0
    for n in elements_per_file:
        file_no += 1
        output_record = NamedTemporaryFile(dir=output_dir,
                                       suffix='.tfrecord',
                                       delete=False)
        with tf.io.TFRecordWriter(output_record.name) as writer:
            # wrap this in a try block to make sure we can get all the way to the end
            for i, record in enumerate(it):
                if i == n: break # Reached maximum elements_per_file
                print('Processing batch: {} of {} in file {}'.format(i, n, file_no))
                _write_to_tfrecord(record, writer)
        if i != n: break # get us out of infinite loop

    print("Completed writing tfrecords to {}.".format(output_dir))
    return output_dir


if False:
    import os
    import json
    import itertools
    import tensorflow as tf
    from tempfile import NamedTemporaryFile
    import numpy as np
    from tf_dataset import *
    from tf_dataset.record_dataset import *
    from tf_dataset.record_dataset import _build_json_dict, _write_to_tfrecord
    from tf_preprocess import dataset_compute_jspectrogram2
    df = construct_metadata("./tf_dataset/data") 
    ds = signal_dataset(df)
    ds = dataset_signal_slice_windows(ds, 320000)
    ds = dataset_compute_jspectrogram2(ds)
    for x in ds: info(x); break
    record_dir = output_dir="./records"
    num_elems = 10
    elements_per_file = 2

    # TODO: handle correct writing for grams and other new features!
    record_dataset(
        ds,
        output_dir=output_dir, 
        num_elems=num_elems, 
        elements_per_file=elements_per_file)

    dsr = replay_dataset(output_dir)
    for x in dsr: break
    info(x)
    

    json = _build_json_dict(grab(ds))

    # if is_dataset_batched(dataset):
    #     warn("Dataset will be unbatched first and then rebatched.")
    #     nb = grab(dataset)
    #     batch_size = nb[list(nb.keys())[0]].shape[0]
    #     del nb
    #     dataset = dataset_unbatch(dataset)
    # else:
    #     batch_size = None

    # dataset_batch(dataset, batch_size) if batch_size is not None else dataset