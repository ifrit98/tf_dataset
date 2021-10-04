import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor, EagerTensor
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.framework.config import list_physical_devices

def alleq(x):
    try:
        iter(x)
    except:
        x = [x]
    current = x[0]
    for v in x:
        if v != current:
            return False
    return True     

def first(x):
    if is_scalar(x):
        return x
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[0] * len(x.shape)]

def last(x):
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[-1] * len(x.shape)]

def compact(x, x_keys, y_keys):
    """Compacts the `x` dictionary into (x, y), potentially nested
       tuple pairs for use with `tf.keras.Model.fit()`
    """
    x_scalar = is_scalar(x_keys)
    y_scalar = is_scalar(y_keys)
    if x_scalar:
        if y_scalar:
            return (x[x_keys], x[y_keys])
        else:
            return (x[x_keys], tuple(x[k] for k in y_keys))
    else:
        if y_scalar:
            return (tuple(x[k] for k in x_keys), x[y_keys])
    return (tuple(x[k] for k in x_keys), tuple(x[k] for k in y_keys))

# TODO: Make sure whis works both nested and not
# TODO: fix load_AIS data to allow batching?? (`data`: tf.string)?
def dataset_set_shapes(ds):
    for nb in ds: break
    shapes = {
        k: v.shape.as_list() if type(v) is not dict \
                             else {
                                 _k: _v.shape.as_list() for _k,_v in v.items()
                            } for k,v in nb.items()}
    del nb
    def set_shapes(x):
        [v.set_shape(shapes[k]) if type(shapes[k]) is not dict \
                                else {ki: vi.set_shape(shapes[k][ki]) for ki,vi in v.items()}\
                                for k,v in x.items()]
        return x
    return ds.map(set_shapes)

def dataset_onehot(ds, target_key='target'):
    def onehot(x):
        x[target_key] = tf.one_hot(
            as_integer_tensor(x[target_key]), 
            as_integer_tensor(x['num_classes']),
            dtype=tf.int64
        )
        return x
    return ds.map(onehot)

def dataset_flatten(ds, key):
    def flatten(x):
        x[key] = tf.keras.layers.Flatten()(x[key])
        return x
    return ds.map(flatten)

def dataset_batch(ds, batch_size):
    batch_size = as_tensor(batch_size, 'int64')
    if not is_scalar_tensor(batch_size):
        raise ValueError("`batch_size` must be a scalar.")
    return ds.batch(batch_size, drop_remainder=True)

def dataset_shuffle(ds, shuffle_buffer_size, reshuffle_each_iteration=True):
    shuffle_buffer_size = as_tensor(shuffle_buffer_size, 'int64')
    if not is_scalar_tensor(shuffle_buffer_size):
        raise ValueError("`shuffle_buffer_size` must be a scalar.")
    return ds.shuffle(
        shuffle_buffer_size, reshuffle_each_iteration=reshuffle_each_iteration)

def dataset_repeat(ds, count=None):
    return ds.repeat(count=count)

def dataset_unbatch(ds):
    return ds.unbatch()

def dataset_prefetch(ds, n_prefetch=1):
    return ds.prefetch(n_prefetch)

def is_dataset_batched(ds):
    if  ds.__class__ == BatchDataset:
        return True
    for x in ds: break
    return False if any(list(map(lambda v: v.ndim == 0 if is_tensor(v) else True, x.values()))) \
         else alleq(list(map(lambda a: tf.shape(a)[0].numpy(), x.values())))

def dataset_enumerate(ds, start=0):
    return ds.enumerate(as_tensor(start, "int64"))

def safe_rescale(x, epsilon=0.05):
    epsilon = as_float_tensor(epsilon)
    is_complex = is_complex_tensor(x)
    axis = 0 if (is_scalar(x.shape.as_list())) else 1

    if is_complex:
        x = tf.bitcast(x, tf.float32)

    max_vals = tf.reduce_max(x, axis, True)
    max_vals = tf.compat.v2.where(as_float_tensor(max_vals) < epsilon,
                                tf.ones_like(max_vals), max_vals)
    x = x / max_vals

    if is_complex:
        x = tf.bitcast(x, tf.complex64)

    return x

def safe_rescale_graph(signal, axis=0, epsilon=0.05):
    epsilon = as_float_tensor(epsilon)
    is_complex = is_complex_tensor(signal)

    if is_complex:
        signal = tf.bitcast(signal, tf.float32)
    max_vals = tf.reduce_max(signal, axis, True)
    max_vals = tf.compat.v2.where(as_float_tensor(max_vals) < epsilon,
                                tf.ones_like(max_vals), max_vals)
    signal = signal / max_vals

    if is_complex:
        signal = tf.bitcast(signal, tf.complex64)
    return signal

def tf_as_scalar(x):
    if (len(tf.shape(x))):
        x = tf.squeeze(x)
        try:
            tf.assert_rank(x, 0)
        except:
            raise ValueError("Argument `x` must be of rank <= 1")
    return x

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def is_tensor(x):
    return x.__class__ in [Tensor, EagerTensor]

def is_strlike(x):
    if is_tensor(x):
        return x.dtype == tf.string
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def is_complex_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_complex

def is_float_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_floating

def is_integer_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_integer

def as_tensor(x, dtype=None):
    if x is None: return x

    if type(dtype) == str:
        dtype = tf.as_dtype(dtype)

    if is_tensor(x) and not (dtype is None):
        return tf.cast(x, dtype)
    else:
        # this can do an overflow, but it'll issue a warning if it does
        # in that case, use tf$cast() instead of tf$convert_to_tensor, but
        # at that range precision is probably not guaranteed.
        # the right fix then is tf$convert_to_tensor('float64') %>% tf$cast('int64')
        return tf.convert_to_tensor(x, dtype=dtype)

def is_empty(tensor):
    if not is_tensor(tensor):
        tensor = as_tensor(tensor)
    return tf.equal(tf.size(tensor), 0)

def as_float_tensor(x):
    return as_tensor(x, tf.float32)

def as_double_tensor(x):
    return as_tensor(x, tf.float64)

def as_integer_tensor(x):
    return as_tensor(x, tf.int32)

def as_complex_tensor(x):
    return as_tensor(x, tf.complex64)

def as_scalar_integer_tensor(x, dtype=tf.int32):
    if dtype not in [tf.int32, tf.int64, 'int32', 'int64']:
        raise ValueError("`dtype` must be integer valued")
    return tf_as_scalar(as_tensor(x, dtype=dtype))

def as_scalar_float_tensor(x):
    return tf_as_scalar(as_tensor(x, dtype=tf.float32))

def tf_assert_in_range(freq, lower=0, upper=1):
    tf.compat.v1.assert_greater_equal(freq, lower)
    tf.compat.v1.assert_less_equal(freq, upper)

def tf_assert_is_odd(x):
    k = lambda y: tf.cast(y, x.dtype)
    return tf.assert_equal(x % k(2), k(1))

def tf_assert_length(x, length):
    return tf.assert_equal(
        as_scalar_integer_tensor(length), 
        tf.reduce_prod(x.shape))

def tf_assert_length2(x, valid_lengths):
    length = tf.reduce_prod(tf.shape(x))
    valid_lengths = as_integer_tensor(valid_lengths)
    x = tf.control_dependencies([
            tf.Assert(tf.reduce_any(length == valid_lengths),
            [x])
        ])
    return x

def is_scalar(x):
    if is_tensor(x):
        return x.ndim == 0
    if isinstance(x, str) or type(x) == bytes:
        return True
    if hasattr(x, "__len__"):
        return len(x) == 1
    try:
        x = iter(x)
    except:
        return True
    return np.asarray(x).ndim == 0

def is_scalar_tensor(x, raise_err=False):
    if is_tensor(x):
        return x.ndim == 0
    if raise_err:
        raise ValueError("`x` is not a tensor")
    return False

# TODO: dtype not checked for case when dim is correct
def tf_assert_is_one_signal(x, dtype=tf.complex64):
    if not is_scalar_tensor(x):
        x = tf.squeeze(x)
    x = tf.control_dependencies([
        tf.compat.v1.assert_rank(x, 1),
        tf.compat.v1.assert_type(x, dtype)
    ])
    return x


def complex_range(x):
    R = tf.math.real(x)
    I = tf.math.imag(x)
    return { 
        'real': (tf.reduce_min(R).numpy(), tf.reduce_max(R).numpy()), 
        'imag': (tf.reduce_min(I).numpy(), tf.reduce_max(I).numpy())
    }

def tfrange(x):
    if is_complex_tensor(x):
        return complex_range(x)
    return (tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())

def list2csv(l):
    s = ''
    for x in l:
        s += str(x) + ','
    return s

def set_gpu(i):
    """Set one or more GPUs to use for training by index
        Args:
            `i` may be a list of indices or a scalar integer index
    """
    if not type(i) is list:
        i = list2csv(i)
    # ensure other gpus not initialized by tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i) 

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpus = np.asarray(gpus)
        try:
            tf.config.set_visible_devices(gpus[i], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU(s) available")

def tf_list_devices():
    return list_physical_devices()

# OLD, TF < 2.3: tf.python removed from import path, config out of experimental
# def tf_list_devices():
#     from tensorflow.python.client.device_lib import list_local_devices
#     return list_local_devices()

def tf_list_device_names(XLA=False):
    out = list(map(lambda x: x.name, tf_list_devices()))
    if not XLA:
        out = [i for i in out if not ":XLA_" in i]
    return out

def tf_count_gpus_available():
  x = tf_list_device_names()
  return len(x) - 1

def grab(dataset):
    r"""Convenient but expensive way to quickly view a batch.
        Args:
            dataset: A tensorflow dataset object.
        Returns:
            nb: dict, a single batch of data, having forced evaluation of
            lazy map calls.
    """
    return next(as_iter(dataset))

def as_iter(dataset):
    return dataset.as_numpy_iterator()

# TODO: pretty printing these dicts?
def shapes(x):
    shapes_fun = FUNCS[type(x)]
    return shapes_fun(x)

def shapes_list(l, print_=False):
    r"""Grab shapes from a list of tensors or numpy arrays"""
    shps = []
    for x in l:
        if print_:
            print(np.asarray(x).shape)
        shps.append(np.asarray(x).shape)
    return shps

def shapes_dict(d, print_=False):
    r"""Recursively grab shapes from potentially nested dictionaries"""
    shps = {}
    for k,v in d.items():
        if isinstance(v, dict):
            shps.update(shapes(v))
        else:
            if print_:
                print(k, ":\t", np.asarray(v).shape)
            shps[k] = np.asarray(v).shape
    return shps

def shapes_tuple(tup, return_shapes=False):
    shps = {i: None for i in range(len(tup))}
    for i, t in enumerate(tup):
        shps[i] = np.asarray(t).shape
    print(shps)
    if return_shapes:
        return shps

FUNCS = {
    dict: shapes_dict,
    list: shapes_list,
    tuple: shapes_tuple
}

def info(d, return_dict=False, print_=True):
    r"""Recursively grab shape, dtype, and size from (nested) dictionary of tensors"""
    info_ = {}
    lines = []
    for k,v in d.items():
        if isinstance(v, dict):
            info_.update(info(v) or {})
        else:
            info_[k] = {
                'size': tf.size(np.asarray(v)).numpy(), 
                'shape' :tf.constant(v).shape, 
                'dtype': tf.constant(v).dtype.name
            }
        l = str('\n\n' + k + ':\n' + tf.constant(v).dtype.name +\
            '\n' + ('value: ' + str(tf.constant(v).numpy())\
            if tf.constant(v).shape == () else str(tf.constant(v).shape)))
        lines.append(l)
    if print_:
        print(''.join(lines))
    if return_dict:
        return info_

def maybe_list_up(x):
    if len(tf.shape(x)) == 0:
        return [x]
    return x

def within(x, y, eps=1e-3):
    ymax = y + eps
    ymin = y - eps
    return x <= ymax and x >= ymin

def within1(x, y):
    return within(x, y, 1.)

def normalize_signal(x, b_init=12, epsilon=1e-3):
    if is_integer_tensor(x):
        x = as_float_tensor(x)
    b = b_init
    n = tf.reduce_max(x / tf.pow(2., (b_init-1)))
    m = tf.reduce_min(x / tf.pow(2., (b_init-1)))
    while not (within(n, 1, eps=epsilon) or within(m, -1, eps=epsilon)):
        n = tf.reduce_max(x / tf.pow(2., (b-1)))
        m = tf.reduce_min(x / tf.pow(2., (b-1)))
        if n < 1:
            b -= epsilon
        elif n > 1:
            b += epsilon
    return x / tf.pow(2., (b-1))
    
def find_bit_depth(x, b_init=12, epsilon=1e-3):
    if is_integer_tensor(x):
        x = as_float_tensor(x)
    b = b_init
    n = tf.reduce_max(x / tf.pow(2., (b_init-1)))
    m = tf.reduce_min(x / tf.pow(2., (b_init-1)))
    while not (within(n, 1, eps=epsilon) or within(m, -1, eps=epsilon)):
        n = tf.reduce_max(x / tf.pow(2., (b-1)))
        m = tf.reduce_min(x / tf.pow(2., (b-1)))
        if n < 1:
            b -= epsilon
        elif n > 1:
            b += epsilon
    return b

def tf_normalize_range(x, lo=-1, hi=1):
    _min = tf.reduce_min(x)
    a = x - _min
    b = tf.reduce_max(x) - _min
    c = hi - lo
    return c * (a / b) + lo

def bit_normalize(x, bit_depth=16):
    if is_integer_tensor(x):
        x = as_float_tensor(x)
    return x / tf.pow(2., (bit_depth-1))

def dim(x):
    if is_tensor(x) or x.__class__ in [np.ndarray]:
        return x.shape
    return np.asarray(x).shape

def tensor_assign_1D_graph(t, x, idx):
    """
    Pseudo 'in place' modification of tensor `t` with value `x` at index `idx`
    Really just constructs new tensor via slicing... slow, and maybe expensive.
    Param:
        t: tensor to update
        x: value to update (scalar)
        idx: index across `axis` to place `x` (scalar)
    Returns:
        ~t: new tensor with `x` updated at t.shape[axis] = idx
    """
    if not is_tensor(idx):
        idx = tf.constant(idx)
    if len(tf.shape(idx)) == tf.constant(0):
        idx = tf.expand_dims(idx, 0)
    tshape = tf.shape(t)
    hi = tshape - tf.constant(1)
    left = idx
    right = hi - idx
    _left = tf.slice(t, [0], left)
    _right = tf.slice(t, [0], right)
    _x = tf.constant(x, shape=[1], dtype=t.dtype)
    return tf.concat([_left, _x, _right], 0)

def tensor_assign(t, x, idx, axis=-1):
    """
    Pseudo 'in place' modification of tensor `t` with value `x` at index `idx`
    Really just constructs new tensor via slicing... slow, and maybe expensive.
    Param:
        t: tensor to update
        x: value to update (scalar)
        idx: index across `axis` to place `x` (scalar)
    Returns:
        ~t: new tensor with `x` updated at t.shape[axis] = idx
    """
    axis = tf.math.argmax(t.shape) if axis is None else axis
    print("axis:", axis)
    ndim = len(t.shape)
    if idx < 0: 
        raise ValueError("`idx` must be positive.")
    if hasattr(x, '__len__'):
        raise ValueError(
            "`idx` must be a scalar. Currently only support single index.")
    if idx >= t.shape[axis]:
        raise ValueError("`idx` must be <= t.shape[axis]")
    hi   = t.shape[axis] - 1
    left = t.shape.as_list()
    left[axis] = idx
    right = [l for l in left]
    right[axis] = hi-idx
    _mid   = tf.ones([1 for _ in range(len(t.shape))], dtype=t.dtype)
    _left  = tf.slice(t, [0 for _ in range(ndim)], left, t.dtype)
    _right = tf.slice(t, [0 for _ in range(ndim)], right, t.dtype)
    _x = tf.constant(x, shape=_mid.shape, dtype=t.dtype)
    return tf.concat([_left, _x, _right], axis)

# https://github.com/tensorflow/tensorflow/issues/14132#issuecomment-483002522
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/ops/image_ops_impl.py#L328-L366
# Eager tensor assignment with tensor mask
def tensor_assign_mask(t, x, idx, axis=None):
    """
    Pseudo 'in place' modification of tensor `t` with value `x` at index `idx`
    Really just constructs new tensor and multiplies with a mask
    Param:
        t: tensor to update
        x: value to update (scalar)
        idx: index across `axis` to place `x` (scalar)
    Returns:
        ~t: new tensor with `x` updated at t.shape[axis] = idx
    """
    axis = tf.math.argmax(t.shape) if axis is None else axis
    if hasattr(x, '__len__'):
        raise ValueError(
            "`idx` must be a scalar. Currently only support single index.")
    if idx >= t.shape[axis]:
        raise ValueError("`idx` must be <= t.shape[axis]")
    hi   = t.shape[axis] - 1
    left = t.shape.as_list()
    left[axis] = idx
    right = [l for l in left]
    right[axis] = hi-idx
    _mid   = tf.ones([1 for _ in range(len(t.shape))], dtype=t.dtype)
    _left  = tf.zeros(left, t.dtype) 
    _right = tf.zeros(right, t.dtype)
    mask   = tf.concat([_left, _mid, _right], axis)
    o  = tf.concat([_left, tf.constant(x, shape=_mid.shape, dtype=t.dtype), _right], axis)
    return mask * o + t * (1 - mask)
