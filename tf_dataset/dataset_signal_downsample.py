import tensorflow as tf
from scipy.signal import resample_poly
from .tf_utils import as_tensor, is_dataset_batched, is_complex_tensor

@tf.autograph.experimental.do_not_convert
def dataset_signal_downsample(dataset, 
                              du=640, 
                              su=50, 
                              data_key='signal', 
                              num_parallel_calls=1):
    batched = is_dataset_batched(dataset)

    for x in dataset:
        if is_complex_tensor(x[data_key]):
            raise NotImplementedError(
                "data must be real valued.  Cannot be complex.")
        break
    del x

    resamp = lambda x: as_tensor(resample_poly(x, su, du), dtype=x.dtype)

    def map_fn(x):
        _x = x.copy()
        del x
        x = _x
        signal = x[data_key]
        if not batched:
            signal = tf.expand_dims(signal, 0)
        x[data_key] = tf.py_function(
            lambda x: tf.squeeze(
                tf.map_fn(resamp, x)), inp=[signal], Tout='float32')
        x['fs'] = tf.cast(tf.cast(x['fs'], 'float64') * (1./(du/su)), 'int32')
        return x

    return dataset.map(
        tf.function(map_fn),
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))


if False:
    import tensorflow as tf
    from tf_dataset import *

    downsample=True
    du = 640
    su = 50
    x = tf.complex(tf.random.normal([128000]), tf.random.normal([128000]))
    xb = tf.complex(tf.random.normal([4, 128000]), tf.random.normal([4, 128000]))

    from tf_dataset import *
    ds = signal_dataset(
        construct_metadata('../tf_dataset/data'), use_soundfile=True)
    ds = dataset_signal_slice_windows(ds, 128000)
    ds = dataset_signal_downsample(ds)

    for x in ds:
        break

    print(x)

