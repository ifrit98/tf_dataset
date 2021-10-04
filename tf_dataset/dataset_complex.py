import numpy as np
import tensorflow as tf
from .tf_utils import as_tensor, as_complex_tensor, as_float_tensor, tensor_assign_1D_graph


def tf_analytic(x):
    x = as_complex_tensor(x)
    shp = tf.shape(x)
    X = tf.signal.fft(x) / tf.cast(shp[0], tf.complex64)
    xf = as_float_tensor(tf.range(0, shp[0]))
    xf = xf - tf.reduce_mean(xf)
    X = X * as_complex_tensor((1 - tf.sign(xf - 0.5)))
    ifft = tf.signal.ifft(x)
    return ifft

def dataset_signal_apply_analytic(dataset, num_parallel_calls=4, 
                                  data_key='signal'):
    r"""Compute the analytic signal (i.e. move to the complex plane) using
        the hilbert transform.

        Args:
            dataset: Tensorflow dataset of signals and associated metadata.
            
            num_parallel_calls: -- passed onto `tf.data.Dataset.interleave`
        
            data_key: string value for key in dictionary for map call.  
                Default: 'signal'.

        Returns:
            Tensorflow dataset in which signals now have dtype == 'complex64'
    """
    def map_analytic(x):
        x_copy = x.copy()
        del x
        x = x_copy
        x[data_key] = tf_analytic(x[data_key])
        return x
        
    return dataset.map(
        tf.function(map_analytic), 
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))

# Expects (1, samples) or (samples,)
# Adaped from scipy.signal.hilbert()
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
def tf_hilbert(x, axis=-1):
    r"""Compute the analytic signal (i.e. move to the complex plane) using
        the hilbert transform.

        Args:
            dataset: Tensorflow dataset of signals and associated metadata.
            
            num_parallel_calls: -- passed onto `tf.data.Dataset.interleave`
        
            data_key: string value for key in dictionary for map call.  
                Default: 'signal'.

        Returns:
            Tensorflow dataset in which signals now have dtype == 'complex64'
    """
    Xf = tf.signal.fft(as_complex_tensor(x))
    h = tf.zeros_like(x)
    n = tf.shape(x)[axis]
    size = (n // 2) if (n % 2) == 0 else ((n + 1) // 2)
    if n % 2 == 0:
        h = tf.concat([tf.ones([size], dtype=x.dtype) + 1, h[:size:]], 0)
        h = tensor_assign_1D_graph(h, 1, 0)
        h = tensor_assign_1D_graph(h, 1, tf.math.floordiv(n, 2))
    else:
        h = tf.concat([tf.ones([size], dtype=x.dtype) + 1, h[size:]], 0)
        h = tensor_assign_1D_graph(h, 1, 0)
    return tf.signal.ifft(Xf * as_complex_tensor(h))

def dataset_signal_apply_hilbert(dataset, num_parallel_calls=None, data_key='signal'):
    def map_hilbert(x):
        x[data_key] = tf_hilbert(x[data_key])
        return x
    return dataset.map(map_hilbert)
    

if False:
    from tf_dataset import *
    ds = signal_dataset(construct_metadata("./data"), use_soundfile=True)
    ds = dataset_signal_slice_windows(ds, 30000)
    dsa = dataset_signal_apply_analytic(ds)
    for a in dsa: print(a); break

