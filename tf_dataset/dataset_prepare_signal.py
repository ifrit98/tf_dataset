import tensorflow as tf
import numpy as np
import warnings
from scipy.signal import resample_poly

from .dataset_complex import dataset_signal_apply_analytic
from .tf_utils import safe_rescale_graph, as_tensor, as_float_tensor, is_complex_tensor
from .tf_utils import as_complex_tensor, is_dataset_batched


def upcast_complex_to_float(x, dtype='float64'):
    return tf.cast(tf.bitcast(x, 'float32'), dtype)

# Best for power values (analog)
def prepare_signal_for_ml_lite(x, 
                               batched=False, 
                               use_inst_vals=False, 
                               use_norm_ctr=True,
                               real_valued=False):
    if not batched:
        x = tf.expand_dims(x, 0)
        axis = 0
    else:
        axis = 1

    out = list()
    if x.dtype in [tf.complex64, tf.complex128]:
        out.append(tf.stack([tf.math.real(x), tf.math.imag(x)], -1))
    else:
        real_valued = True
        out.append(tf.expand_dims(x, -1))
    
    if use_inst_vals:
        inst_amp = tf.math.abs(x)
        inst_pow = tf.pow(inst_amp, 2)
        inst_ph = tf.math.angle(x)
        out.append(tf.stack([inst_amp, inst_pow, inst_ph], -1))

    if use_norm_ctr:
        def norm_centered_amplitude(y):
            res = tf.cast(tf.abs(y), 'float64')
            res = res / tf.maximum(tf.reduce_min(res), tf.reduce_max(res))
            return res - tf.reduce_mean(res)
        nca = norm_centered_amplitude(x)
        nca = tf.expand_dims(nca, -1)
        out.append(tf.cast(nca, out[0].dtype))

    return tf.squeeze(tf.concat(out, -1))

# Best for RF/IQ data
def prepare_signal_for_ml(x, domain=['time', 'frequency'], batched=False):
    if not batched:
        x = tf.expand_dims(x, 0)
        axis = 0
    else:
        axis = 1

    real_valued = False
    if x.dtype not in [tf.complex64]:
        real_valued = True

    out = list()
    if "time" in domain:
        inst_amp = tf.math.abs(x)
        inst_pow = tf.pow(inst_amp, 2)
        inst_ph = tf.math.angle(tf.cast(x, 'complex128')) 
        amplitude_epsilon = 1e-3 # 0.05
        inst_freq_clamp = 1.

        # resale to -1:1
        inst_amp = safe_rescale_graph(inst_amp, axis)
        inst_pow = safe_rescale_graph(inst_pow, axis)
        inst_ph = inst_ph / tf.constant(np.pi, dtype='float64')

        # zero out phase where there is no inst_amplitude
        inst_ph = tf.compat.v2.where(inst_amp < amplitude_epsilon,
                                        tf.zeros_like(inst_ph),
                                        inst_ph)
        inst_freq = inst_ph[:, 1:] - inst_ph[:, :-1]

        # mask spikes in inst freq in case there are sharp discontinuities
        # default leave it already with a range of -1:1, but if
        # a non-default inst_freq_clamp is supplied, rescale to ensure -1:1
        if inst_freq_clamp is not None:
            inst_freq = tf.compat.v2.where(
                tf.abs(inst_freq) > inst_freq_clamp, 
                tf.constant(0., dtype='float64'), inst_freq)
            if inst_freq_clamp != 1:
                inst_freq = inst_freq / (inst_freq_clamp + 1e-10)

        # instead of padding, would a tf.concat() be faster?
        inst_freq = tf.pad(inst_freq, paddings=[[0, 0], [0, 1]])

        inst_ph   = tf.expand_dims(inst_ph, -1)
        inst_amp  = tf.expand_dims(inst_amp, -1)
        inst_freq = tf.expand_dims(inst_freq, -1)
        inst_pow  = tf.expand_dims(inst_pow, -1)

        inst_features = [inst_ph, inst_amp, inst_freq, inst_pow]
        inst_features = list(map(lambda t: tf.cast(t, 'float32'), inst_features))
        inst_features = tf.concat(inst_features, -1)
        out.append(inst_features)

        # Center and normalize amplitude values in addition to inst values
        def norm_centered_amplitude(y):
            res = tf.cast(tf.abs(y), 'float32')
            res = res / tf.maximum(tf.reduce_min(res), tf.reduce_max(res))
            return res - tf.reduce_mean(res)

        nca = norm_centered_amplitude(x)
        nca = tf.expand_dims(nca, -1)
        ctr = x - tf.reduce_mean(x)
        ctr = tf.expand_dims(tf.cast(ctr, 'float32'), -1) \
            if real_valued else tf.bitcast(x, 'float32')
        out.append(tf.concat([ctr, nca], -1))

    if "frequency" in domain:
        fd = tf.signal.rfft(
            x, [tf.shape(x)[-1]*2-1]) if real_valued else tf.signal.fft(x)
        freq_arr = tf.bitcast(fd, "float32")

        # TODO; rework this to compute energy from tf.sqrt(power) to
        # save a duplicate node on the graph
        energy = tf.cast(tf.abs(fd), 'float32')
        power  = tf.pow(energy, 2)  # tf.conj(fd) * fd,
        phase  = tf.cast(tf.math.angle(fd), 'float32')

        energy = tf.expand_dims(energy, -1)
        power  = tf.expand_dims(power, -1)
        phase  = tf.expand_dims(phase, -1)

        freq_features = tf.concat([freq_arr, energy, power, phase], -1)
        out.append(freq_features)

    out = tf.concat(out, -1)

    return tf.squeeze(out)


def dataset_prepare_signal_for_ml(dataset, 
                                  keep_signal=True, 
                                  use_lite=True,
                                  num_parallel_calls=4, 
                                  data_key='signal'):
    """Prepare analytic signal for use in deep learning models.

    Extracts time and frequency domain information, bitcasts to float32, and
    then concatenates the resultant tensor, ready for entry into models.

    E.g. Single signal example:
        sig = tf.random.normal([8192], dtype=tf.float32)
        print(sig.shape)
        >>> (8192,)
        x = transform_signal_for_dl(sig)
        print(x.shape)
        >>> (8192, 12)

    Dataset example:
        from tf_dataset import *
        ds = signal_dataset(df)
        ds = dataset_signal_slice_windows(ds, win_len=1024)
        ds = dataset_transform_signal_for_dl(ds, use_lite=True)
        ds = dataset_batch(ds, 8)
        nb = grab(ds)
        print(nb['signal_features'].shape)
        >>> (8, 1024, 12)
    """
    batched = is_dataset_batched(dataset)
    new_key = data_key + '_features' if keep_signal else data_key

    prepare_signal = prepare_signal_for_ml_lite if use_lite else \
        prepare_signal_for_ml

    def transform(x):
        x[new_key] = prepare_signal(x[data_key], batched=batched)
        return x
    return dataset.map(
        transform, 
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))


if False:
    import os
    import json
    import itertools
    from numpy import asarray, ndarray, complex64
    import tensorflow as tf
    from tempfile import NamedTemporaryFile
    from tf_dataset import *
    import tensorflow as tf
    from tf_dataset.record_dataset import *
    from tf_dataset.record_dataset import _build_json_dict
    
    import tensorflow as tf
    from tf_dataset import *
    df = construct_metadata("./data")
    ds = signal_dataset(df, use_soundfile=True)
    ds = dataset_signal_slice_windows(ds, 16000)
    # ds = dataset_signal_normalize(ds)
    # ds = dataset_signal_apply_analytic(ds)
    ds = dataset_prepare_signal_for_ml(ds)
    for x in ds: break
    print(x)

    for x in ds.as_numpy_iterator():
        print(x)
        break
    config = _build_json_dict(x)
    
    output_dir=record_dir="./records"
    output_dir = os.path.abspath(output_dir)
    record_dataset(ds, "./records", elements_per_file=4)
    dsr = replay_dataset(record_dir)
    for x in dsr.take(1):
        print(x)

    ds = dataset_from_dir("./data")
    dsc = dataset_compact(ds, 'signal', 'target')

