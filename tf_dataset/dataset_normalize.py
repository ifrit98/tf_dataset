from .tf_utils import as_integer_tensor, is_complex_tensor, as_float_tensor, is_integer_tensor
from .tf_utils import as_tensor, is_tensor, is_scalar, is_strlike
from .utils import switch

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import normalize, scale, quantile_transform, RobustScaler

permute = tf.keras.backend.permute_dimensions

def tf_normalize_signal_gain(signal, target_abs_max=1.):
    if not is_tensor(signal):
        signal = as_tensor(signal)
    if not (signal.dtype.is_complex or signal.dtype.is_floating):
        signal = as_float_tensor(signal)

    max_vals = tf.reduce_max(tf.abs(signal), axis=0)
    max_vals = tf.where(max_vals == 0,
                        tf.ones(list(), max_vals.dtype),
                        max_vals)
    scale_factor = target_abs_max / max_vals
    scale_factor = tf.cast(scale_factor, signal.dtype)
    return signal * scale_factor

# remove dc component
def tf_mean_norm_channelwise(x):
    m = tf.map_fn(lambda x: tf.reduce_mean(x), x)
    return x - tf.expand_dims(m, 1)

def tf_mean_norm(x):
    return x - tf.reduce_mean(x)

def tf_median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])

# Expects (batch, samples)
def tfp_median(x, reduce_median=True):
    if len(x.shape) >= 2 and not reduce_median:
        percentile = lambda x: tfp.stats.percentile(
            x, 50.0, interpolation='midpoint')
        return tf.map_fn(percentile, x)
    return tfp.stats.percentile(x, 50.0, interpolation='midpoint')

def tf_minmax_scale(x):
    _min = tf.reduce_min(x)
    _max = tf.reduce_max(x)
    C = tf.subtract(_max, _min)
    return tf.divide(tf.subtract(x, _min), C)

# z-score normalize
def tf_standardize(x):
    if len(x.shape) == 3:
        mus = tf.map_fn(lambda x: tf.math.reduce_mean(x), x)
        stds = tf.map_fn(lambda x: tf.math.reduce_std(x), x)
        return permute((permute(x, [2, 1, 0]) - mus) / stds, [2, 1, 0])
    mu = tf.math.reduce_mean(x)
    std = tf.math.reduce_std(x)
    return tf.divide(tf.subtract(x, mu), std)
z_norm = tf_standardize

def tf_median_norm(x):
    x_rank = len(x.shape)
    permute = tf.keras.backend.permute_dimensions
    if x_rank == 1:
        # (samples,)
        x = tf.expand_dims(x, -1)
        scaled = x - tfp_median(x, reduce_median=True)
        return tf.squeeze(scaled)
    if x_rank == 2:
        if x.shape[0] > x.shape[1]:
            # (samples, features)
            med = tfp_median(permute(x, [1, 0]), reduce_median=False)
            scaled = x - med
        else:
            # (features, samples)
            med = tfp_median(x, reduce_median=False)
            scaled = permute(x, [1, 0]) - med
        return permute(scaled, [1, 0])
    if x_rank == 3:
        # (batch, samples, features)
        perm = [2, 1, 0]
        medians = tf.map_fn(lambda x: tfp_median(permute(x, [1,0])), x)
        scaled = permute(tf.subtract(permute(x, perm), medians), perm)
    return x - tfp_median(x, reduce_median=True)

# expects (n_samples, n_features)
# TODO: Implement 4D Code path for (batch, freq, frames, features)
# TODO: 2D and 3D ARE WRONG! (Must compute across each individual sample, nested tf.map_fn())?
def tf_robust_scale(x):
    x_rank = len(x.shape)
    permute = tf.keras.backend.permute_dimensions
    f = lambda a: RobustScaler().fit_transform(a)
    # Unbatched 1D signal
    if x_rank == 1:
        x = tf.expand_dims(x, -1) # (samples, 1)
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # Potentially unbatched 2D signal features (samples, features)
    if x_rank == 2:
        if x.shape[0] < x.shape[1]:
            x = permute(x, [1, 0])
            return permute(
                tf.squeeze(tf.py_function(f, inp=[x], Tout='float32')), 
                [1, 0])
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # batched 3D signal (batch, samples, features)
    if x_rank == 3:
        f_batched = lambda b: tf.map_fn(f, b)
        return tf.squeeze(
            tf.py_function(f_batched, inp=[x], Tout=x.dtype.name))

# About 5x slower than robust_scaler
# TODO: Implement 4D Code path for (batch, freq, frames, features)
# TODO: rework so there is a separate tf_quantile_norm, and py_quantile_norm for graph mode.
def tf_quantile_norm(x):
    x_rank = len(x.shape)
    f = lambda a: quantile_transform(a)
    # Unbatched 1D signal
    if x_rank == 1:
        x = tf.expand_dims(x, -1) # (samples, 1)
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # Potentially unbatched 2D signal features (samples, features)
    if x_rank == 2:
        if x.shape[0] < x.shape[1]:
            x = permute(x, [1, 0])
            return permute(
                tf.squeeze(tf.py_function(f, inp=[x], Tout='float32')), 
                [1, 0])
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # batched 3D signal (batch, samples, features)
    if x_rank == 3:
        f_batched = lambda b: tf.map_fn(f, b)
        return tf.squeeze(
            tf.py_function(f_batched, inp=[x], Tout=x.dtype.name))

def tf_log_norm(x, center=False, absolute=True, repl_with='median'):
    if center:
        mn = tf.reduce_min(x)
        if tf.sign(mn) == -1:
            x = x - mn
        else:
            x = x + mn
    if absolute:
        x = tf.abs(x)
    return replace_nan_inf(tf.math.log(x), tfp_median(x))

def tf_replace_nan(x, rpl=0):
    return tf.where(
        tf.math.is_nan(x), 
        tf.zeros_like(x) if rpl == 0 else tf.zeros_like(x) + rpl, x)

def tf_replace_nan2(x):
    value_not_nan = tf.dtypes.cast(
        tf.math.logical_not(tf.math.is_nan(x)), dtype=x.dtype)
    return tf.math.multiply_no_nan(x, value_not_nan)

def tf_replace_inf(x, rpl=0):
    return tf.where(
        tf.math.is_inf(x), 
        tf.zeros_like(x) if rpl == 0 else tf.zeros_like(x) + rpl, x)

def log10(x):
    absolute = tf.math.abs(x)
    n = tf.math.log(absolute)
    d = tf.math.log(tf.constant(10, dtype=n.dtype))
    return n / d

def apply_in_real_space(x, f):
    if x.dtype not in [tf.complex64, tf.complex128]:
        raise ValueError("`x` must be a complex valued tensor")
    return tf.complex(real=f(tf.math.real(x)), imag=f(tf.math.imag(x)))
apply_complex = apply_in_real_space

def apply_bitcast(x, f):
    x = cplx1D_to_float2D(x)
    return float2D_to_cplx1D(f(x))

def cplx1D_to_float2D(x):
    if x.dtype not in [tf.complex64, tf.complex128]:
        raise ValueError('Expected a complex type...')
    dtype = tf.float64 if x.dtype == tf.complex128 else tf.float32    
    return tf.bitcast(x, dtype)

def float2D_to_cplx1D(x):
    if x.dtype not in [tf.float32, tf.float64]:
        raise ValueError('Expected a floating type...')
    dtype = tf.complex128 if x.dtype == tf.float64 else tf.complex64
    return tf.bitcast(x, dtype)

def replace_nan(x, rpl=0):
    if x.dtype in [tf.complex64, tf.complex128]:
        return apply_in_real_space(x, tf_replace_nan)
    return tf.where(
        tf.math.is_nan(x), 
        tf.zeros_like(x, x.dtype) if rpl == 0 else tf.zeros_like(
            x, x.dtype) + tf.cast(rpl, x.dtype), x)

def replace_inf(x, rpl=0):
    if x.dtype in [tf.complex64, tf.complex128]:
        return apply_in_real_space(x, tf_replace_nan)
    return tf.where(
        tf.math.is_inf(x), 
        tf.zeros_like(x, x.dtype) if rpl == 0 else tf.zeros_like(
            x, x.dtype) + tf.cast(rpl, x.dtype), x)

def replace_nan_inf(x, rpl=0.):
    if x.dtype in [tf.complex64, tf.complex128]:
        f = lambda y: replace_nan(replace_inf(y, rpl), rpl)
        return apply_in_real_space(x, f)
    return replace_nan(replace_inf(x, rpl), rpl)

def any_nan(x, dtype=None):
    if x.dtype == tf.complex64:
        dtype = tf.float32
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.imag(x))
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    if x.dtype == tf.complex128:
        dtype = tf.float64
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.imag(x))    
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    return tf.math.reduce_any(
        tf.cast(tf.map_fn(lambda x: tf.cast(tf.math.is_nan(x), x.dtype), x), 'bool'))

def any_inf(x, dtype=None):
    if x.dtype.name == tf.complex64:
        dtype = tf.float32
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.imag(x))
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    if x.dtype.name == tf.complex128:
        dtype = tf.float64
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.imag(x))    
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    return tf.math.reduce_any(
        tf.cast(tf.map_fn(lambda x: tf.cast(tf.math.is_inf(x), x.dtype), x), 'bool'))

def any_nan_or_inf(x, dtype=None):
    return any_nan(x, dtype=dtype) or any_inf(x, dtype=dtype)

def dataset_signal_normalize(dataset,
                             norm_type='mean', 
                             num_parallel_calls=1, 
                             data_key='signal'):
    norm_fn = switch(
        on=norm_type,
        pairs={
            'mean': tf_mean_norm,
            'mean_channelwise': tf_mean_norm_channelwise,
            'minmax': tf_minmax_scale,
            'standard': tf_standardize,
            'median': tf_median_norm,
            'quantile': tf_quantile_norm,
            'log': tf_log_norm,
            'gain': tf_normalize_signal_gain,
        },
        default=tf_minmax_scale
    )

    def map_normalize(x):
        _x = x.copy()
        del x
        x = _x
        x [data_key] = tf.py_function(
            norm_fn, inp=[x[data_key]], Tout='float32')
        return x
    return dataset.map(
        tf.function(map_normalize),
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))


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

def dataset_signal_safe_rescale(dataset, num_parallel_calls=None, data_key='signal'):
    r"""Safe Rescale

        Rescales the values of signals of a tensorflow dataset produced by [signal_dataset].

        Args:
            dataset: a signal_dataset, as returned by `signal_dataset()`. Must not
            be batched.

            num_parallel_calls: -- passed onto `tf.data.Dataset.interleave`
        
            data_key: string value for key in dictionary for map call.  Default: 'signal'.

        Returns:
            Tensorflow dataset in which signals have been rescaled by x / max
    """
    def map_rescale(x):
        x[data_key] = safe_rescale(x[data_key])
        return x
    return dataset.map(
        map_rescale,
        num_parallel_calls=as_integer_tensor(num_parallel_calls))

