import tensorflow as tf
from .tf_utils import is_tensor, as_tensor, as_integer_tensor, is_dataset_batched
from .utils import isTrueOrFalse


def _standardize_win_len(x):
    if is_tensor(x):
        return as_integer_tensor(x)
    x = int(x)
    return as_integer_tensor(x)

def dataset_signal_slice_windows(dataset, win_len=1024, block_length=1,
                                 pad=False, deterministic=True, 
                                 max_windows_per_signal=None,
                                 num_parallel_calls=None, data_key='signal'):
    r"""Slice Signal Windows

        Slices the signals of a tensorflow dataset produced by [signal_dataset] into
        signal windows. This will result in more samples being included in the
        dataset, so metadata for each slice of a signal will be copied to
        have the same characteristics as the input dataset. It is STRONGLY
        recommended to shuffle the dataset after calling this function.

        Args:
            dataset: a signal_dataset, as returned by `signal_dataset()`. Must not
            be batched.

            win_len: Either a scalar numeric window length or numeric vector of
                possible window lengths.

            cycle_length: Cycle length for
                \href{https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave}{tesorflow parallel interleave}.

            block_length: Block length for 
                \href{https://www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave}{tesorflow parallel interleave}.

            pad: T/F, whether the first and last window should optionally be padded 
                to the correct length. If `FALSE` (the default), then the first and 
                last window are discarded if they are not sufficiently long to fill 
                the requested `win_len`.

            deterministic: bool whether to deterministically slice windows. If
                False, then the window start positions and the the order of the windws
                returned are randomized. Useful for debugging, testing, and validation
                sets.

            max_windows_per_signal: int, if provided this limits the number of windows 
                that can be pulled from any given signal

            num_parallel_calls: passed onto `tf.data.Dataset.interleave`

            data_key: string value for key in dictionary for map call.  Default: 'signal'.
            
        Returns:
            Tensorflow dataset in which signals have been sliced into windows of
            specified length.
    """
    if is_dataset_batched(dataset):
        raise ValueError("Dataset must not be batched. \
            Please unbatch your dataset and try again!")

    if not isTrueOrFalse(deterministic):
        raise ValueError("`deterministic` must be either `True` or `False`")

    cycle_length = max_windows_per_signal or 1
    win_len = _standardize_win_len(win_len)

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def interleave_ds(x):
        signal = x[data_key]
        sig_len = tf.size(signal)

        start = 0 if deterministic else tf.random.uniform(
            [],
            minval=0,
            maxval = tf.minimum(win_len, sig_len // 2),
            dtype=tf.int32
        )

        len_partial_last_win = (sig_len - start) % win_len
        if pad:
            right_pad_len = win_len - len_partial_last_win
            left_pad_len = win_len - start

            signal = tf.concat(
                [
                    tf.zeros([left_pad_len], dtype=signal.dtype),
                    signal,
                    tf.zeros([right_pad_len], dtype=signal.dtype)
                ], axis=0
            )
            sig_len += (right_pad_len + left_pad_len)
        else:
            end = sig_len - len_partial_last_win
            signal = signal[start:end]
            sig_len -= (len_partial_last_win + start)
        
        signal = tf.reshape(signal, [-1, win_len])
        win_start_idx = tf.range(start=start, 
                                 limit=sig_len, 
                                 delta=win_len)            
        if pad:
            win_start_idx = tf.cond(
                start >= 0,
                tf.concat([
                    tf.expand_dims(-left_pad_len, 0),
                    win_start_idx
                ], 0),
                lambda: win_start_idx
            )
        if not deterministic:
            n_windows = tf.shape(signal)[0]
            idx = tf.random.shuffle(tf.range(n_windows))
            signal = tf.gather(signal, idx)
            win_start_idx = tf.gather(win_start_idx, idx)
        if max_windows_per_signal is not None:
            signal = signal[:max_windows_per_signal, ]
            win_start_idx = win_start_idx[:max_windows_per_signal]

        sigs = tf.data.Dataset.from_tensor_slices(tensors=(signal))
        idxs = tf.data.Dataset.from_tensor_slices(tensors=(win_start_idx))
        sig_and_idx = tf.data.Dataset.zip((sigs, idxs))
        n_windows = tf.shape(signal)[0]

        metadata = dict.fromkeys(x)
        metadata.pop(data_key)
        metadata = {k: x[k] for k in metadata.keys()}
        metadata['sig_len'] = sig_len

        for k,v in metadata.items():
            v = tf.cond(
                tf.equal(tf.rank(v), tf.constant(0)),
                true_fn=lambda: tf.identity(v),
                false_fn=lambda: tf.expand_dims(v, 0))

            multiples = tf.cond(
                tf.equal(tf.rank(v), tf.constant(2)),
                true_fn=lambda: tf.stack([n_windows, tf.constant(1)]), 
                false_fn=lambda: tf.expand_dims(n_windows, 0))

            rep = tf.cond(
                tf.equal(tf.rank(v), tf.constant(0)),
                true_fn=lambda: tf.repeat(v, multiples),
                false_fn=lambda: tf.tile(v, multiples))

            metadata[k] = tf.cond(
                tf.equal(tf.size(v), 0),
                true_fn=lambda:  tf.reshape(rep, [n_windows, -1]),
                false_fn=lambda: tf.identity(rep))
        metadata = tf.data.Dataset.from_tensor_slices(metadata)
        return tf.data.Dataset.zip((sig_and_idx, metadata))

    interleaved_ds = dataset.interleave(
        interleave_ds,
        cycle_length=as_tensor(cycle_length, tf.int64),
        block_length=as_tensor(block_length, tf.int64),
        num_parallel_calls=as_tensor(num_parallel_calls, tf.int64))
    
    def map_idx(x, metadata):
        signal = x[0]
        win_start_idx = x[1]
        x = {
            data_key: signal, 
            'win_start_idx': win_start_idx,
            'win_len': win_len
        }
        x.update(metadata)
        return x

    return interleaved_ds.map(map_idx)


if False:
    win_len=320000
    block_length=1
    pad=False
    deterministic=True
    max_windows_per_signal=None
    num_parallel_calls=None
    data_key='signal'

    from tf_dataset import *
    from fr_train import *
    df = dataset.select_dataframe()
    ds = signal_dataset(df, use_soundfile=True)
    for x in ds: break
    dsw = dataset_signal_slice_windows(ds, win_len)
    for y in dsw: print(y['target'])