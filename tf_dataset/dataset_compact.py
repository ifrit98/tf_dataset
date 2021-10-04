import tensorflow as tf
from .tf_utils import as_tensor, is_scalar


# Forces tf.data.Dataset object to be keras.Model.fit() compatible: An (x, y) tuple.
def dataset_compact(ds, compact_x_key='signal', compact_y_key='target', num_parallel_calls=4):
    """Condense dataset object down to contain only (x, y) tuple pairs.

    Args:
        ds: A dataset (tf.data.Dataset) object

        compact_x_key: string or iterable of keys for `x` (input) tensors.

        compact_y_key: string or iterable of keys for `y` (output) tensors.

    pair = (x['signal'], x['target'])
    
    This is to be called immedately preceeding passing the dataset to 
    `tf.keras.Model.fit()`:
        import tensorflow as tf
        from tf_dataset import *
        
        input = tf.keras.Input(shape=[1024], dtype='float32')
        output = tf.keras.layers.Dense(input, activation='softmax')
        model = tf.keras.Model(input, ouput)
        model.compile(...)

        data_dir = "./data"
        df = construct_metadata(data_dir)
        ds = signal_dataset(df)
        ds = dataset_compact(ds, x_key='signal', y_key='target')

        h = model.fit(ds, ...)
    """
    def compact(x):
        nonlocal compact_x_key, compact_y_key
        x_scalar = is_scalar(compact_x_key)
        y_scalar = is_scalar(compact_y_key)

        if x_scalar:
            if y_scalar:
                return (x[compact_x_key], x[compact_y_key])
            else:
                return (x[compact_x_key], tuple(x[k] for k in compact_y_key))
        else:
            if y_scalar:
                return (tuple(x[k] for k in compact_x_key), x[compact_y_key])
            else:
                return (tuple(x[k] for k in compact_x_key), tuple(x[k] for k in compact_y_key))

    return ds.map(
        compact, 
        num_parallel_calls=as_tensor(num_parallel_calls, 'int64'))
