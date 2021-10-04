# tf_dataset


`Tested with Python >= 3.6.8`

`tf_dataset` is a python module inspired by the `signal.dataset` package in R for creating tensorflow datasets from metadata to be used in machine learning pipelines.

Contains functionality to create a `tf.data.Dataset` object from (1) a metadata dictionary as a `pd.DataFrame` object, or (2) a `dirpath` to data files, for use with ML pipelines.  

`tf_dataset` also contains functions to efficiently parallel map over `tf.data.Dataset` objects, lazily applying arbitrary ops to each element.  This code is extensible (in a functional sense), so that users may define their own map calls (see [user-defined-maps](doc/loading-and-mapping.md)) and may even define their own callables (see [user-defined-loads](doc/loading-and-mapping.md)) for loading data to suit their needs.


## Installation
  ```{bash}
  git clone https://github.com/ifrit98/tf_dataset.git
  cd tf_dataset && pip install .
  ```


## Usage

There are four main entry points we direct users to, which will have what you need for most use cases.  Below is a short overview of these main functions. If you have more specialized needs, head to [advanced-example](#advanced-example).

### construct-metadata
This function takes a path to the directory where your data lives and a file extension, such as `.wav`.  It assumes the data is all at the top-level. (i.e. no nested levels).  May pass a `labels` dictionary mapping, containing class labels (if a classification problem) associated with its respective data filepath in `data_dir`.
    E.g.:
      ```{python}
    labels =  {
      'data/signal123.wav': 'cargo',
      'data/signal456.wav': 'tug',
      ...,
      'data/signal666.wav': 'whale'
    }

    df = construct_metadata("./data", ext=".wav", labels=labels)
    print(df.head())
    ```
If no labels dict is passed, `construct_metadata()` will attempt to parse labels using a regex, extracting the capture group from the last underscore `_` to the file extension, e.g. `.wav`.
```{python}
# Example filepath:
fp = "data/signal123_blah_blah_cargo.wav"
extracted_label = rexeg_filter(fp)
print(extracted_label)
>>> "cargo"
```
NOTE: If your data filepaths do not follow this naming convention or did not pass a `labels` dict mapping, then inferred labels will be meaningless.


If you have regressed values as targets, as opposed to categorical, you may pass a `np.ndarray` containing the regressed `Y` values for each example to `targets`:

```{python}
df = construct_metadata("./data", create_targets=False, targets=[0,0,1,2])
```

### signal-dataset

`signal_dataset()` accepts a pandas dataframe minimally containing columns: ['filepath', 'class'], and possibly other assocaited metadata stored as additional columns (such as is returned by `construct_metadata()`).
```{python}
df = construct_metadata("./data")
print(df.head())
>>>                                             filepath      class  target  num_classes
>>> 0  C:\internal\tf_dataset\data\1561965276.252...  passenger       0            4
>>> 1  C:\internal\tf_dataset\data\1561970414.69_...  passenger       0            4
>>> 2  C:\internal\tf_dataset\data\1561971147.625...      cargo       1            4
>>> 3  C:\internal\tf_dataset\data\1561971697.731...        tug       2            4

ds = signal_dataset(df)
```

Instead of using the default, it is also possible to pass a user-defined data loading function, named `process_db_example` to `signal_dataset()`. See [signal-dataset-demo](doc/signal-dataset-demo.md) and [loading-and-mapping](doc/loading-and-mapping.md).

### training-dataset
`training_dataset` is designed to be the standard front-end for converting your metadata into training datasets that can be passed to `tf.keras.Model.fit()` all in one go.  This allows you to control parameters like `batch_size`, `win_len`, `shuffle_buffer_size`, `prefetch_buffer_size`, `infinite` (e.g. to repeat the dataset infinitely for training). etc, from one function call.

This function takes a metadata dataframe containing data filepaths and associated metadata, and returns a `tf.data.Dataset` object, a lazily evaluated graph object that can be run in eager mode by simply iterating over the dataset using a `for` construct:
```{python}
ds = training_dataset(df)
```

Where `x` is a python dictionary whose entries contain all relevant data and metadata for a single example.

If `process_db_example` is not passed (i.e. defaults to `None`), then signals will be loaded as 1D arrays with `dtype == int32`.


### dataset-from-dir

You may also create a tensorflow dataset directly from a directory using `dataset_from_dir`, following certain constraints.

They are:

1. Files in `data_dir` must be at the top-level.  (No nested dirs with data). For example:
```
data_dir
│   README.md
│   signal001.wav
│   signal002.wav
│   signal003.wav
│   signal004.wav
|   ...
│   signal022.wav
```
2. You must do ONE of the following:
- Pass a python `dict`, mapping filepaths to your data with their associated labels.

```{python}
import os
import numpy as np
from tf_dataset import dataset_from_dir

data_dir = "./data"

files = os.listdir(os.path.abspath(data_dir))
print(files)
>>> ['1561965276.252_1561965582.106_24346_passenger.wav',
   >>>  '1561970414.69_1561970702.051_24346_passenger.wav',
   >>>  '1561971147.625_1561971202.077_24346_cargo.wav',
   >>>  '1561971697.731_1561971718.283_24346_tug.wav']

label_arr =  ['passenger', 'passenger', 'cargo', 'tug']

labels = dict(zip(files, label_arr))
print(labels)
>>> {'1561965276.252_1561965582.106_24346_passenger.wav': 'passenger',
>>>  '1561970414.69_1561970702.051_24346_passenger.wav': 'passenger',
>>>  '1561971147.625_1561971202.077_24346_cargo.wav': 'cargo',
>>>  '1561971697.731_1561971718.283_24346_tug.wav': 'tug'}
```

- Pass a `np.ndarray` (or python list), containing only the labels for each file in `data_dir`.

```{python}
labels = np.asarray(['passenger', 'passenger', 'cargo', 'tug'])
```

- Pass `None` (default). Filepaths must be of the form `"*_label.ext"`
e.g. `"signal_123_cargo.wav"` -> `"cargo"`

```{python}
labels = None
```

If your data targets are not categorically valued, you must pass a `targets` list or numpy array containing the (regressed or otherwise) values:
```{python}
targets = [0.5, 0.1, 0.9, 0.33]
ds = dataset_from_dir("./data", targets=targets)
ds = dataset_compact(ds, 'signal', 'target')
```

Note that you may pass any arguments here that will be passed to `training_dataset()`, such as
`batch_size`, `win_len`, etc.  See [training-dataset](#training-dataset).

Now, using any of the methods above to create `labels`, we can create the dataset:
```{python}
ext = ".wav"
ds = dataset_from_dir(data_dir, labels=labels, ext=ext, batch_size=2)
```


## advanced-example
The main design principle behind `tf_dataset` is a functional API, much like the `tf.keras` functinal API, in which `tf.data.Dataset` objects can be easily manipulated and users may apply whatever impariments or transformations required.  Users may create their own `map` calls to apply over the dataset object.  See [user-defined-maps](doc/loading-and-mapping.md)
```{python}
from os import path
import pandas as pd
import tensorflow as tf
import tf_dataset as tfd

d = {
    'filepath': [
        'data/1561965276.252_1561965582.106_24346_passenger.wav',
        'data/1561970414.69_1561970702.051_24346_passenger.wav',
        'data/1561971147.625_1561971202.077_24346_cargo.wav',
        'data/1561971697.731_1561971718.283_24346_tug.wav'],
    'class': ['passenger', 'passenger', 'cargo', 'tug']
    }
df = pd.DataFrame.from_dict(d)
df['filepath'] = df['filepath'].apply(lambda x: os.path.abspath(x))

ds = tfd.signal_dataset(df) # <-- one line to create your basic tf dataset!

# Normalize gain to [-1, 1]
ds = tfd.dataset_signal_normalize(ds)
for x in ds.take(1):
    print(x, "\n")
    print("min:", tf.reduce_min(x['signal']))
    print("max:", tf.reduce_max(x['signal']))

# Slice signals to a specified window length
ds = tfd.dataset_signal_slice_windows(ds, win_len=8192)
# `x` dict now contains `win_start_idx` as metadata and the windowed signal.
for x in ds.take(1):
    print(x)
    print(x['win_start_idx'])

# Move to complex plane using hilbert transform
ds = tfd.dataset_signal_apply_hilbert(ds)
for x in ds.take(1):
    print(x, "\n")
    print("signal dtype:", x['signal'].dtype)

batch_size = 8
ds = ds.batch(batch_size)
for x in ds.take(1):
    print(x) # `x` now batched e.g. x.shape == (8, 8192)
    print("\nsignal shape:", x['signal'].shape)
    
# Reduce dataset to (x,y) tuple pairs for use with `tf.keras.Model`s.
ds = dataset_compact(ds, 'signal', 'target')

val_ds_size = 5
val_ds = ds.take(val_ds_size)
...
# To be used with tensorflow/keras models
h = model.fit(ds, validation_data=val_ds)
```

NOTE: `training_dataset()` applies all of these map calls by default, tunable by function params (See docstring), such that:
```{python}
df = construct_metadata("./data")
ds = training_dataset(df, win_len=8192, batch_size=8)
```

Which is equivalent to using the functional API like so:
```{python}
df = construct_metadata("./data")
ds = tfd.signal_dataset(df)
ds = tfd.dataset_signal_slice_windows(ds, win_len=8192)
ds = tfd.dataset_signal_apply_hilbert(ds)
ds = tfd.dataset_signal_normalize(ds)
ds = ds_batch(ds, batch_size=8)
ds = ds_prefetch(ds, 1)
```

Which is also equivalent to:
```{python}
ds = dataset_from_dir("./data")
```
This further convenience combines both previous steps into one call from `dataset_from_dir()`, which takes `kwargs` that will be passed to `training_dataset()`.


## advanced-usage-links
`dataset_compact()`, compacts the python `dict`s in a dataset object down into `(x,y)` tuple pairs.
See [compact](doc/compact.md).

A [demo](doc/signal-dataset-demo.md) of how `signal_dataset()` works under the hood.

For a tutorial on user defined map and loading [functions](doc/loading-and-mapping.md).

For a tutorial on [writing-reading](doc/writing-reading-datasets.md) datasets.


### utilities
A submodule named `tf_utils` is accessible from `tf_dataset` and contains functions useful for working in tensorflow.  Feel free to explore and have fun!
```{python}
import tf_dataset as tfd
from tf_dataset.tf_utils import *
import tensorflow as tf

x = tf.random.normal([8192])

is_tensor(x)
>>> True

is_scalar_tensor(x)
>>> False

valid_lengths = tf.size(x)
tf_assert_length2(x, valid_lengths)
>>> <tensorflow.python.framework.ops.NullContextmanager at 0x2324252a430> # Pass

x = as_complex_tensor(x)
tf_assert_is_one_signal(x)
>>> <tensorflow.python.framework.ops.NullContextmanager at 0x2324252a430> # Pass

ds = tfd.signal_dataset(tfd.construct_metadat("./data"))
for x in ds: break
shapes(x)
>>> {'index': (),
 'filepath': (),
 'class': (),
 'num_classes': (),
 'target': (),
 'signal': (986507,)}

print(x)
>>> {'class': <tf.Tensor: shape=(), dtype=string, numpy=b'tug'>,
 'filepath': <tf.Tensor: shape=(), dtype=string,
    numpy=b'C:\\internal\\tf_dataset\\data\\1561971697.731_1561971718.283_24346_tug.wav'>,
 'index': <tf.Tensor: shape=(), dtype=int64, numpy=3>,
 'num_classes': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
 'signal': <tf.Tensor: shape=(986507,), dtype=int32, 
    numpy=array([ 1179011410, 3946020, ..., -2113897881, 1764622450])>,
 'target': <tf.Tensor: shape=(), dtype=int64, numpy=2>}
```


### pytest
Tests are stored in `/tf_dataset/test.py`, and can be run with a simple shell command from the top-level directory of this repo:
```{python}
pytest /tf_dataset/test.py
```

### maintainers
Go to the maintainers: [@bares, @stgeorge] for bug reports, and usage questions that aren't answered here.

Pull requests welcome!

#� �t�f�_�d�a�t�a�s�e�t�
�
�
