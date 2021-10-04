## signal-dataset-demo

This is mainly for understanding what is happening under the hood.  If you do not need to know these details, or do not care, then skip to [advanced-example](#advanced-example) in the README for more usage.

Create your metadata dataframe. Test signals are in `/data` of this repo so you can validate easily.
```{python}
# NOTE: Make sure you are executing from /tf_dataset top-level dir to find these relpaths!
import os
import pandas as pd
import tensorflow as tf
from tf_dataset import *
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
print(df.head())
>>>                                             filepath      class
>>> 0  C:\internal\sam\tf_dataset\data\1561965276.252...  passenger
>>> 1  C:\internal\sam\tf_dataset\data\1561970414.69_...  passenger
>>> 2  C:\internal\sam\tf_dataset\data\1561971147.625...      cargo
>>> 3  C:\internal\sam\tf_dataset\data\1561971697.731...        tug
```

`write_signal_dataset_precursor_database()` is where the sqlite database is created. Resulting `path` will link to the time and process_id stamped sqlite database, ending in `.db`
```{python}
path = write_signal_dataset_precursor_database(df, overwrite=True)
print(path)
>>> C:\internal\tf_dataset\signal_dataset_precursor_1444_10_19_20_15-35-28_.sqlite
```

Grab the SQL query we want to execute.  This query will grab all relevant information in the `metadata` sql table, which has been stored in the sqlite database at `path`.
```{python}
deterministic = False
query = "SELECT * FROM `metadata` ORDER BY " +\
    ("`_ROWID_`" if deterministic else "RANDOM()")
```

Create a precursor dataset (tf.data.Dataset object) containing a minimum of [filepaths, labels] entries.  No signal data is actually loaded at this point, only the filepaths, labels, and potentially other metadata are housed in the tf.data.Dataset.
```{python}
print("\nCreating signal dataset precursor...\n".upper())
print("`tf.data.Dataset` now contains all metadata from padas `DataFrame`")
ds = signal_dataset_precursor(path, query)
for x in ds.take(1):
    print(x)
>>> {'class': <tf.Tensor: shape=(), dtype=string, numpy=b'tug'>,
>>>  'filepath': <tf.Tensor: shape=(), dtype=string,
>>>     numpy=b'C:\\internal\\tf_dataset\\data\\1561971697.731_1561971718.283_24346_tug.wav'>,
>>>  'index': <tf.Tensor: shape=(), dtype=int64, numpy=3>,
>>>  'num_classes': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
>>>  'target': <tf.Tensor: shape=(), dtype=int64, numpy=2>}
```

The default behavior in `process_db_entry()` is to load signals as int32 1D tensors using `tf.io`.  You may, however, pass your own `process_db_entry()` callable that processes the data to your liking. (See [user-defined-maps](#user-defined-maps))
```{python}
# Example code for `process_db_entry()`.
# This function is mapped over every element in the dataset
keep_filepath = True
def process_db_entry(x):
    sig = tf.io.read_file(x['filepath'])
    sig = tf.io.decode_raw(sig, tf.int32)
    x['signal'] = sig
    if not keep_filepath:
        x.pop('filepath')
    return x
```

Now, to return a usable dataset with loaded signal data (lazily evaluated), we must map over the precursor dataset using `process_db_entry()` defined above.
```{python}
num_parallel_calls = 4
ds = ds.map(process_db_entry,
        num_parallel_calls=tf.convert_to_tensor(num_parallel_calls, tf.int64))
for x in ds.take(1):
    print(x)
>>> {'class': <tf.Tensor: shape=(), dtype=string, numpy=b'cargo'>,
>>>  'filepath': <tf.Tensor: shape=(), dtype=string,
>>>     numpy=b'C:\\internal\\tf_dataset\\data\\1561971147.625_1561971202.077_24346_cargo.wav'>,
>>>  'index': <tf.Tensor: shape=(), dtype=int64, numpy=2>,
>>>  'num_classes': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
>>>  'signal': <tf.Tensor: shape=(2613707,), dtype=int32,
>>>       numpy=array([1179011410, 1163280727, ..., 1526755493, 7365376])>,
>>>  'target': <tf.Tensor: shape=(), dtype=int64, numpy=1>}
```

NOTE: If your data are not homogeneously shaped, then it is strongly advised to pass the resultant dataset from `signal_dataset()` to `dataset_signal_slice_windows()` in order to have uniform shapes across batches.

```{python}
ds = dataset_signal_slice_windows(ds, win_len=32768)
```
