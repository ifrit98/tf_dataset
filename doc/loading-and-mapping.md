# tf_dataset - user-loads-maps

## loading and mapping over data
We realize all users may not only have 1D acoustic or RF signal data, but rather differently structured (i.e. 2D image) data. For this module to be as generic as possible, `tf_dataset` allows for users to define their own custom mapping and data loading functions to suit their project-specific needs.

`tf_dataset` contains functions to efficiently parallel map over `tf.data.Dataset` objects, lazily applying arbitrary ops to each element.  This code is extensible (in a functional sense), so that users may define their own map calls (see [user-defined-maps](#user-defined-maps)) and may even define their own callables (see [user-defined-loads](#user-defined-loads)) for loading data to suit their needs. 



### user-defined-maps
Users may define their own `map` calls to process data and leverage the parallelism of tensorflow.  These `map`s are lazily evaluated, meaning that they are only executed when you need it, i.e. at training/inference time for any given batch.  This way you may make arbitrary map calls (mostly) without fear of memory overflows.  See [tf.data.Dataset.map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map). For more information on building data pipelines in tensorflow see [here](https://www.tensorflow.org/guide/data).

`map` calls operate (in python) over dictionaries.  E.g. the argument `x` is a python dict, containing all data for that example or batch (if a batched dataset).

The simplest way to create a map call is by accesing the `map` method of `tf.data.Dataset` objects:
```{python}
import tensorflow as tf

ds = tf.data.Dataset.range(1, 6)
list(ds.as_numpy_iterator())
>>> [1, 2, 3, 4, 5]

# Map x**2 over entire dataset.
ds = ds.map(lambda x: x**2)
list(ds.as_numpy_iterator())
>>> [1, 4, 9, 16, 25]
```

Note: You may use a `lambda` function (above) if there is only one tensor for each example in the dataset.

However, if you have more than one feature tensor in each dataset element, you must use a (regular) named function as `x` in the example below will be a dictionary:
```{python}
ds = tf.data.Dataset.from_tensor_slices(
  {
    'a': ([1, 2], [3, 4]),
    'b': [5, 6]
  }
)

for x in ds.take(1):
  print(x)
>>> # A dict!
>>> {'a': (<tf.Tensor: shape=(), dtype=int32, numpy=1>,
>>>        <tf.Tensor: shape=(), dtype=int32, numpy=3>),
>>>  'b': <tf.Tensor: shape=(), dtype=int32, numpy=5>}

# Define the map call that operates on dicts
def dataset_apply_sqrt(x):
  x['a'] = tf.sqrt(tf.cast(x['a'], tf.float32))
  x['b'] = tf.sqrt(tf.cast(x['b'], tf.float32))
  return x
```

This now can be used to lazily map over the entire dataset:
```{python}
ds = ds.map(dataset_apply_sqrt)
print(list(ds.as_numpy_iterator()))
>>> [{'a': array([1.       , 1.7320508], dtype=float32), 'b': 2.236068},
>>>  {'a': array([1.4142135, 2.       ], dtype=float32), 'b': 2.4494898}]

```


Here is the basic template for most use cases:
```{python}
def dataset_apply_my_func(x):
  # retrieve elements required from dictionary
  elem = x['key']

  # apply transformations to relevant tensors
  elem = tf.cast(elem, 'float32')
  elem = my_func(elem)

  # put it back
  x['key'] = elem

  # add any relevant metadata by creating new dict entry
  new_md_key = 'prev_dtype'
  x[new_md_key] = elem.dtype

  # return the whole dictionary
  return x
```


### user-defined-loads
As shown in `process_db_entry()` from [signal-dataset-demo](#signal-dataset-demo), users may define their own initial map calls to lazily load data from the sqlite db upon creation of the tensorflow dataset object.  This callable can be passed to any of `signal_dataset()`, `training_dataset()` or `dataset_from_dir()`.  This is useful if your data does not conform to the expected default of 1D signals with dtype [int32, float32, complex64], and file extension of `.wav`.

If the audio data requires special processing (say, if your amplitude values are stored in 24-bits, instead of 32, and would need to determine what's inside the header of the `.wav` file to be able to parse integers correctly. This is because tensorflow's `tf.io.decode_raw` assumes integer values are 32-bit depth.).  You also may have differently encoded audio (e.g. `.mpg`, `.ogg`, etc.), and can simply pass the `ext` argument to any of the dataset constructor functions, provided that the encoding is 32-bit.

If in another data format, i.e. 2D image data, stored as `.png` files, you may define a custom loading function to suit your needs:
```{python}
def process_db_example(x):
  img_bytes = x['image']
  image = tf.io.decode_png(img_bytes)
  x['image'] = image
  return x
```

A more complicated example as `.pq` (parquet) files:

```{python}
import cv2
import im2
import numpy as np
import pandas as pd

HEIGHT = 137
WIDTH  = 236
RESIZE = 128

num_gph_classes = 168
num_vow_classes = 11
num_con_classes = 7

d = {
    'filepath': [
        'data/img1.pq',
        'data/img2.pq',
        ...,
        'data/img99.pq'],
    'grapheme':  ['অ', 'ঊ', ..., 'tএ '],
    'vowel':     [' া', ' ী',  ..., 'ে '],
    'consonant': ['র্', ' র', ... , ' ্']
    }
df = pd.DataFrame.from_dict(d)
df['filepath'] = df['filepath'].apply(lambda x: os.path.abspath(x))


def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]

    # remove low intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad

    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))

def process_db_example(x, compact_data=False):
  # assumes single image per pq file, usually contain more than 1.
  image_df = pd.read_parquet(x['filepath'])

  img = image_df.iloc[0].values # grab single image in df
  img = img.reshape([HEIGHT, WIDTH])
  img = crop_resize(img, size=RESIZE, size2=75, pad=0)
  x['image'] = tf.expand_dims(img, -1L) # add channel dim for conv2D layer

  # Set up 3 labels for each img: (x, (w,y,z)) tuple pairs.
  x['grapheme']  = tf.one_hot(x['grapheme'], num_gph_classes, dtype=tf.int32)
  x['vowel']     = tf.one_hot(x['vowel'], num_vow_classes, dtype=tf.int32)
  x['consonant'] = tf.one_hot(x['consonant'], num_con_classes, dtype=tf.int32)

  return x

# Create the dataset by passing custom `process_db_example`
ds = signal_dataset(df, process_db_example=process_db_example)

```
