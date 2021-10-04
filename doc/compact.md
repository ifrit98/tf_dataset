

### dataset-compact
When working with a dataset object with your ML pipeline, it is important to know what type of object you are `map`ping over, e.g. either a python `dict` or python `tuple`.  By default, each element of the `tf.data.Dataset` object as constructed with this module is represented as a python `dict`, containing string keys and tensor (or numpy ndarray) values.

If you are writing custom a training loop or function, then it is likely most useful to retain this dictionary structure and simply fetch elements by key.  If you are intending to use the convenience function `tf.keras.Model.fit()` via a keras `Model`, datasets with `dict` elements will not work. They must be unpacked into `(x,y)` tuple pairs before passing to `fit()`.

`dataset_compact()` is a lazy `map` call that will unpack your dataset elements into appropriate `(x,y)` pairs based on the tensor keys desired.  You may pass multiple input and output tensor keys if your model is multi-input/output.


Create the dataset anyway you like:

```{python}
from tf_dataset import dataset_from_dir, dataset_compact

ds = dataset_from_dir("./data", batch_size=2, win_len=1024)
```

Then we must pass the necessary keys to `dataset_compact()` to unpack the `dict`s:
```{python}
for x in ds.take(1):
  print(list(x.keys()))
>>> ['signal', 'win_start_idx', 'index', 'filepath', 'class', 'num_classes', 'target']
x_key = 'signal'
y_key = 'target'
ds = dataset_compact(ds, x_key, y_key)
for x in ds.take(1):
  print(x)
>>> (<tf.Tensor: shape=(2, 1024, 12), dtype=float32, numpy=
array([[[ 0.0000000e+00,  1.0000000e+00,  9.5592844e-01, ...,
          4.5569549e+00,  2.0765837e+01,  6.0085924e-08],
        ...,
        [-3.3870557e-01,  1.5348996e-01,  0.0000000e+00, ...,
          2.7739763e-02,  7.6949445e-04,  2.1732165e-04]]], dtype=float32)>, <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0], dtype=int64)>)
```

And now we have our dataset in the correct format for `fit()`, where each element contains a (potentially nested) tuple:
```{python}
# The dataset now contains (x,y) pairs
input_tensor  = x[0]
output_tensor = x[1]
print(input_tensor.shape)
print(output_tensor.shape)
>>> (2, 1024, 12)
>>> (2,)

# Define model
model = ...
...

# Execute training with `fit()`
h = model.fit(ds, epochs=10, ...)
```
