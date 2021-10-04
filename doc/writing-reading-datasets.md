
### record-dataset
`record_dataset()` saves off a `tf.data.Dataset` object in an arbitrary state by serializing and writing `tfrecord`s to disk.  `tfrecord`s are based on Google's [protobuff](https://developers.google.com/protocol-buffers/docs/overview), and allow for efficnent compression.

What is important here is that we can apply arbitrary transformations to our data via map calls and then save out to disk for later use with training pipelines.

This allows you to do computationally expensive operations (e.g. computing giant FFTs, image processing, etc.) "offline", meaning they can be done before the training process, and saved in the state you need to enter your models.

Offline preprocessing will make training more efficient and affords the flexibility to granularly and modularly search for hyperparameters of your input pipeline (e.g. FFT params, AR model params, spectrogram params, etc.) without wasting significant training time.

See diagram below for an overview of the process:
[block-diagram](assets/block-diagram.png)

```{python}
from tf_dataset import dataset_from_dir
from tf_dataset import record_dataset

num_elems = 10
elements_per_file = 4

output_dir="./records"
output_dir = os.path.abspath(output_dir)

ds = dataset_from_dir("./data", infinite=True)
```
Tensors must have consistent shapes across batches/elements of the dataset. Therefore you must usually call `dataset_signal_slice_windows()` prior to using `record_dataset()` if your data do not contain homogenous shapes.

```{python}
ds = dataset_signal_slice_windows(ds, win_len=32768)
for x in ds.take(1):
  print(x)
>>> {'class': <tf.Tensor: shape=(), dtype=string, numpy=b'passenger'>,
>>>  'filepath': <tf.Tensor: shape=(), dtype=string,
>>>     numpy=b'C:\\internal\\data\\1561965276.252_1561965582.106_24346_passenger.wav'>,
>>>  'index': <tf.Tensor: shape=(), dtype=int64, numpy=0>,
>>>  'num_classes': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
>>>  'signal': <tf.Tensor: shape=(32768,), dtype=int32,
>>>     numpy=array([856621101, 4206080,   570443867, ..., -1962903607, 1893793905, 7725056])>,
>>>  'target': <tf.Tensor: shape=(), dtype=int64, numpy=0>,
>>>  'win_start_idx': <tf.Tensor: shape=(), dtype=int32, numpy=526185>}

```

It is possible to control how many elements from the dataset to write out, as well as how large each `tfrecord` should be using `num_elems` and `elements_per_file` arguments of `record_dataset()`, respectively.  `num_elems` is not specified (defaults to `None`), `record_dataset()` will write out the entire contents of the input dataset.
```{python}
record_dataset(
  ds, num_elems=num_elems, elements_per_file=elements_per_file,
  output_dir="./records"
)
>>> Processing batch: 1 of 4 in file 1
>>> Processing batch: 2 of 4 in file 1
>>> Processing batch: 3 of 4 in file 1
>>> Processing batch: 4 of 4 in file 1
>>> Processing batch: 1 of 4 in file 2
>>> Processing batch: 2 of 4 in file 2
>>> Processing batch: 3 of 4 in file 2
>>> Processing batch: 4 of 4 in file 2
>>> Processing batch: 1 of 2 in file 3
>>> Processing batch: 2 of 2 in file 3
>>> Done!
```

### replay-dataset
The inverse function to `record_dataset()`, which reads in serialized data in `tfrecord` format and constructs a `tf.data.Dataset` object from the contents.  Uses an internally managed `json` file to store the metadata necessary to reconstruct the data that was written out in `record_dataset()`.
```{python}
from tf_dataset import dataset_from_dir
from tf_dataset import replay_dataset

output_dir = "./records"
ds = replay_dataset(output_dir)
for x in ds.take(1):
  print(x)
>>> {'class': <tf.Tensor: shape=(), dtype=string, numpy=b'tug'>,
>>>  'filepath': <tf.Tensor: shape=(), dtype=string,
>>>     numpy=b'C:\\internal\\data\\1561971697.731_1561971718.283_24346_tug.wav'>,
>>>  'index': <tf.Tensor: shape=(), dtype=int64, numpy=3>,
>>>  'num_classes': <tf.Tensor: shape=(), dtype=int64, numpy=4>,
>>>  'signal': <tf.Tensor: shape=(32768,), dtype=int32,
>>>     numpy=array([7763456,  218131263, ..., 1325430985, 5689088])>,
>>>  'target': <tf.Tensor: shape=(), dtype=int64, numpy=2>,
>>>  'win_start_idx': <tf.Tensor: shape=(), dtype=int32, numpy=138888>}
```
