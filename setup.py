#!/usr/bin/python3
import os
from warnings import warn
from distutils.core import setup

REQ_FILE = 'requirements.txt'

if not os.path.exists(REQ_FILE):
      warn("No requirements file found.  Using defaults deps")
      deps = [
           'numpy',
           'scipy',
           'pandas',
           'pytest',
           'soundfile',
           'db-sqlite3',
           'tensorflow==2.5.0']
      warn(', '.join(deps))
else:
      with open(REQ_FILE, 'r') as f:
            deps = f.read().splitlines()


setup(name='tf_dataset',
      version='1.2',
      description='Create a tensorflow dataset (tf.data.Dataset) object from metadata for use with ML pipelines',
      author='Jason St George',
      author_email='stgeorgejas@gmail.com',
      packages=['tf_dataset'],
     install_requires=deps)
