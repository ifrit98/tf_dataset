import time
import scipy
import numpy as np
import pandas as pd
import sqlite3 as sq3
from sys import platform
from .tf_utils import is_tensor

is_function = lambda x: x.__class__.__name__ == 'function'
timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

def which_os():
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macOS"
    elif platform == "win32":
        return "windows"
    else:
        raise ValueError("Mystery os...")

def on_windows():
    return which_os() == "windows"

def on_linux():
    return which_os() == "linux"

def on_mac():
    return which_os() == "macOS"

def switch(on, pairs, default=None):
    """ Create dict switch-case from key-word pairs, mimicks R's `switch()`

        Params:
            on: key to index OR predicate returning boolean value to index into dict
            pairs: dict k,v pairs containing predicate enumeration results
        
        Returns: 
            indexed item by `on` in `pairs` dict
        Usage:
        # Predicate
            pairs = {
                True: lambda x: x**2,
                False: lambda x: x // 2
            }
            switch(
                1 == 2, # predicate
                pairs,  # dict 
                default=lambda x: x # identity
            )

        # Index on value
            key = 2
            switch(
                key, 
                values={1:"YAML", 2:"JSON", 3:"CSV"},
                default=0
            )
    """
    if type(pairs) is not dict:
        raise ValueError("`pairs` must be a list of tuple pairs or a dict")
    return pairs.get(on, default)

def isTrueOrFalse(x):
    if x not in [True, False]:
        return False
    return True

def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return cur.fetchall()

def unwrap_df(df):
    if len(df.values.shape) >= 2:
        return df.values.flatten()
    return df.values

# this is 1 line in R... smh
def lengths(x):
    def maybe_len(e):
        if type(e) == list:
            return len(e)
        else:
            return 1
    if type(x) is not list: return [1]
    if len(x) == 1: return [1]
    return(list(map(maybe_len, x)))

def db_to_csv(db_path='database.db'):
    db = sq3.connect(db_path)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(table_name + '.csv', index_label='index')
    cursor.close()
    db.close()

def db_list_fields(conn, table_nm):
    return pd.read_sql_query(
        "SELECT * from %s" % table_nm, conn).drop(['index'], axis = 1)

def write_table(conn, df, tbl_name):
    df.to_sql(tbl_name, conn, if_exists="replace")

def get_table_colnames(tbl_name, path=None, conn=None, drop_index=False):
    tbl = read_table(tbl_name, path=path, conn=conn, drop_index=drop_index)
    return tbl.columns.values

def read_sql(query, path=None, conn=None, drop_index=False):
    if conn is None:
        if path is None:
            raise ValueError("Must supply either `conn` or `path` to a sql db")
        conn = sq3.connect(path)
    res = pd.read_sql(query, conn) # pd.read_sql_query(query, conn)
    if 'index' in res.columns.values and drop_index:
        return res.drop(columns=['index'])
    return res

def read_table(tbl_name, path=None, conn=None, drop_index=False):
    if conn is None:
        if path is None:
            raise ValueError("Must supply either `conn` or `path` to a sql db")
        conn = sq3.connect(path)
    df = pd.read_sql_query("SELECT * FROM `{}` ORDER BY `_ROWID_`".format(tbl_name), conn)
    if 'index' in df.columns and drop_index:
        df = df.drop(columns='index')
    if conn is not None:
        conn.close()
    return df

def get_conn(db_path):
    return sq3.connect(db_path)

def read_db_tables(db_path):
    conn = get_conn(db_path)
    tbls = list_tables(conn)
    return dict(
        map(lambda x: (str(x[0]), read_table(x[0], path=db_path)), tbls))

def df_to_sqltype(df):    
    d = {
        None: 'NULL',
        np.dtype('int32'): 'INTEGER',
        np.dtype('int64'): 'INTEGER',
        np.dtype('float32'): 'REAL',
        np.dtype('float64'): 'REAL',
        np.dtype('str'): 'TEXT',
        np.dtype('O'): 'TEXT', # can we safely assume this?!
        np.dtype('bytes'): 'BLOB'
    }
    def check_types(s):
        print("in check_types, dtype:", s)
        if isinstance(s.dtype, object):
            try: 
                s = s.astype('str')
            except:
                raise TypeError("Ambiguous data type `Object` not coercible to `str`.")
        return s
    return df.apply(lambda x: d[check_types(x).dtype])

def df_to_tftype(df):
    d = {
        np.dtype('int32'): 'int32',
        np.dtype('int64'): 'int64',
        np.dtype('float32'): 'float32',
        np.dtype('float64'): 'float64',
        np.dtype('complex64'): 'complex64',
        np.dtype('complex128'): 'complex128',
        np.dtype('str'): 'string',
        np.dtype('O'): 'string', # can we safely assume this?!
        np.dtype('bytes'): 'string',
        np.dtype('bool'): 'bool'
    }
    return df.apply(lambda x: [d[x.dtype]])

def stats(x, axis=None, epsilon=1e-7):
    if is_tensor(x):
        x = x.numpy()
    else:
        x = np.asarray(x)
    if np.min(x) < 0:
        _x = x + abs(np.min(x) - epsilon)
    gmn = scipy.stats.gmean(_x, axis=axis)
    hmn = scipy.stats.hmean(_x, axis=axis)
    mode = scipy.stats.mode(x, axis=axis).mode[0]
    mnt2, mnt3, mnt4 = scipy.stats.moment(x, [2,3,4], axis=axis)
    lq, med, uq = scipy.stats.mstats.hdquantiles(x, axis=axis)
    lq, med, uq = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    var = scipy.stats.variation(x, axis=axis) # coefficient of variation
    sem = scipy.stats.sem(x, axis=axis) # std error of the means
    res = scipy.stats.describe(x, axis=axis)
    nms = ['nobs          ', 
           'minmax        ', 
           'mean          ', 
           'variance      ', 
           'skewness      ', 
           'kurtosis      ']
    description = dict(zip(nms, list(res)))
    description.update({
        'coeff_of_var  ': var,
        'std_err_means ': sem,
        'lower_quartile': lq,
        'median        ': med,
        'upper_quartile': uq,
        '2nd_moment    ': mnt2,
        '3rd_moment    ': mnt3,
        '4th_moment    ': mnt4,
        'mode          ': mode,
        'geometric_mean': gmn,
        'harmoinc_mean ': hmn
    })
    return description
