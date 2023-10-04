import polars as pl
import tensorflow as tf

tf_tab={
    pl.List(pl.Binary):lambda value: tf.train.Feature(bytes_list=tf.train.BytesList(value=value)),
    pl.Binary:lambda value:tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])),
    pl.List(pl.Int64):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.Int32):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.Int16):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.Int8):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.UInt64):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.UInt32):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.UInt16):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.List(pl.UInt8):lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    pl.Int64:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.Int32:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.Int16:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.Int8:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.UInt64:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.UInt32:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.UInt16:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.UInt8:lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
    pl.Float64:lambda value:tf.train.Feature(float_list=tf.train.FloatList(value=[value])),
    pl.Float32:lambda value:tf.train.Feature(float_list=tf.train.FloatList(value=[value])),
    pl.List(pl.Float64):lambda value:tf.train.Feature(float_list=tf.train.FloatList(value=value)),
    pl.List(pl.Float32):lambda value:tf.train.Feature(float_list=tf.train.FloatList(value=value))
}


import os
import glob

def _pl2tf(df,path):
    df=df.with_columns(
        pl.col(pl.Utf8).cast(pl.Binary),
        pl.col(pl.List(pl.Utf8))
        .explode()
        .cast(pl.Binary)
        .reshape((df.shape[0],-1))
    )
    translist=[tf_tab[dtype] for dtype in df.dtypes]
    columns=df.columns
    with tf.io.TFRecordWriter(path) as file_writer:
        for row in df.iter_rows():
            feature={}
            for i in range(len(row)):
                feature[columns[i]]=translist[i](row[i])
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            record_bytes=tf_example.SerializeToString()
            file_writer.write(record_bytes)

def pl2tf(df,folder,chunk_size=100000):
    df=df.with_columns(
        pl.col(pl.Utf8).cast(pl.Binary),
        pl.col(pl.List(pl.Utf8))
        .explode()
        .cast(pl.Binary)
        .reshape((df.shape[0],-1))
    ).sample(fraction=1,shuffle=True)
    translist=[tf_tab[dtype] for dtype in df.dtypes]
    columns=df.columns

    start=0
    end=0
    idx=0
    while end<df.shape[0]:
        start=end
        end=min(end+chunk_size,df.shape[0])
        path=os.path.join(folder,f"records_{idx:02}.tfrecord")
        subdf=df[start:end,:]
        with tf.io.TFRecordWriter(path) as file_writer:
            for row in subdf.iter_rows():
                feature={}
                for i in range(len(row)):
                    feature[columns[i]]=translist[i](row[i])
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                record_bytes=tf_example.SerializeToString()
                file_writer.write(record_bytes)
        idx+=1


from multiprocessing import Pool, cpu_count
import time

def _csv2tf_mtl_wraper(src,path,expr=[]):
    df=pl.read_csv(src)
    if expr:
        df=df.with_columns(expr)
    df=df.with_columns(
        pl.col(pl.Utf8).cast(pl.Binary),
        pl.col(pl.List(pl.Utf8))
        .explode()
        .cast(pl.Binary)
        .reshape((df.shape[0],-1))
    )
    translist=[tf_tab[dtype] for dtype in df.dtypes]
    columns=df.columns
    with tf.io.TFRecordWriter(path) as file_writer:
        for row in df.iter_rows():
            feature={}
            for i in range(len(row)):
                feature[columns[i]]=translist[i](row[i])
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            record_bytes=tf_example.SerializeToString()
            file_writer.write(record_bytes)


def csv2tf_mtl(pair,expr=[]):
    # pair=[(file,dst),()]
    pool = Pool(processes=(cpu_count() - 1))
    for i in range(len(pair)):
        info=pair[i]
        pool.apply_async(_csv2tf_mtl_wraper, args=(info[0], info[1],expr))
    pool.close()
    pool.join()
