from utils.tfrecords import csv2tf_mtl
import polars as pl
import glob

train_filelist=glob.glob("data/train/*.csv")
val_filelist=glob.glob("data/val/*.csv")
test_filelist=glob.glob("data/test/*.csv")

pair=[]
for idx,file in enumerate(train_filelist):
    pair.append((file,f"data/train_tf/{idx:02}.tf"))
for idx,file in enumerate(val_filelist):
    pair.append((file,f"data/val_tf/{idx:02}.tf"))
for idx,file in enumerate(test_filelist):
    pair.append((file,f"data/test_tf/{idx:02}.tf"))

expr=[
    pl.col("appid").cast(pl.Utf8),
    pl.col("position").cast(pl.Utf8)
]

csv2tf_mtl(pair,expr)

