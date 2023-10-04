from utils.tfrecords import csv2tf_mtl
import polars as pl
import glob

datafolder="../data"
train_filelist=glob.glob(datafolder+"/train/*.csv")
val_filelist=glob.glob(datafolder+"/val/*.csv")
test_filelist=glob.glob(datafolder+"/test/*.csv")

pair=[]
for idx,file in enumerate(train_filelist):
    pair.append((file,datafolder+f"/train_tf/{idx:02}.tf"))
for idx,file in enumerate(val_filelist):
    pair.append((file,datafolder+f"/val_tf/{idx:02}.tf"))
for idx,file in enumerate(test_filelist):
    pair.append((file,datafolder+f"/test_tf/{idx:02}.tf"))

expr=[
    pl.col("appid").cast(pl.Utf8),
    pl.col("position").cast(pl.Utf8)
]

csv2tf_mtl(pair,expr)

