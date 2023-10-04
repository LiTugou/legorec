import tensorflow as tf
import glob

if tf.__version__.split(".",1)[0] == "1":
    pass
else:
    tf.truncated_normal_initializer=tf.compat.v1.truncated_normal_initializer
    tf.feature_column.shared_embedding_columns=tf.feature_column.shared_embeddings
    tf.feature_column.input_layer=tf.compat.v1.feature_column.input_layer
    tf.losses.log_loss=tf.compat.v1.losses.log_loss
    tf.metrics.auc=tf.compat.v1.losses.log_loss
    tf.train.get_global_step=tf.compat.v1.train.get_global_step


def _batched_parse(serialized_examples,schema):
    features=tf.io.parse_example(
        serialized_examples,
        features=schema
    )
    ctr=features.pop("finalClickFlag")
    cvr=features.pop("pay_flag")
    # ctr = tf.cast(features["finalClickFlag"],tf.float32)
    # cvr = tf.cast(features["pay_flag"],tf.float32)
    return features, {'ctr': ctr, 'cvr': cvr}


def loadtf(filenames,schema,batch_size=64,num_epochs=1):
    #filenames=glob.glob(pattern)
    seqlen=10
    for key in schema.keys():
        if "list" in key:
            schema[key]=tf.io.FixedLenFeature([seqlen], tf.string,default_value=["-1"]*seqlen)
    ds=tf.data.TFRecordDataset(
        filenames,
        compression_type=None,
        buffer_size=None,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds=ds.repeat(num_epochs)
    ds=ds.shuffle(buffer_size= 50 * batch_size)
    ds=ds.batch(batch_size)
    ds=ds.map(lambda x:_batched_parse(x,schema),num_parallel_calls=tf.data.AUTOTUNE)
    ds=ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


class CSVDataSet():
    def __init__(self,filelist,record_defaults,column_names,drop_columns=[],n_readers=4,num_parallel_calls=4,buffer_size=10240):
        self.filename_dataset = tf.data.Dataset.list_files(filelist)
        self.n_readers=n_readers
        self.record_defaults=record_defaults
        self.column_names=column_names
        self.drop_columns=drop_columns
        self.buffer_size=buffer_size
        self.num_parallel_calls=num_parallel_calls

    def getds(self,batch_size=1024,num_epochs=1):
        dataset = self.filename_dataset.interleave(
            lambda filename: tf.data.TextLineDataset(filename).skip(1).shuffle(self.buffer_size),
            #并行数为5，默认一次从并行数中取出一条数据
            cycle_length = self.n_readers
        )
        dataset = dataset.map(self.parse_csv, num_parallel_calls=self.num_parallel_calls)
        return dataset

    def parse_csv(self,value):
        columns = tf.io.decode_csv(value, record_defaults=self.record_defaults)
        features = dict(zip(self.column_names, columns))
        for col in self.drop_columns :
            features.pop(col)
        # for col in self.sequental_columns:
        #     features[col]=tf.strings.to_number(tf.strings.split(features[col], sep='|').values, out_type=tf.dtypes.int64)
        ctr = features.pop('finalClickFlag')
        cvr = features.pop('pay_flag')
        ctr = tf.cast(tf.reshape(ctr, [1]),tf.float32)
        cvr = tf.cast(tf.reshape(cvr, [1]),tf.float32)
        return features, {'ctr': ctr, 'cvr': cvr}


def input_fn(file_list,column_names,record_defaults,drop_columns,num_epochs=1,batch_size=1024):
    dsiter=CSVDataSet(file_list,record_defaults,column_names,drop_columns)
    dataset=dsiter.getds()
    dataset = dataset.repeat(num_epochs)
    dataset=dataset.shuffle(buffer_size= 50 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset=dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    # schema=modelobj.feat_schema
    # loadtf("data/train_tf/*.tf",schema,batch_size=64)
    pass