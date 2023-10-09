import tensorflow as tf
from tensorflow.keras import layers,Input
from tensorflow import feature_column
import json

if tf.__version__.split(".",1)[0] == "1":
    pass
else:
    tf.truncated_normal_initializer=tf.compat.v1.truncated_normal_initializer
    tf.feature_column.shared_embedding_columns=tf.feature_column.shared_embeddings
    tf.feature_column.input_layer=tf.compat.v1.feature_column.input_layer
    tf.losses.log_loss=tf.compat.v1.losses.log_loss
    tf.metrics.auc=tf.compat.v1.losses.log_loss
    tf.train.get_global_step=tf.compat.v1.train.get_global_step

feat_type_map={
    "string":tf.string,
    "mapkey":tf.string,
    "listvalue":tf.float32,
    "mapvalue":tf.float32,
    'numtobucket':tf.int64,
    'bignumtobucket':tf.int64,
    'doubletobucket':tf.int64,
    'bucketid':tf.int64,
    'timetobucket':tf.int64,
    'stringlist':tf.string,
    'stringtolist':tf.string,
    'stringlist_ragged':tf.string,
    'bucketidlist':tf.int64
    'bucketidlist_ragged':tf.int64
}

default_map={
    "string":"default_value",
    "mapkey":"default_value",
    "listvalue":0,
    "mapvalue":0,
    'numtobucket':0,
    'bignumtobucket':0,
    'doubletobucket':0,
    'bucketid':0,
    'timetobucket':0,
    'stringlist':"",
    'stringtolist':"",
}

def shared_embedding(key_name, val_name, hash_bucket, feat_field, dim):
    cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name, hash_bucket, dtype=tf.string)
    if val_name is not None:
        cate_feature = tf.feature_column.weighted_categorical_column(cate_feature, val_name, dtype=tf.float32)

    cate_feature = tf.feature_column.shared_embedding_columns([cate_feature], dim,
                                               combiner='mean',
                                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               shared_embedding_collection_name=feat_field)
    return cate_feature[0]


def map_embedding(key_name, val_name, hash_bucket, dim):
    cate_feature = tf.feature_column.categorical_column_with_hash_bucket(key_name, hash_bucket, dtype=tf.string)
    w_cate_feature = tf.feature_column.weighted_categorical_column(cate_feature, val_name, dtype=tf.float32)
    emb_col = tf.feature_column.embedding_column(w_cate_feature, dimension=dim,
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    return emb_col

## 如图片embeding等
def nums_embeddding(key_name, dim):
    emb_col = tf.feature_column.numeric_column(key_name, (dim,))
    return emb_col

def numeric_column(key_name):
    return tf.feature_column.numeric_column(key_name, default_value=0.0)


def index_embedding(key_name, hash_bucket, dim):
    id_feature = tf.feature_column.categorical_column_with_identity(key_name, num_buckets=hash_bucket,
                                                                    default_value=0)
    emb_col = tf.feature_column.embedding_column(id_feature, dim,
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    return emb_col


def hash_embedding(key_name, hash_bucket, dim):
    cate_feature = feature_column.categorical_column_with_hash_bucket(key_name,
                                                                      hash_bucket,
                                                                      dtype=tf.string)
    emb_col = feature_column.embedding_column(
        cate_feature,
        dimension=dim,
        combiner='mean', initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    return emb_col


## 不接受ragged tensor,可以通过 .to_tensor(-1,shape=(...)) cut 和 pad
## 然后使用 tf.keras.experimental.SequenceFeatures([emb1,emb2])(features)
## tf.keras.experimental.SequenceFeatures 会将emb1,emb2在emb维度拼接
## 输出两个向量 embedding,seqlen
## 通过 tf.sequence_mask(seqlen) 就可以生成 mask了
## 对于 int -1, string “” 不会 embedding

def sequence_index_embedding(key_name,hash_bucket,dim):
    cate_feature = feature_column.feature_column.sequence_categorical_column_with_identity(key_name,
                                                                      hash_bucket,
                                                                      dtype=tf.string)
    emb_col = feature_column.embedding_column(
        cate_feature,
        dimension=dim,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    return emb_col
    
def sequence_hash_embedding(key_name,hash_bucket,dim):
    cate_feature = feature_column.feature_column.sequence_categorical_column_with_hash_bucket(key_name,
                                                                      hash_bucket,
                                                                      dtype=tf.string)
    emb_col = feature_column.embedding_column(
        cate_feature,
        dimension=dim,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    return emb_col

# def tag_embedding(self, key_name, hash_bucket, dim):
#     id_feature = layers.sparse_column_with_hash_bucket(
#         column_name=key_name,
#         hash_bucket_size=hash_bucket,
#         combiner='mean',
#         dtype=tf.string,
#     )
#     emb_col = layers.embedding_column(
#         id_feature,
#         dimension=dim,
#         combiner='mean', initializer=tf.truncated_normal_initializer(stddev=0.1)
#     )
#     return emb_col


class FeatureConf():
    def __init__(self,
                 feature_config_path,
                 feature_type_path,
                 KV_map_path,
                 field_emb_config_path
        ):
        self.feature_type_dict = json.load(tf.io.gfile.GFile(feature_type_path, 'r'))
        self.feature_config_dict = json.load(tf.io.gfile.GFile(feature_config_path, 'r'))
        self.KV_dict = json.load(tf.io.gfile.GFile(KV_map_path, 'r'))
        self.field_emb_dict=json.load(tf.io.gfile.GFile(field_emb_config_path, 'r'))
        self.feat_columns=None
        self.feat_input=None
        self.feat_type=None
        self.feat_default=None
        self.build_feature()
        self.build_input()
        self.build_recordstype()
        self.build_default()
        
    def build_feature(self):
        self.feat_columns={}
        for feat_field,feat_list in self.feature_type_dict.items():
            column_list=[]
            for feat_name in feat_list:
                feat_column = self.get_feat_column(feat_name)
                if feat_column is not None:
                    column_list.append(feat_column)
            self.feat_columns[feat_field+"_column_list"]=column_list

    def build_input(self):
        self.feat_input={}
        for feat_field,feat_list in self.feature_type_dict.items():
            column_list=[]
            for feat_name in feat_list:
                feat_input = self.get_feat_input(feat_name)
                self.feat_input[feat_name]=feat_input

    def build_recordstype(self):
        self.feat_schema={}
        for feat_field,feat_list in self.feature_type_dict.items():
            column_list=[]
            for feat_name in feat_list:
                self.feat_schema[feat_name]=self.get_feat_schema(feat_name)

    def build_default(self):
        self.feat_default={}
        for feat_field,feat_list in self.feature_type_dict.items():
            column_list=[]
            for feat_name in feat_list:
                self.feat_default[feat_name]=self.get_feat_default(feat_name)

    def get_feat_default(self,feat_name):
        config = self.feature_config_dict[feat_name]
        feat_type = config['type'].lower()
        feat_len = config.get("length",1)
        default=config.get("default_value",default_map[feat_type])
        if feat_len==1:
            return default
        else:
            return "|".join([default]*feat_len)

    def get_feat_schema(self,feat_name):
        config = self.feature_config_dict[feat_name]
        feat_type = config['type'].lower()
        feat_len = config.get("length",1)
        tf_type=feat_type_map[feat_type]
        default=config.get("default_value",default_map[feat_type])
        return tf.io.FixedLenFeature([feat_len], tf_type,default_value=[default]*feat_len)


    def get_feat_input(self,feat_name):
        config = self.feature_config_dict[feat_name]
        feat_type = config['type'].lower()
        feat_field = config['field']
        feat_len = config.get("length",1)
        return Input(
                    shape=(feat_len,),
                    name=feat_name,
                    dtype=feat_type_map[feat_type],
                )

    def get_feat_column(self, feat_name):
        ## 全都是使用 field embeding dim，没有使用 feature_config里的embedding_dim
        ## field 不为 other时存在共享embedding
        config = self.feature_config_dict[feat_name]
        feat_type = config['type'].lower()
        feat_field = config['field']
        
        emb_dim=self.field_emb_dict.get(feat_field,None)
        
        if feat_type == "listvalue":
            return nums_embeddding(feat_name, config['embedding_dim'])
        elif feat_type == "numeric":
            return numeric_column(feat_name)
        elif feat_field != "other" and feat_type != "mapvalue":
        ## shared feature, 肯定是string
            map_val_fea_name = None
            if feat_name in self.KV_dict:
                map_val_fea_name = self.KV_dict[feat_name]
            return shared_embedding(feat_name, map_val_fea_name, config['hash_bucket'], feat_field,emb_dim)

        elif feat_type == 'string':
            return hash_embedding(feat_name, config['hash_bucket'], emb_dim)
        elif feat_type == 'mapkey':
            ## mapvalue 作为权重
            map_val_fea_name = self.KV_dict[feat_name]
            return map_embedding(feat_name, map_val_fea_name, config['hash_bucket'],emb_dim)
        elif feat_type in ('numtobucket', 'bignumtobucket', 'doubletobucket', 'bucketid', 'timetobucket'):
            return index_embedding(feat_name, config['hash_bucket'], emb_dim)
        elif feat_type in ('stringlist', 'stringtolist', 'wordsegmentation'):
            # return self.tag_embedding(feat_name, config['hash_bucket'], embedding_dim)
            return hash_embedding(feat_name, config['hash_bucket'], emb_dim)


# tmp=FeatureConf(FLAGS.feature_config_path,FLAGS.feature_type_path,FLAGS.KV_map_path,FLAGS.field_emb_config_path)