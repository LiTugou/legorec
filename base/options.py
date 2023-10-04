#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# import tensorflow.compat.v1 as tf
#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import json
import os
import sys

FLAGS = flags.FLAGS


flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
flags.DEFINE_float("min_learning_rate", 0.00001, "min learning rate")
flags.DEFINE_float("drop_rate", 0.0, "drop out rate")
flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
flags.DEFINE_string("is_shuffle", 'no_shuffle', "is shuffle")
flags.DEFINE_integer("train_batch_size", 2000, "Number of batch size")
flags.DEFINE_integer("val_batch_size", 1000, "Number of batch size")
flags.DEFINE_integer("max_train_step", None, "max train step")
flags.DEFINE_integer("save_summary_steps", 20000, "save summary step")
flags.DEFINE_integer("save_checkpoint_and_eval_step", 20000, "save checkpoint and eval step")
flags.DEFINE_string("checkpoint_dir", '', "model checkpoint dir")
flags.DEFINE_string("checkpoint_file", '', "model checkpoint file")
flags.DEFINE_string("output_dir", '', "model checkpoint dir")

flags.DEFINE_string("task_type", 'train', "task type {train, predict, savemodel}")
flags.DEFINE_string("job_name", "", "job name")
flags.DEFINE_integer("task_index", 0, "Worker or server index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_integer("inter_op_parallelism_threads", 12, "inter_op_parallelism_threads")
flags.DEFINE_integer("intra_op_parallelism_threads", 12, "intra_op_parallelism_threads")

flags.DEFINE_string("feature_type_path", "newconf/feature_type.json", "feature_type_path")
flags.DEFINE_string("feature_config_path", "newconf/feature_config.json", "feature_config_path")
flags.DEFINE_string("KV_map_path", "newconf/KV_map.json", "KV_map_path")
flags.DEFINE_string("field_emb_config_path", "newconf/field_emb.json", "field_emb_config_path")
flags.DEFINE_bool("distribute", False, "distribute training")
flags.DEFINE_string("f","","kernel")


mtarget_params = {
    'hidden_units_mmoe': [[256, 128], [256, 128]],
    'num_cross_layers': 4,
    "expert_num": 2,
    "expert_size": 100,
    "tower_size": 16,
    "gate_num": 2,
    "bottom_dense_mlp": [512, 128, 64],
    "bottom_sparse_mlp": [128, 64],
    "num_dense_features": 13,
    "top_mlp": [1024, 1024, 512, 256],
}


def get_lastest_ckpt(checkpoint_dir):
    import re
    max_steps = 0
    result = tf.gfile.ListDirectory(checkpoint_dir)
    for file in result:
        print(str(file))
        if str(file).find('ckpt') >= 0:
            steps = re.findall(r'ckpt-(\d+)', str(file))
            if (len(steps) < 1):
                continue
            step_ = int(steps[0])
            max_steps = step_ if (step_ > max_steps) else max_steps
                # print('find: ' + oss_object.key)
    return max_steps


def set_psconfig_environ():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    tf.logging.info("job name = %s" % FLAGS.job_name)
    tf.logging.info("task index = %d" % FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}

    task_index = FLAGS.task_index
    task_type = FLAGS.job_name

    tf_config = dict()
    worker_num = len(cluster["worker"])
    if task_type == "ps":
        tf_config["task"] = {"index": task_index, "type": task_type}
        FLAGS.job_name = "ps"
        FLAGS.task_index = task_index
    else:
        if task_index == 0:
            tf_config["task"] = {"index": 0, "type": "chief"}
        else:
            tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

    if worker_num == 1:
        cluster["chief"] = cluster["worker"]
        del cluster["worker"]
    else:
        cluster["chief"] = [cluster["worker"][0]]
        del cluster["worker"][0]

    tf_config["cluster"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    tf.logging.info("TF_CONFIG %s", json.loads(os.environ["TF_CONFIG"]))

    # if "INPUT_FILE_LIST" in os.environ:
    #     INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    #     if INPUT_PATH:
    #           tf.logging.info("input path: %s", INPUT_PATH)
    #           FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
    #           FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
    #     else:  # for ps
    #           tf.logging.info("load input path failed.")
    #           FLAGS.train_data = None
    #           FLAGS.eval_data = None


class Options(object):
    def __init__(self):
        self.lr = FLAGS.learning_rate
        self.min_lr = FLAGS.min_learning_rate
        self.optimizer = FLAGS.optimizer
        self.num_epochs = FLAGS.num_epochs
        self.train_batch_size = FLAGS.train_batch_size
        self.val_batch_size = FLAGS.val_batch_size
        self.is_shuffle = FLAGS.is_shuffle
        self.checkpoint_path = FLAGS.checkpoint_dir
        # os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_file = FLAGS.checkpoint_file
        self.output_dir = FLAGS.output_dir

        # os.makedirs(self.output_dir, exist_ok=True)
        self.save_checkpoint_and_eval_step = FLAGS.save_checkpoint_and_eval_step
        self.max_train_step = FLAGS.max_train_step
        self.save_summary_steps = FLAGS.save_summary_steps
        self.inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads
        self.intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads
        # if (len(self.train_table) > 0):
        #     current_steps = get_lastest_ckpt(self.output_dir)
        #     # if (self.checkpoint_path != self.output_dir):
        #     #     current_steps = 0
        #     now_train_steps = get_train_tables_count()
        #     self.max_train_step = current_steps + int(
        #         self.num_epochs * now_train_steps // FLAGS.train_batch_size // (len(FLAGS.worker_hosts.split(',')) - 1))
        #     print("max_steps: " + str(self.max_train_step))
        #     print(current_steps, now_train_steps)

        self.feature_type_path=FLAGS.feature_type_path
        self.feature_config_path=FLAGS.feature_config_path
        self.KV_map_path=FLAGS.KV_map_path
        self.field_emb_config_path=FLAGS.field_emb_config_path
        self.distribute=FLAGS.distribute
        if FLAGS.distribute:
            init_distribute()

    def init_ps(self):
        pass
    
    def init_ring(self):
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
            },
            'task': {'type': 'worker', 'index': 0}
        })

    def init_distribute(self):
        self.task_index = FLAGS.task_index
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        self.ps_num = len(ps_hosts)
        print("ps_num: {}".format(self.ps_num))
        self.is_chief = False

        if FLAGS.task_type == 'predict':
            self.worker_num = len(worker_hosts)
            self.task_index = FLAGS.task_index
            if 'TF_CONFIG' in os.environ:
                del os.environ['TF_CONFIG']
        else:
            print('old task_index: ' + str(FLAGS.task_index))
            if FLAGS.task_index > 1:
                self.task_index = FLAGS.task_index - 1
            print('new task_index: ' + str(self.task_index))

            self.worker_num = len(worker_hosts) - 1
            if len(worker_hosts):
                cluster = {"chief": [worker_hosts[0]], "ps": ps_hosts, "worker": worker_hosts[2:]}
                if FLAGS.job_name == "ps":
                    os.environ['TF_CONFIG'] = json.dumps(
                        {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
                elif FLAGS.job_name == "worker":
                    if FLAGS.task_index == 0:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': "chief", 'index': 0}})
                    elif FLAGS.task_index == 1:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': "evaluator", 'index': 0}})
                    else:
                        os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index - 2}})
            if 'TF_CONFIG' in os.environ:
                print(os.environ['TF_CONFIG'])
            os.environ['TASK_INDEX'] = str(FLAGS.task_index)
            os.environ['JOB_NAME'] = str(FLAGS.job_name)

            if self.task_index == 0 and FLAGS.job_name == "worker":
                print("This is chief")
                self.is_chief = True

            self.hook_sync_replicas = None
            self.sync_init_op = None
