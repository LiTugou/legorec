#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import os
import sys
import logging

import tensorflow as tf
import numpy as np
from tensorflow.core.util.event_pb2 import SessionLog

if tf.__version__.split(".",1)[0] == "1":
    pass
else:
    tf.train.SessionRunHook=tf.compat.v1.train.SessionRunHook
    tf.train.CheckpointSaverHook=tf.compat.v1.train.CheckpointSaverHook

class PrintHook(tf.train.SessionRunHook):
    def __init__(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=400)

    # def begin(self):
    #     self._global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        """返回SessionRunArgs和session run一起跑"""

        #loss = tf.get_collection("embedding_input_shape")
        loss = tf.get_collection("loss")
        loss1 = tf.get_collection("loss1")
        loss2 = tf.get_collection("loss2")
        # embedding_input_shape=tf.get_collection("embedding_input_shape")
        # final_embedding_shape=tf.get_collection("final_embedding_shape")
        
        # feature_name_val_shape = tf.get_collection("feature_name_val_shape")
        # feat_input = tf.get_collection("feat_input")
        # feature_name_val = tf.get_collection("feature_name_val")


        #return tf.train.SessionRunArgs([embedding_input_shape,final_embedding_shape])

    def after_create_session(self, session, coord):
        self._global_step = tf.train.get_or_create_global_step()
        global_step = session.run(self._global_step)
        #print("global_step_ori: ", global_step)
        #print("global_variables: ", tf.global_variables())
        # print("---------------graph_struct------------")
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._global_step)
        loss = tf.get_collection("loss")
        loss1 = tf.get_collection("loss1")
        loss2 = tf.get_collection("loss2")
        if(global_step%100==0):
            # embed_input = run_values.results
            # print("global_step: ", global_step, "\nembed_input0:", embed_input[0])
            print("global_step: ", global_step, "loss:",loss,"loss ctr:",loss1,"loss cvr",loss2)

            sys.stdout.flush()


class CheckpointSaverHook(tf.train.CheckpointSaverHook):
    def _save(self, session, step):
        """saves the latest checkpoint, returns should_stop."""
        logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step, write_meta_graph=False)
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
            step)

        should_stop = False
        for l in self._listeners:
            if l.after_save(session, step):
                logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop
