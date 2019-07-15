
# Created by https://github.com/m3dev/redshells/blob/master/redshells/model/matrix_factorization_model.py

import numpy as np
import sklearn
from collections import defaultdict
from copy import copy, deepcopy
import random
from logging import getLogger
from typing import List, Any, Dict


logger = getLogger(__name__)


import tensorflow as tf


class MatrixFactorizationGrpah(object):
    def __init__(self,
                n_users: int,
                n_items: int,
                n_latent_factors: int,
                scope_name: str):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # placeholder
            self.input_user_idx = tf.placeholder(tf.int32, shape=[None], name="input_user_idx")
            self.input_item_idx = tf.placeholder(tf.int32, shape=[None], name="input_item_idx")
            self.input_click = tf.placeholder(tf.float32, shape=None, name="input_click")
            # TODO learning_rateを動的に変える
            self.input_learning_rate = tf.placeholder(tf.float32, name="input_learning_rate")
            
            # TODO initializerをいい感じにする
            initializer = tf.random_uniform_initializer(0,0.1)
            
            self.user_embedding = tf.keras.layers.Embedding(
                input_dim=n_users,
                output_dim=n_latent_factors,
                embeddings_initializer=initializer,
                name="user_embedding")
            self.user_factors = self.user_embedding(self.input_user_idx)
            self.item_embedding = tf.keras.layers.Embedding(
                input_dim=n_items,
                output_dim=n_latent_factors,
                embeddings_initializer=initializer,
                name="item_embedding")
            self.item_factors = self.item_embedding(self.input_item_idx)
            
            self.logit_click = tf.reduce_sum(tf.multiply(self.user_factors, self.item_factors), axis=1)
            self.predictions = 1 / (1 + tf.exp(-self.logit_click))
            
        with tf.name_scope("loss"):
            # sigmoid cross entropy loss
            # TODO l2 regularizer
            self.loss = tf.reduce_mean(
                -(self.input_click * tf.log(self.predictions) + \
                  (1 - self.input_click) * tf.log(1 - self.predictions)),
                name="loss")
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.input_learning_rate,
                name="optimizer")
            self.train_step = self.optimizer.apply_gradients(
                self.optimizer.compute_gradients(self.loss, var_list=var_list),
                name="train_step")
        self.train_batch_loss = tf.summary.scalar("train_batch_loss", self.loss)

        
class MatrixFactorization(object):
    """Matrix Factorization with only positive and implicit feedback"""
    def __init__(self,
                 n_latent_factors: int,
                 learning_rate: float,
#                  reg_user: float,
#                  reg_item: float,
                 batch_size: int,
                 epoch_size: int,
                 test_size: float,
                 save_directory_path: str,
                 scope_name: str,
                 try_count: int = 3,
                 n_users = None,
                 n_items = None,
                 standard_deviation=1,
                 user2index=None,
                 item2index=None) -> None:
        self.n_latent_factors = n_latent_factors
        self.learning_rate = learning_rate
#         self.reg_user = reg_user
#         self.reg_item = reg_item
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.test_size = test_size
        self.scope_name = scope_name
        self.try_count = try_count
        self.save_directory_path = save_directory_path
        self.tensorboard_path = save_directory_path + "/tensorboard"
        self.checkpoint_path = save_directory_path + "/checkpoint"
        self.n_users = n_users
        self.n_items = n_items
        self.standard_deviation = standard_deviation
        self.uesr2index = user2index
        self.item2index = item2index
        self.session = tf.Session()
        self.graph = None
        
        self.writer = tf.summary.FileWriter(self.tensorboard_path)
    
    def fit(self, user_ids: List[Any], item_ids: List[Any]) -> None:
        logger.info("data size={}".format(len(user_ids)))
        # TODO make it possible to reload model and train easily
        if self.graph is None:
            self.n_users = len(set(user_ids))
            self.n_items = len(set(item_ids))
            self.user2index = dict(zip(np.unique(user_ids), range(self.n_users)))
            self.item2index = dict(zip(np.unique(item_ids), range(self.n_items)))
            self.graph = self._make_graph()
            
        self.writer.add_graph(tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=0)
        
        user_indices_pos = self._convert(user_ids, self.user2index)
        item_indices_pos = self._convert(item_ids, self.item2index)
        
        user_train_pos, user_test_pos, item_train_pos, item_test_pos = sklearn.model_selection.train_test_split(
            user_indices_pos, item_indices_pos, test_size=self.test_size)
        self.user_train_pos = list(user_train_pos)
        self.user_test_pos = list(user_test_pos)
        self.item_train_pos = list(item_train_pos)
        self.item_test_pos = list(item_test_pos)
        
        with self.session.as_default():
            self.train_dataset = tf.data.Dataset.from_generator(self._train_generator_with_negative_sampling_approx,
                                                               output_types=(tf.int32, tf.int32, tf.float32),
                                                               output_shapes=(tf.TensorShape([]),
                                                                              tf.TensorShape([]),
                                                                              tf.TensorShape([])))
            self.train_dataset = self.train_dataset.batch(self.batch_size)
            self.test_dataset = tf.data.Dataset.from_generator(self._test_generator_with_negative_sampling_approx,
                                                              output_types=(tf.int32, tf.int32, tf.float32),
                                                              output_shapes=(tf.TensorShape([]),
                                                                             tf.TensorShape([]),
                                                                             tf.TensorShape([])))
            self.test_dataset = self.test_dataset.batch(self.batch_size)
            
            train_iterator = self.train_dataset.make_initializable_iterator()
            train_next_batch = train_iterator.get_next()
            test_iterator = self.test_dataset.make_initializable_iterator()
            test_next_batch = test_iterator.get_next()
            
            logger.info("start training ...")
            self.session.run(tf.global_variables_initializer())
            history = self._train(train_iterator=train_iterator,
                                  train_next_batch=train_next_batch,
                                  epoch_size=self.epoch_size,
                                  valid_iterator=test_iterator,
                                  valid_next_batch=test_next_batch,
                                  try_count=self.try_count,
                                  tensorboard_path=self.tensorboard_path,
                                  checkpoint_path=self.checkpoint_path)
    
    def predict(self, user_ids: List[Any], item_ids: List[Any], default=np.nan) -> np.ndarray:
        pass
#         """If input data is invalid, return `default`"""
#         if self.model is None:
#             RuntimeError("Please call fit first.")
        
#         user_indices = self._convert(user_ids, self.user2index)
#         item_indices = self._convert(item_ids, self.item2index)
#         valid_inputs = np.where(
#             np.logical_and(user_indices != -1, item_indices != -1))[0]
        
#         valid_predictions = self.model.predict(valid_inputs, batch_size=self.batch_size)[:,1]
#         valid_predictions = 1 / (1 + np.exp(-valid_predictions)) # logistic function
#         predictions = np.array([default]*len(user_ids))
#         predictions[valid_inputs] = valid_predictions
#         return predictions
    
    def get_user_factors(self, user_ids: List[Any], default=None, normalize: bool = False) -> np.ndarray:
        pass
#         """Return latent factors for given users."""
        
#         if self.model is None:
#             RuntimeError("Please call fit first.")
        
#         user_indices = self._convert(user_ids, self.user2index)
#         valid_inputs = np.where(user_indices != -1)[0]
        
#         user_embeddings = self.model.get_layer(name="user_embeddings").weights
#         valid_user_factors = self.sess.run(user_embeddings)
        
#         if normalize:
#             valid_user_factors = sklearn.preprocessing.normalize(valid_user_factors, axis=1, norm="l2")
        
#         default = default or np.zeros(valid_user_factors.shape[1])
#         predictions = np.array([default] * len(user_ids))
#         predictions[valid_inputs, :] = valid_user_factors
#         return predictions
    
    def get_item_factors(self, item_ids: List[Any], default=None, normalize: bool = False) -> np.ndarray:
        pass
#         """Return latent factors for given items."""
        
#         if self.model is None:
#             RuntimeError("Please call fit first.")
        
#         item_indices = self._convert(item_ids, self.item2index)
#         valid_inputs = np.where(item_indices != -1)[0]
        
#         item_embeddings = self.model.get_layer(name="item_embeddings").weights
#         valid_item_factors = self.sess.run(item_embeddings)
        
#         if normalize:
#             valid_item_factors = sklearn.preprocessing.normalize(valid_item_factors, axis=1, norm="l2")
        
#         default = default or np.zeros(valid_item_factors.shape[1])
#         predictions = np.array([default] * len(item_ids))
#         predictions[valid_inputs, :] = valid_item_factors
#         return predictions
    
    def get_valid_user_ids(self, ids: List):
        return [i for i in ids if i in self.user2index]
    
    def get_valid_item_ids(self, ids: List):
        return [i for i in ids if i in self.item2index]
    
    def _convert(self, ids: List[Any], id2index: Dict[Any, int]) -> np.ndarray:
        return np.array([id2index.get(i, -1) for i in ids])
    
    def _train_generator_with_negative_sampling_approx(self):
        """Using random (user, item) pair as a negative sample."""
        user_train_neg = list(np.random.choice(np.arange(self.n_users), size=len(self.user_train_pos)))
        item_train_neg = list(np.random.choice(np.arange(self.n_items), size=len(self.item_train_pos)))
        user_train = self.user_train_pos + user_train_neg
        item_train = self.item_train_pos + item_train_neg
        clicks = [1] * len(self.user_train_pos) + [0] * len(self.user_train_pos)
        train_pairs = list(zip(user_train, item_train, clicks))
        random.shuffle(train_pairs)
        for (user, item, click) in train_pairs:
            yield user, item, click

    def _test_generator_with_negative_sampling_approx(self):
        """Using random (user, item) pair as a negative sample."""
        user_test_neg = list(np.random.choice(np.arange(self.n_users), size=len(self.user_test_pos)))
        item_test_neg = list(np.random.choice(np.arange(self.n_items), size=len(self.item_test_pos)))
        user_test = self.user_test_pos + user_test_neg
        item_test = self.item_test_pos + item_test_neg
        clicks = [1] * len(self.user_test_pos) + [0] * len(self.user_test_pos)
        test_pairs = list(zip(user_test, item_test, clicks))
        random.shuffle(test_pairs)
        for (user, item, click) in test_pairs:
            yield user, item, click
    
    def _train(self,
              train_iterator,
              train_next_batch,
              epoch_size,
              valid_iterator,
              valid_next_batch,
              try_count,
              # TODO make it possible to choose to use tensorboard and checkpoint
              tensorboard_path,
              checkpoint_path):
        
        patience = 0
        prev_valid_loss = 1e18
        train_batch_cnt = 0
        valid_batch_cnt = 0
        
        for i in range(epoch_size):
            self.session.run(train_iterator.initializer)
            train_batch_loss_list = []
            while True:
                try:
                    train_batch_cnt += 1
                    user_, item_, click_ = self.session.run(train_next_batch)
                    feed_dict = {
                        self.graph.input_user_idx: user_,
                        self.graph.input_item_idx: item_,
                        self.graph.input_click: click_,
                        self.graph.input_learning_rate: self.learning_rate}
                    _, train_batch_loss, train_batch_loss_summary = self.session.run(
                        [self.graph.train_step, self.graph.loss, self.graph.train_batch_loss],
                        feed_dict=feed_dict)
                    self.writer.add_summary(train_batch_loss_summary, train_batch_cnt)
                    train_batch_loss_list.append(train_batch_loss)
                except tf.errors.OutOfRangeError:
                    print("train_dataset out of range")
                    train_epoch_loss = np.mean(train_batch_loss_list)
                    print(train_epoch_loss)
                    self.writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="train_epoch_loss",
                                                          simple_value=train_epoch_loss)]), i)
                    break
            
            self.session.run(valid_iterator.initializer)
            valid_batch_loss_list = []
            while True:
                try:
                    valid_batch_cnt += 1
                    user_, item_, click_ = self.session.run(valid_next_batch)
                    feed_dict = {
                        self.graph.input_user_idx: user_,
                        self.graph.input_item_idx: item_,
                        self.graph.input_click: click_,
                        self.graph.input_learning_rate: self.learning_rate}
                    valid_batch_loss = self.session.run(
                        self.graph.loss, feed_dict=feed_dict)
                    valid_batch_loss_list.append(valid_batch_loss)
                except tf.errors.OutOfRangeError:
                    print("valid_dataset out of range")
                    valid_epoch_loss = np.mean(valid_batch_loss_list)
                    print(valid_epoch_loss)
                    self.writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="valid_epoch_loss",
                                                          simple_value=valid_epoch_loss)]), i)
                    break
            
            self.saver.save(sess=self.session,
                           save_path=self.checkpoint_path + "/model_{}.ckpt".format(i),
                           write_meta_graph=False)
    
    def _make_graph(self) -> MatrixFactorizationGrpah:
        return MatrixFactorizationGrpah(
            n_users = self.n_users,
            n_items = self.n_items,
            n_latent_factors= self.n_latent_factors,
            scope_name = self.scope_name)