
import os, h5py
import tensorflow as tf
import numpy as np
from cnn_config import get_config, print_usage
from utils.preprocessing import package_data
from tqdm import trange


class CNN(object):
    """Network class """

    def __init__(self, config, x_shp):

        self.config = config

        # Get shape for placeholder (ignore first index = number of inputs)
        self.x_shp = [None] + list(x_shp[1:])

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""

        # Create Placeholders for inputs
        self.x_in = tf.placeholder(tf.float32, shape=self.x_shp)
        self.y_in = tf.placeholder(tf.int64, shape=(None, ))

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
            # Create placeholders for saving mean, range to a TF variable for easy save/load
            self.n_mean_in = tf.placeholder(tf.float32, shape=())
            self.n_range_in = tf.placeholder(tf.float32, shape=())
            # Make the normalization as a TensorFlow variable. This is to make sure it is saved in the graph
            self.n_mean = tf.get_variable(
                "n_mean", shape=(), trainable=False)
            self.n_range = tf.get_variable(
                "n_range", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.n_assign_op = tf.group(
                tf.assign(self.n_mean, self.n_mean_in),
                tf.assign(self.n_range, self.n_range_in),
            )

    def _build_model(self):
        """Build MLP network."""

        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):

            # Normalize using the training-time statistics
            cur_in = (self.x_in - self.n_mean) / self.n_range

            # Use architecture specified in config using tf.layers
            self.logits, self.kernels_list, self.features = self.config.model(config, cur_in)

    def _build_loss(self):
        """Build cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            # Create cross entropy loss
            onehot = tf.one_hot(self.y_in, self.config.num_class)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=onehot, logits=self.logits)
            )

            # Create l2 regularizer loss and add
            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):

            # Compute the accuracy of the model. When comparing labels elemwise, use tf.equal instead of `==`. `==` will evaluate if your Ops are identical Ops.
            self.pred = tf.argmax(self.logits, axis=1)
            self.acc = tf.reduce_mean(
                tf.to_float(tf.equal(self.pred, self.y_in))
            )

            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.acc)

            # Save best validation accuracy
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable(
                "best_va_acc", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.acc_assign_op = tf.assign(
                self.best_va_acc, self.best_va_acc_in)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training data.

        y_tr : ndarray
            Training labels.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        # ----------------------------------------
        # Preprocess data

        # Use the data_mean for x_tr_mean, and 128 for the range, as we are dealing with image and CNNs now
        x_tr_mean = x_tr.mean()
        x_tr_range = 128.0

        # Report data statistic
        print("Training data before: mean {}, std {}, min {}, max {}".format(
            x_tr_mean, x_tr.std(), x_tr.min(), x_tr.max()
        ))

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # Assign normalization variables from statistics of the train data
            sess.run(self.n_assign_op, feed_dict={
                self.n_mean_in: x_tr_mean,
                self.n_range_in: x_tr_range,
            })

            # Check if previous train exists
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.log_dir)
            b_resume = tf.train.latest_checkpoint(
                self.config.log_dir) is not None
            if b_resume:
                # Restore network
                print("Restoring from {}...".format(
                    self.config.log_dir))
                self.saver_cur.restore(
                    sess,
                    latest_checkpoint
                )
                # Restore number of steps so far
                step = sess.run(self.global_step)
                # Restore best acc
                best_acc = sess.run(self.best_va_acc)
            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            batch_size = config.batch_size
            max_iter = config.max_iter
            # For each epoch
            for step in trange(step, max_iter):

                # Get a random training batch
                ind_cur = np.random.choice(
                    len(x_tr), batch_size, replace=False)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])

                # Write summary every kN iterations as well as the first iteration (write the summary after we do the optimization)
                b_write_summary = ((step + 1) % self.config.report_freq) == 0
                b_write_summary = b_write_summary or step == 0
                if b_write_summary:
                    fetches = {
                        "optim": self.optim,
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }
                else:
                    fetches = {
                        "optim": self.optim,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                    },
                )

                # Write Training Summary if we fetched it (don't write meta graph)
                if "summary" in res:
                    self.summary_tr.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_tr.flush()

                    # Also save current model to resume when we write the summary.
                    self.saver_cur.save(
                        sess, self.save_file_cur,
                        global_step=self.global_step,
                        write_meta_graph=False,
                    )

                # Validate every N iterations and at the first iteration
                b_validate = ((step + 1) % self.config.val_freq) == 0
                b_validate = b_validate or step == 0
                if b_validate:
                    res = sess.run(
                        fetches={
                            "acc": self.acc,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                        },
                        feed_dict={
                            self.x_in: x_va,
                            self.y_in: y_va
                        })
                    # Write Validation Summary
                    self.summary_va.add_summary(
                        res["summary"], global_step=res["global_step"],
                    )
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and best accuracy. Only return the best W and b
                    if res["acc"] > best_acc:
                        best_acc = res["acc"]
                        # Write best acc to TF variable
                        sess.run(
                            self.acc_assign_op,
                            feed_dict={
                                self.best_va_acc_in: best_acc
                            })
                        # Save the best model
                        self.saver_best.save(
                            sess, self.save_file_best,
                            write_meta_graph=False,
                        )

    def test(self, x_te, y_te):
        """Test routine"""

        h5f = h5py.File('data/fcLayerData.h5', 'w')

        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.save_dir)
            if tf.train.latest_checkpoint(self.config.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.config.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

            # Test on the test data
            bs = config.batch_size
            num_test_b = len(x_te) // bs
            acc_list = []
            features = []
            for idx_b in range(num_test_b):
                res = sess.run(
                    fetches={
                        "acc": self.acc,
                        "features": self.features,
                    },
                    feed_dict={
                        self.x_in: x_te[idx_b * bs: (idx_b + 1) * bs],
                        self.y_in: y_te[idx_b * bs: (idx_b + 1) * bs],
                    },
                )
                acc_list += [res["acc"]]
                features.append(res["features"])
            res_acc = np.mean(acc_list)

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(res_acc))
            features = np.asarray(features)
            features = features.reshape(
                features.shape[0] * features.shape[1], features.shape[2], 1)

            h5f.create_dataset('features', data=features, dtype='float32')
            h5f.close()


def main(config):
    """The main function."""
    config.package_data = False

    # Package data from directory into HD5 format
    if config.package_data:
        print("Packaging data into H5 format...")
        package_data(config.image_size, config.data_dir)
    else:
        print("Packaging data skipped.")

    # Load packaged data
    print("Loading data...")
    f = h5py.File('data/videoData.h5', 'r')

    x = f['videos']
    y = f['info']

    # Create array of random values, where length and range of s = length of datasets
    s = np.arange(np.asarray(x).shape[0])
    np.random.shuffle(s)
    # Shuffle data randomly
    x = np.asarray(x)[s]
    y = np.asarray(y)[s]

    def split(data):
        # 70% train, 20% val, 10% test split
        train_split = int(data.shape[0] * 0.7)
        val_split = int(data.shape[0] * 0.2) + train_split
        return data[:train_split], data[train_split:val_split], data[val_split:]

    x_tr, x_va, x_te = split(x)
    y_tr, y_va, y_te = split(y)

    # ----------------------------------------
    # Init cnn class
    mynet = CNN(config, x_tr.shape)

    # ----------------------------------------
    # Train
    mynet.train(x_tr, y_tr, x_va, y_va)

    # ----------------------------------------
    # Test
    mynet.test(x_te, y_te)


if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
    