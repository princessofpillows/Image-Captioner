
import os, h5py, csv
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tqdm import trange
from rnn_config import get_config, print_usage

class RNN(object):
    """Network class """

    def __init__(self, config, y_shp, images_shp, start, end):

        self.config = config

        # Maximum sentence length
        self.max_sentence_length = y_shp[1]
        # Number of words
        self.num_words = y_shp[2]

        # Dimensions of image data
        self.img_shp = images_shp[1]

        # The <start> condition for each sentence
        self.start = tf.constant(start)

        # The <end> condition for each sentence
        self.end = tf.constant(end)

        self._build_data()
        self._build_trainable()
        self._build_train_model()
        self._build_test_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_data(self):
        """Build placeholders."""

        # Create placeholders for inputs
        self.x_in = tf.placeholder(tf.float32, (None, self.max_sentence_length, self.num_words, 1))
        self.y_in = tf.placeholder(tf.float32, (None, self.max_sentence_length, 1))
        self.imgs_in = tf.placeholder(tf.float32, (None, self.img_shp, 1))

        # Create dataset
        sentences = tf.data.Dataset.from_tensor_slices((self.x_in, self.y_in, self.imgs_in)).repeat().shuffle(self.config.buffer_size)

        # Get data
        self.sentences_iterator = sentences.make_initializable_iterator()
        self.x, self.y, self.img = self.sentences_iterator.get_next()

    def _build_trainable(self):

        with tf.variable_scope("Weights", reuse=tf.AUTO_REUSE):

            # Create initializer for sample vector
            lower_bound = -1 * tf.sqrt(1 / self.num_words)
            upper_bound = tf.sqrt(1 / self.num_words)
            sample_initializer = tf.random_uniform_initializer(lower_bound, upper_bound)
            # Create initializer for other weight vectors
            num_unit = self.config.num_unit
            lower_bound = -1 * tf.sqrt(1 / num_unit)
            upper_bound = tf.sqrt(1 / num_unit)
            initializer = tf.random_uniform_initializer(lower_bound, upper_bound)

            # Weight for sample vector
            self.Wx = tf.get_variable(
                "Wx", (num_unit, self.num_words), tf.float32, sample_initializer)
            # Weight for hidden vector
            self.Wh = tf.get_variable(
                "Wh", (num_unit, num_unit), tf.float32, initializer)
            # Weight for output vector
            self.Wy = tf.get_variable(
                "Wy", (self.num_words, num_unit), tf.float32, initializer)
            # Weight for image vector
            self.Wi = tf.get_variable(
                "Wi", (num_unit, self.img_shp), tf.float32, initializer)
    
    def _build_train_model(self):

        def cond(t, ht, pred, logits):
            return tf.logical_not(tf.reduce_all(tf.equal(self.x[t], self.end)))

        def timestep(t, ht, pred, logits):
            # Calculate hidden vector at timestep t
            # Note: self.Wx[xt] where xt are indexes, is the same as self.Wx*self.xt, where xt are one_hot vectors
            ht = tf.tanh(
                tf.matmul(self.Wx, self.x[t]) + tf.matmul(self.Wh, ht) + tf.matmul(self.Wi, self.img))

            # Compute probabilities for next word in sentence
            logit = tf.matmul(self.Wy, ht)
            logits = tf.concat([logits[:t], tf.expand_dims(logit, 0)], 0)
            
            # Note: softmax(logits) is the equivalent of tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
            yt = tf.nn.softmax(logit, axis=0)
            
            # Change from shape (N, 1) to shape (N)
            probs = tf.squeeze(yt)
            samples = tf.range(0, self.num_words, 1)
            # Get index of predicted next word in sentence
            idx = tf.py_func(
                np.random.choice, [samples, tf.constant(1), tf.constant(False), probs], tf.int32)
            
            # Get word into right shape (1, N, 1)
            word = tf.expand_dims(
                tf.cast(idx, tf.float32), 0)
            # Add word to sentence
            pred = tf.concat([pred[:t], word], 0)

            return [t + 1, ht, pred, logits]

        t = tf.constant(0)
        # Initialize hidden vector
        ht = tf.zeros([self.config.num_unit, 1])
        # Initialize prediction matrix
        pred = tf.zeros([1, 1])
        logits = tf.zeros([1, self.num_words, 1])

        # Loop through words in sentence, outputting predicted caption
        t, ht, self.sentence, self.logits = tf.while_loop(cond, timestep, [t, ht, pred, logits], [t.shape, ht.shape, tf.TensorShape([None, 1]), tf.TensorShape([None, self.num_words, 1])])

        # Get list of all weights in this scope. They are called "kernel" in tf.layers.dense.
        self.kernels_list = [
            _v for _v in tf.trainable_variables() if "Weights" in _v.name]

    def _build_test_model(self):

        def cond(t, ht, pred, yt):
            return tf.logical_not(tf.reduce_all(tf.equal(yt, self.end)))

        def timestep(t, ht, pred, yt):
            # Calculate hidden vector at timestep t, using predicted word vectors yt
            ht = tf.tanh(
                tf.matmul(self.Wx, yt) + tf.matmul(self.Wh, ht) + tf.matmul(self.Wi, self.img))

            # Compute probabilities for next word in sentence
            # Note: softmax(logits) is the equivalent of tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
            yt = tf.nn.softmax(
                tf.matmul(self.Wy, ht), axis=0)
            
            # Change from shape (N, 1) to shape (N)
            probs = tf.squeeze(yt)
            samples = tf.range(0, self.num_words, 1)
            # Get index of predicted next word in sentence
            idx = tf.py_func(
                np.random.choice, [samples, tf.constant(1), tf.constant(False), probs], tf.int32)

            # Get word into right shape (1, 1)
            word = tf.expand_dims(
                tf.cast(idx, tf.float32), 0)
            # Add word to sentence
            pred = tf.concat([pred[:t], word], 0)

            idx = tf.reshape(idx, [])
            # Create predicted word at idx in one_hot representation
            yt = tf.reshape(
                tf.one_hot(idx, self.num_words), [self.num_words, 1])

            return [t + 1, ht, pred, yt]

        t = tf.constant(0)
        # Initialize hidden vector
        ht = tf.zeros([self.config.num_unit, 1])
        # Initialize prediction matrix
        pred = tf.zeros([1, 1])
        # Initialize first input as <start> vector
        yt = self.start
        
        # Loop through words in sentence, outputting predicted caption
        t, ht, self.pred, yt = tf.while_loop(cond, timestep, [t, ht, pred, yt], [t.shape, ht.shape, tf.TensorShape([None, 1]), yt.shape])
    
    def _build_loss(self):
        """Build cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            # Create cross entropy loss
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.x[1:], logits=self.logits)
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
            optim = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            # compute_gradients() + apply_gradients() = minimize(loss)
            self.optim = optim.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):
            # Compute the accuracy of the model
            self.train_acc = tf.reduce_mean(
                tf.to_float(tf.equal(self.sentence, self.y[1:]))
            )

            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.train_acc)

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
    
    def train(self, x_tr, y_tr, imgs_tr):

        # ----------------------------------------
        # Run TensorFlow Session
        with tf.Session() as sess:

            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            print("Training...")
            # Run the operations necessary for training
            res = sess.run(
                self.sentences_iterator.initializer,
                feed_dict={
                    self.x_in: x_tr,
                    self.y_in: y_tr,
                    self.imgs_in: imgs_tr
                }
            )

            for _ in trange(self.config.max_iter):
                res = sess.run(
                    fetches={
                        "optim": self.optim,
                        "summary": self.summary_op,
                        "global_step": self.global_step
                    }
                )

                # Write to tensorboard
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"],
                )
                self.summary_tr.flush()
            
            # Save the best model
            self.saver_best.save(
                sess, self.save_file_best,
                write_meta_graph=False,
            )

    def test(self, x_ts, y_ts, imgs_ts, word_index):

        # ----------------------------------------
        # Run TensorFlow Session
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

            print("Testing...")
            # Initialize datasets
            res = sess.run(
                self.sentences_iterator.initializer,
                feed_dict={
                    self.x_in: x_ts,
                    self.y_in: y_ts,
                    self.imgs_in: imgs_ts
                }
            )

            labels = []
            iters = len(x_ts)
            for _ in trange(iters):
                res = sess.run(
                    fetches={
                        "pred": self.pred
                    }
                )
                labels.append(res["pred"])

            decode(labels, word_index)

    
def main(config):
    """
    x_in : (W, X, Y, Z)
        4D matrix consisting of:
            Sentences,
            Words in sentence,
            One-hot representation of all words,
            Binary value

    images : (?, 4096, 1)
        CNN image features.
    """

    # Get required data
    x_in, y_in, imgs_in, start, end, word_index = get_data()

    y_in = y_in.reshape((y_in.shape[0], y_in.shape[1], 1))

    # Init rnn class
    rnn = RNN(config, x_in.shape, imgs_in.shape, start, end)

    # Split into train/test
    split = int(len(x_in) * .7)
    x_tr = x_in[:split]
    x_ts = x_in[split + 1:]
    y_tr = y_in[:split]
    y_ts = y_in[split + 1:]
    imgs_tr = imgs_in[:split]
    imgs_ts = imgs_in[split + 1:]

    # Train
    rnn.train(x_tr, y_tr, imgs_tr)

    # Test
    rnn.test(x_ts, y_ts, imgs_ts, word_index)


def get_data():
    # Load packaged data
    print("Loading data...")
    f = h5py.File('data/fcLayerData.h5', 'r')
    images = f['features']

    # Dictionary of phrases the RNN should label
    file = open("data/sentences.txt", 'r')
    dictionary = file.read().split(',')
    file.close()
    # Initialize tokenizer, and ensure it doesn't filter crocodile < > characters
    t = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    # Assign words to numbers
    t.fit_on_texts(dictionary)
    # Get total number of words by converting dictionary to one_hot representation and getting shape
    num_words = t.texts_to_matrix(dictionary, mode='count').shape[1] - 1
    # Get indexes of each word
    labels = t.texts_to_sequences(dictionary)

    one_hot_labels = encode(labels, num_words)

    idx = t.word_index['<start>']
    start = np.zeros((num_words, 1), "float32")
    start[idx-1] = 1

    idx = t.word_index['<end>']
    end = np.zeros((num_words, 1), "float32")
    end[idx-1] = 1

    return np.asarray(one_hot_labels), np.asarray(labels), np.asarray(images), start, end, t.word_index


def encode(labels, num_words):
    one_hot_labels = []
    max_sentence_length = 0
    # Create word vectors in one_hot representation
    for row in labels:
        if len(row) > max_sentence_length:
            max_sentence_length = len(row)
        one_hot_label = []
        for col in row:
            # Shape = (N, 1)
            word = np.zeros((num_words, 1), "float32")
            word[col-1] = 1
            one_hot_label.append(word)
        one_hot_labels.append(one_hot_label)
    
    # Make all sentences the same length with padding
    for sentence in one_hot_labels:
        while len(sentence) != max_sentence_length:
            padding = np.zeros((num_words, 1), "float32")
            sentence.append(padding)
    
    return one_hot_labels


def decode(labels, word_index):

    captions = []
    # Invert dictionary
    inv_word_index = {v: k for k, v in word_index.items()}
    # Convert indexes to words
    for sentences in labels:
        caption = ""
        for word in sentences:
            caption += inv_word_index[int(word[0]) + 1] + " "
        captions.append(caption)

    print(captions)


if __name__ == "__main__":

    # Parse configuration
    config, unparsed = get_config()
    # If there are unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
