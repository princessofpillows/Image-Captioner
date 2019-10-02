
import argparse

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ----------------------------------------
# Arguments for preprocessing
pre_arg = add_argument_group("Preprocessing")

pre_arg.add_argument("--image_size", type=int,
                       default=(227, 227, 3),
                       help="Resize dimensions for input")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--log_dir", type=str,
                       default="./rnn_logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./rnn_save",
                       help="Directory to save the best model")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--buffer_size", type=int,
                       default=100,
                       help="Shuffles subset of dataset if buffer_size is less than dataset size, otherwise uniform shuffle on whole dataset")

train_arg.add_argument("--max_iter", type=int,
                       default=5000,
                       help="Number of iterations to train")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--reg_lambda", type=float,
                       default=1e-4,
                       help="Regularization strength")

model_arg.add_argument("--num_unit", type=int,
                       default=100,
                       help="Number of neurons in the hidden layers")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
