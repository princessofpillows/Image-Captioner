
import argparse
from cnn_models.generic_cnn import generic_cnn
from cnn_models.alexnet import alexnet
from cnn_models.zfnet import zfnet
from cnn_models.vggnet import vggnet

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


train_arg.add_argument("--data_dir", type=str,
                       default="/home/jpatts/Documents/anomaly-detector/data",
                       help="Directory of data")

train_arg.add_argument("--package_data", type=bool,
                       default=True,
                       help="Whether to package data or not")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--max_iter", type=int,
                       default=5000,
                       help="Number of iterations to train")

train_arg.add_argument("--log_dir", type=str,
                       default="./cnn_logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./cnn_save",
                       help="Directory to save the best model")

train_arg.add_argument("--val_freq", type=int,
                       default=500,
                       help="Validation interval")

train_arg.add_argument("--report_freq", type=int,
                       default=50,
                       help="Summary interval")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model",
                       default=alexnet,
                       choices=[generic_cnn, alexnet, zfnet, vggnet],
                       help="CNN model to use")

model_arg.add_argument("--reg_lambda", type=float,
                       default=1e-4,
                       help="Regularization strength")

model_arg.add_argument("--num_conv_base", type=int,
                       default=96,
                       help="Number of neurons in the first conv layer")

model_arg.add_argument("--num_unit", type=int,
                       default=4096,
                       help="Number of neurons in the hidden layer")

model_arg.add_argument("--num_hidden", type=int,
                       default=2,
                       help="Number of hidden layers")

model_arg.add_argument("--num_class", type=int,
                       default=2,
                       help="Number of classes in the dataset")

model_arg.add_argument("--activ_type", type=str,
                       default="relu",
                       choices=["relu", "tanh"],
                       help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
