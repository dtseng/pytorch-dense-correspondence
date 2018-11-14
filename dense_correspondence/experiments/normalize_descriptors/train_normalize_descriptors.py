import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.training.training import *
import sys
import logging

# utils.set_default_cuda_visible_devices()
utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--dataset', type=str, default="caterpillar_only_9.yaml")
parser.add_argument('--dim', type=int, default=3)

parser.add_argument('--iters', type=int, default=3500)
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'dataset', 'composite', args.dataset)
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
dataset = SpartanDataset(config=config)

logging_dir = "/home/davidtseng/pytorch-dense-correspondence/data_volume/pdc/trained_models/simulated"
num_iterations = args.iters
d = args.dim # the descriptor dimension
name = args.name
train_config["training"]["logging_dir_name"] = name
train_config["training"]["logging_dir"] = logging_dir
train_config["dense_correspondence_network"]["descriptor_dimension"] = d
train_config["training"]["num_iterations"] = num_iterations
