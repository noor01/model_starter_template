from utils import *
import argparse
import os
import neptune
from dotenv import load_dotenv
import importlib
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from loggers import *

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/example_config.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    opts = parser.parse_args()
    # Load experiment setting
    config = get_config(opts.config)
    config.output_path = opts.output_path
    config.resume = opts.resume
    # Setup output path
    save_path = os.path.join(config.output_path, config.version)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config.save_path = save_path
    # Setup logger and output folders
    if config.logger == 'neptune':
        logger = NeptuneLogger(config)
    elif config.logger == 'tensorboard':
        logger = TensorboardLogger(config)
    elif config.logger == 'wandb':
        logger = WandbLogger(config)
    # Setup trainer and start training    
    model = config.model
    trainer = get_trainer(model)(config, logger=logger)
    trainer.train()
    
def get_trainer(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trainers_dir = os.path.join(current_dir, 'trainers')
    sys.path.insert(0, trainers_dir)
    module = importlib.import_module(f"{model}_trainer")
    return module.Trainer
    
if __name__ == '__main__':
    main()