import os
from torch.utils.tensorboard import SummaryWriter
import tempfile
from PIL import Image
import pandas as pd
from torchviz import make_dot, make_dot_from_trace
from datetime import datetime
import torchvision.transforms as transforms

class TensorboardLogger:
    def __init__(self, config):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(config.save_path, current_time)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.log_model_params(config)
        self.transform = transforms.ToTensor()
        
    def log_model_params(self, config):
        pass #unsure this is possible with tensorboard
            
    """def log_model_graph(self, model_name, model_save_dir, x, model, show_attrs=False, show_saved=False):
        vis = make_dot(model(x), params=dict(model.named_parameters()), show_attrs=show_attrs, show_saved=show_saved)
        vis_path = os.path.join(model_save_dir, f'{model_name}_vis')
        vis.format = 'png'
        vis.render(vis_path)
        vis_im = Image.open(f'{vis_path}.png')
        self.logger.add_image(f'{model_name}_vis', vis_im)"""
        
    def log_model_graph(self, model_name, model_save_dir, x, model, show_attrs=False, show_saved=False):
        self.logger.add_graph(model, x)
        
    def log_image(self, image, name, step=None):
        self.logger.add_image(name, self.transform(image), global_step=step)
        
    def log_scalar(self, scalar, name, step):
        self.logger.add_scalar(name, scalar, global_step=step)
        
    def close(self):
        self.logger.close()