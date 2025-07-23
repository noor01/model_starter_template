import os
import wandb
from dotenv import load_dotenv
import tempfile
from torchviz import make_dot, make_dot_from_trace
from PIL import Image
import pandas as pd

class WandbLogger:
    def __init__(self, config):
        load_dotenv()
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        self.logger = wandb.init(
            project=config.project_name,
            config=self.log_model_params(config)
            )
        
        
    def log_model_params(self, config):
        config_attributes = [attr for attr in dir(config) if not attr.startswith('__')]
        config_dict = {}
        for attr in config_attributes:
                value = getattr(config, attr)
                config_dict[attr] = value
        return config_dict
            
    def log_model_graph(self, model_name, model_save_dir, x, model, show_attrs=False, show_saved=False):
        vis = make_dot(model(x), params=dict(model.named_parameters()), show_attrs=show_attrs, show_saved=show_saved)
        vis_path = os.path.join(model_save_dir, f'{model_name}_vis')
        vis.format = 'png'
        vis.render(vis_path)
        self.logger.log({f'{model_name}_vis': wandb.Image(f'{vis_path}.png')}
                        )        
    def log_image(self, image, name, step=None):
        if step:
            self.logger.log({name: [wandb.Image(image)]}, step=step)
        else:
            self.logger.log({name: [wandb.Image(image)]})
        
    def log_scalar(self, scalar, name, step=None):
        if step:
            self.logger.log({name: scalar}, step=step)
        else:
            self.logger.log({name: scalar})
            
    def close(self):
        #self.logger.close()
        return