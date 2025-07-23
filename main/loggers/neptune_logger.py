import os
import neptune
from dotenv import load_dotenv
import tempfile
from torchviz import make_dot, make_dot_from_trace
from PIL import Image
import pandas as pd

class NeptuneLogger:
    def __init__(self, config):
        load_dotenv()
        neptune_api_key = os.environ.get('NEPTUNE_API_TOKEN')
        self.logger = neptune.init_run(
            project=config.project_name,
            api_token=neptune_api_key,
            tags=['training'],
            source_files='*.py'
        )
        self.logger["parameters"] = self.log_model_params(config)
        
    def log_model_params(self, config):
        config_attributes = [attr for attr in dir(config) if not attr.startswith('__')]
        config_dict = {}
        for attr in config_attributes:
                value = getattr(config, attr)
                config_dict[attr] = value
        config_df = pd.DataFrame([config_dict])
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            config_df.to_csv(temp.name, index=False)
            self.logger["config"].upload(temp.name)
            temp.close()
            
    def log_model_graph(self, model_name, model_save_dir, x, model, show_attrs=False, show_saved=False):
        vis = make_dot(model(x), params=dict(model.named_parameters()), show_attrs=show_attrs, show_saved=show_saved)
        vis_path = os.path.join(model_save_dir, f'{model_name}_vis')
        vis.format = 'png'
        vis.render(vis_path)
        self.logger[f'{model_name}_vis'].append(Image.open(f'{vis_path}.png'))
        
    def log_image(self, image, name, step=None):
        if step:
            self.logger[name].log(image, step=step)
        else:
            self.logger[name].log(image)
        
    def log_scalar(self, scalar, name, step=None):
        if step:
            self.logger[name].log(scalar, step=step)
        else:
            self.logger[name].log(scalar)
            
    def close(self):
        #self.logger.close()
        return