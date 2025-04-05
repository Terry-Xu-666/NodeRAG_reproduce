import argparse
import os
import shutil
from ..utils import YamlHandler

args = argparse.ArgumentParser()
args.add_argument('-f','--folder',type=str,required=True)


args = args.parse_args()

config_path = os.path.join(args.folder,'Node_config.yaml')
input_folder = os.path.join(args.folder,'input')
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    
if not os.path.exists(config_path):


    shutil.copyfile(os.path.join(os.path.dirname(__file__),'Node_config.yaml'),config_path)
    yaml_handler = YamlHandler(config_path)
    yaml_handler.update_config(['config','main_folder'],args.folder)
    yaml_handler.save()
    print(f'Config file created at {config_path}')


else:
    print(f'Config file already exists at {config_path}')
