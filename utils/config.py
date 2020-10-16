import os
import json
import yaml
import argparse
from easydict import EasyDict
from utils.dirs import create_dirs

def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict

def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        config_dict = yaml.load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config, config_dict

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-s', '--seed',
        default=100,
        type=int,
        help='The random seed')
    args = argparser.parse_args()
    return args

def get_config():
    args = get_args()
    config_file = args.config
    random_seed = args.seed

    if config_file.endswith('json'):
        config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    config.random_seed = random_seed
    config.cache_dir = os.path.join("cache", '{}_{}'.format(config.exp_name, config.random_seed))
    config.model_dir = os.path.join(config.cache_dir, 'models')
    config.log_dir = os.path.join(config.cache_dir, 'logs')
    config.img_dir = os.path.join(config.cache_dir, 'imgs')
    
    # create the experiments dirs
    create_dirs([config.cache_dir, config.model_dir,
        config.log_dir, config.img_dir])

    return config
