import torch
import random
import pprint
from utils.config import get_config
from utils.utils import get_logger,set_random_seed 

from models.aux_model import AuxModel
from models.cdan_model import CDANModel
from data.data_loader import get_train_val_dataloader
from data.data_loader import get_target_dataloader
from data.data_loader import get_test_dataloader

def main():
    config = get_config()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    # fix random seed to reproduce results
    set_random_seed(config.random_seed)
    logger.info('Random seed: {:d}'.format(config.random_seed))
    logger.info(pprint.pformat(config))

    if config.method in ['src', 'jigsaw', 'rotate']:
        model = AuxModel(config, logger)
    elif config.method in ['cdan', 'cdan+e', 'dann']:
        model = CDANModel(config, logger)
    else:
        raise ValueError("Unknown method: %s" % config.method)
    
    # create data loaders
    src_loader, val_loader = get_train_val_dataloader(config.datasets.src)
    test_loader = get_test_dataloader(config.datasets.test)

    tar_loader = None
    if config.datasets.get('tar', None):
        tar_loader = get_target_dataloader(config.datasets.tar)

    # main loop
    if config.mode == 'train':
        model.train(src_loader, tar_loader, val_loader, test_loader)

    elif config.mode == 'test':
        model.test(test_loader)

if __name__ == '__main__':
    main()
