import torch
from torch.utils.data import DataLoader

from data.jigsaw_dataset import JigsawDataset, JigsawTestDataset
from data.rotate_dataset import RotateDataset, RotateTestDataset
from data.image_dataset import ImageDataset, ImageTestDataset
from data.concat_dataset import ConcatDataset
from data.transformers import get_jig_train_transformers, get_train_transformers
from data.transformers import get_val_transformer, get_multi_crop_transformers
from data.transformers import get_image_train_transformer, get_image_test_transformer

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_train_val_dataloader(args):
    dataset_list = args.name
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    limit = args.limit
    # image mode
    mode = args.get('mode', 'RGB')
    for dname in dataset_list:
        if args.type == 'jigsaw':
            img_transformer, tile_transformer = get_jig_train_transformers(args)
            train_dataset = JigsawDataset(dname, split='train', val_size=args.val_size,
                    img_transformer=img_transformer, tile_transformer=tile_transformer,
                    jig_classes=args.aux_classes, bias_whole_image=args.bias_whole_image)
            val_dataset = JigsawTestDataset(dname, split='val', val_size=args.val_size,
                img_transformer=get_val_transformer(args), jig_classes=args.aux_classes)
        elif args.type == 'rotate':
            img_transformer = get_train_transformers(args)
            train_dataset = RotateDataset(dname, split='train', val_size=args.val_size,
                                          img_transformer=img_transformer,
                                          rot_classes=args.aux_classes,
                                          bias_whole_image=args.bias_whole_image, mode=mode)
            val_dataset = RotateTestDataset(dname, split='val', val_size=args.val_size,
                img_transformer=get_val_transformer(args), rot_classes=args.aux_classes, mode=mode)
        elif args.type == 'image':
            img_transformer = get_image_train_transformer(args)
            train_dataset = ImageDataset(dname, split='train', val_size=args.val_size,
                                          img_transformer=img_transformer, mode=mode)
            val_dataset = ImageTestDataset(dname, split='val', val_size=args.val_size,
                img_transformer=get_val_transformer(args), mode=mode)

        if limit:
            train_dataset = Subset(train_dataset, limit)

        datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def get_target_dataloader(args):

    name = args.name
    mode = args.get('mode', 'RGB')
    if args.type == 'jigsaw':
        img_transformer, tile_transformer = get_jig_train_transformers(args)
        dataset = JigsawDataset(name, 'train', img_transformer=img_transformer,
                tile_transformer=tile_transformer, jig_classes=args.aux_classes,
                bias_whole_image=args.bias_whole_image)
    elif args.type == 'rotate':
        img_transformer = get_train_transformers(args)
        dataset = RotateDataset(name, 'train', img_transformer=img_transformer,
                rot_classes=args.aux_classes, bias_whole_image=args.bias_whole_image, mode=mode)
    elif args.type == 'image':
        img_transformer = get_image_train_transformer(args)
        dataset = ImageDataset(name, 'train', img_transformer=img_transformer, mode=mode)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader

def get_test_dataloader(args):

    name = args.name
    mode = args.get('mode', 'RGB')
    loaders = []
    img_trs = get_multi_crop_transformers(args)

    for img_tr in img_trs:
        if args.type == 'jigsaw':
            val_dataset = JigsawTestDataset(name, split='test',
                    img_transformer=img_tr, jig_classes=args.aux_classes)
        elif args.type == 'rotate':
            val_dataset = RotateTestDataset(name, split='test',
                    img_transformer=img_tr, rot_classes=args.aux_classes, mode=mode)
        elif args.type == 'image':
            val_dataset = ImageTestDataset(name, split='test',
                    img_transformer=img_tr, mode=mode)

        if args.limit and len(val_dataset) > args.limit:
            val_dataset = Subset(val_dataset, args.limit)
            print("Using %d subset of dataset" % args.limit)
        dataset = ConcatDataset([val_dataset])
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        if args.get('multi_crop', False):
            loaders.append(loader)
        else:
            return loader
    
    return loaders
