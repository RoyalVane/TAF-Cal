from torchvision import transforms

from .datasets import LazyLoader, ImageSet, Mixed

PACS = ['art_painting', 'sketch', 'cartoon', 'photo']
VLCS = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
OfficeHome = ['art', 'clipart', 'product', 'real_world']
OfficeHome_larger = ['Art', 'Clipart', 'Product', 'Real_World']
DATASETS = {
    'PACS': PACS,
    'VLCS': VLCS,
    'OfficeHome': OfficeHome,
    'OfficeHome_larger': OfficeHome_larger
}
CLASSES = {'PACS': 7, 'VLCS': 5, 'OfficeHome': 65, 'OfficeHome_larger': 65}
DOMAINS = {'PACS': 3, 'VLCS': 3, 'OfficeHome': 3, 'OfficeHome_larger': 3}


def random_color_jitter(magnitude=1):
    assert magnitude > 0
    return transforms.ColorJitter(magnitude * .4, magnitude * .4,
                                  magnitude * .4, min(magnitude * .4, 0.5))


def matsuura_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4, .4)
    ])


def robustdg_train_augmentation():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def robustdg_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def normed_tensors():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def inv_norm():
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255])


def get_splits(name, leave_out=None, original=False):
    dataset_names = [d for d in DATASETS[name] if d != leave_out]
    return {
        heldout: {
            'train':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(
                        ImageSet,
                        f'paths/{name}/{dset}/train.txt' \
                            if original else \
                        f'paths/{name}/{dset}/test.txt',
                        parent_dir=f'data/{name}')
                    for dset in dataset_names if dset != heldout)),
            'val':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(ImageSet,
                               f'paths/{name}/{dset}/val.txt',
                               parent_dir=f'data/{name}')
                    for dset in dataset_names if dset != heldout)),
            'test':
            LazyLoader(ImageSet,
                       f'paths/{name}/{heldout}/test.txt',
                       parent_dir=f'data/{name}')
        }
        for heldout in dataset_names
    }, CLASSES[name], DOMAINS[name]


def ctr_train_augmentation():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
