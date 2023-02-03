# all 19 VTAB datasets
vtab_datasets = [
    {'name': "caltech101", 'task': None, 'model_name': "caltech101", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "sun397", 'task': None, 'model_name': "sun397", 'category': "natural",
     'num_classes': 397, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "dtd", 'task': None, 'model_name': "dtd", 'category': "natural",
     'num_classes': 47, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "eurosat", 'task': None, 'model_name': "eurosat", 'category': "specialized",
     'num_classes': 10, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "resisc45", 'task': None, 'model_name': "resisc45", 'category': "specialized",
     'num_classes': 45, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "patch_camelyon", 'task': None, 'model_name': "patch_camelyon", 'category': "specialized",
     'num_classes': 2, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "diabetic_retinopathy_detection/btgraham-300", 'model_name': "diabetic_retinopathy", 'category': "specialized",
     'num_classes': 5, 'image_size': 384, 'bit_image_size': 384, 'task': None, 'enabled': True},
    {'name': "clevr", 'task': "count", 'model_name': "clevr-count", 'category': "structured",
     'num_classes': 8, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "clevr", 'task': "distance", 'model_name': "clevr-distance", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dsprites", 'task': "location", 'model_name': "dsprites-xpos", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "dsprites", 'task': "orientation", 'model_name': "dsprites-orientation", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "smallnorb", 'task': "azimuth", 'model_name': "smallnorb-azimuth", 'category': "structured",
     'num_classes': 18, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "smallnorb", 'task': "elevation", 'model_name': "smallnorb-elevation", 'category': "structured",
     'num_classes': 9, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dmlab", 'task': None, 'model_name': "dmlab", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "kitti", 'task': None, 'model_name': "kitti-distance", 'category': "structured",
     'num_classes': 4, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

vtab_natural_datasets = [
    {'name': "caltech101", 'task': None, 'model_name': "caltech101", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "sun397", 'task': None, 'model_name': "sun397", 'category': "natural",
     'num_classes': 397, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "dtd", 'task': None, 'model_name': "dtd", 'category': "natural",
     'num_classes': 47, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

vtab_specialized_datasets = [
    {'name': "eurosat", 'task': None, 'model_name': "eurosat", 'category': "specialized",
     'num_classes': 10, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "resisc45", 'task': None, 'model_name': "resisc45", 'category': "specialized",
     'num_classes': 45, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "patch_camelyon", 'task': None, 'model_name': "patch_camelyon", 'category': "specialized",
     'num_classes': 2, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "diabetic_retinopathy_detection/btgraham-300", 'model_name': "diabetic_retinopathy", 'category': "specialized",
     'num_classes': 5, 'image_size': 384, 'bit_image_size': 384, 'task': None, 'enabled': True},
]

vtab_structured_datasets = [
    {'name': "clevr", 'task': "count", 'model_name': "clevr-count", 'category': "structured",
     'num_classes': 8, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "clevr", 'task': "distance", 'model_name': "clevr-distance", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dsprites", 'task': "location", 'model_name': "dsprites-xpos", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "dsprites", 'task': "orientation", 'model_name': "dsprites-orientation", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "smallnorb", 'task': "azimuth", 'model_name': "smallnorb-azimuth", 'category': "structured",
     'num_classes': 18, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "smallnorb", 'task': "elevation", 'model_name': "smallnorb-elevation", 'category': "structured",
     'num_classes': 9, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dmlab", 'task': None, 'model_name': "dmlab", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "kitti", 'task': None, 'model_name': "kitti-distance", 'category': "structured",
     'num_classes': 4, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

# individual datasets
caltech101 = [
    {'name': "caltech101", 'task': None, 'model_name': "caltech101", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

cifar10 = [
    {'name': "cifar10", 'task': None, 'model_name': "cifar10", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
]

cifar100 = [
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True}
]

oxford_flowers = [
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

oxford_iiit_pet = [
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

sun397 = [
    {'name': "sun397", 'task': None, 'model_name': "sun397", 'category': "natural",
     'num_classes': 397, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

svhn_cropped = [{'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
                 'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True}
]

dtd = [
    {'name': "dtd", 'task': None, 'model_name': "dtd", 'category': "natural",
     'num_classes': 47, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

eurosat = [
    {'name': "eurosat", 'task': None, 'model_name': "eurosat", 'category': "specialized",
     'num_classes': 10, 'image_size': 384, 'bit_image_size': 128, 'enabled': True}
]

resisc45 = [
    {'name': "resisc45", 'task': None, 'model_name': "resisc45", 'category': "specialized",
     'num_classes': 45, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

patch_camelyon = [
    {'name': "patch_camelyon", 'task': None, 'model_name': "patch_camelyon", 'category': "specialized",
     'num_classes': 2, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

diabetic_retinopathy_detection = [
    {'name': "diabetic_retinopathy_detection/btgraham-300", 'model_name': "diabetic_retinopathy", 'category': "specialized",
     'num_classes': 5, 'image_size': 384, 'bit_image_size': 384, 'task': None, 'enabled': True}
]

clevr_count = [
    {'name': "clevr", 'task': "count", 'model_name': "clevr-count", 'category': "structured",
     'num_classes': 8, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

clevr_distance = [
    {'name': "clevr", 'task': "distance", 'model_name': "clevr-distance", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

dsprites_location = [
    {'name': "dsprites", 'task': "location", 'model_name': "dsprites-xpos", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
]

dsprites_orientation = [
    {'name': "dsprites", 'task': "orientation", 'model_name': "dsprites-orientation", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True}
]

smallnorb_azimuth = [
    {'name': "smallnorb", 'task': "azimuth", 'model_name': "smallnorb-azimuth", 'category': "structured",
     'num_classes': 18, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

smallnorb_elevation = [
    {'name': "smallnorb", 'task': "elevation", 'model_name': "smallnorb-elevation", 'category': "structured",
     'num_classes': 9, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

dmlab = [
    {'name': "dmlab", 'task': None, 'model_name': "dmlab", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]

kitti = [
    {'name': "kitti", 'task': None, 'model_name': "kitti-distance", 'category': "structured",
     'num_classes': 4, 'image_size': 384, 'bit_image_size': 384, 'enabled': True}
]


# miscellaneous collections of datasets
few_shot_datasets = [
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "cifar10", 'task': None, 'model_name': "cifar10", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
]

small_set_datasets = [
    {'name': "cifar10", 'task': None, 'model_name': "cifar10", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "caltech101", 'task': None, 'model_name': "caltech101", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

dataset_map = {
    "vtab_1000": vtab_datasets,
    "vtab_natural": vtab_natural_datasets,
    "vtab_specialized": vtab_specialized_datasets,
    "vtab_structured": vtab_structured_datasets,
    "few_shot": few_shot_datasets,
    "small_set": small_set_datasets,
    "caltech101": caltech101,
    "cifar10": cifar10,
    "cifar100": cifar100,
    "oxford_flowers102": oxford_flowers,
    "oxford_iiit_pet": oxford_iiit_pet,
    "sun397": sun397,
    "svhn_cropped": svhn_cropped,
    "dtd": dtd,
    "eurosat": eurosat,
    "resisc45": resisc45,
    "patch_camelyon": patch_camelyon,
    "diabetic_retinopathy_detection": diabetic_retinopathy_detection,
    "clevr_count": clevr_count,
    "clevr_distance": clevr_distance,
    "dsprites_location": dsprites_location,
    "dsprites_orientation": dsprites_orientation,
    "smallnorb_azimuth": smallnorb_azimuth,
    "smallnorb_elevation": smallnorb_elevation,
    "dmlab": dmlab,
    "kitti": kitti
}
