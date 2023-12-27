
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_dataset(dataset_name):
    if dataset_name == 'newaug_align_ct_xray_views_std':
        # import the dataset class
        from .alignDataSetViews import AlignDataSet
        # import the data augmentation class
        from .data_augmentation import CT_XRAY_Data_Augmentation_Multi, \
            CT_XRAY_Data_Test_Multi
        # import the method: torch warp up the batch images
        from .collate_fn import collate_gan_views
        return AlignDataSet, CT_XRAY_Data_Augmentation_Multi, CT_XRAY_Data_Test_Multi, collate_gan_views
    else:
        raise KeyError('Dataset class should select from align / ')
