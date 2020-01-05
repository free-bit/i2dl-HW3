from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from exercise_code.data_utils import get_keypoints, get_image
from exercise_code.transforms import Normalize, ToTensor

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        #######################################################################
        # DONE:                                                               #
        # Return the length of the dataset                                    #
        #######################################################################

        return self.key_pts_frame.shape[0]

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def __getitem__(self, idx):
        sample = {'image': None, 'keypoints': None}
        #######################################################################
        # DONE:                                                               #
        # Return the idx sample in the Dataset. A sample should be a          #
        # dictionary where the key, value should be like                      #
        #        {'image': image of shape [C, H, W],                          #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}         #
        #######################################################################

        df = self.key_pts_frame
        sample['image'] = get_image(idx, df)[np.newaxis, ...]
        sample['keypoints'] = get_keypoints(idx, df)

        if self.transform:
            sample = self.transform(sample)

        return sample

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample
