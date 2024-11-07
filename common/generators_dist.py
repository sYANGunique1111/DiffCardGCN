# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from logging import NullHandler
# import torch
from itertools import zip_longest
from torch.utils.data import Dataset
import numpy as np

     
class ChunkedGenerator_Seq(Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        # if cameras is not None:
        #     self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        # if poses_3d is not None:
        #     self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        # self.batch_2d = np.empty((batch_size, chunk_length, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return len(self.pairs) * self.chunk_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        chunk = self.pairs[idx]
        seq_i, start_3d, end_3d, flip = chunk[0], chunk[1], chunk[2], chunk[3]
        start_2d = start_3d
        # end_2d = end_3d + self.pad - self.causal_shift
        end_2d = end_3d

        # 2D poses
        seq_2d = self.poses_2d[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            psd_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            psd_2d = seq_2d[low_2d:high_2d]

        if flip:
            # Flip 2D keypoints
            psd_2d[:, :, 0] *= -1
            psd_2d[:, self.kps_left + self.kps_right] = psd_2d[:, self.kps_right + self.kps_left]

        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d = low_3d - start_3d
            pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                psd_3d = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                psd_3d = seq_3d[low_3d:high_3d]

            if flip:
                # Flip 3D joints
                psd_3d[:, :, 0] *= -1
                psd_3d[:, self.joints_left + self.joints_right] = \
                        psd_3d[:, self.joints_right + self.joints_left]

        # Cameras
        if self.cameras is not None:
            cam = self.cameras[seq_i]
            if flip:
                # Flip horizontal distortion coefficients
                cam[2] *= -1
                cam[7] *= -1

        return cam, psd_3d, psd_2d

        # start_idx, pairs = self.next_pairs()
        # for b_i in range(start_idx, self.num_batches):
        #     chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
        #     for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
        #         # start_2d = start_3d - self.pad - self.causal_shift
        #         start_2d = start_3d
        #         # end_2d = end_3d + self.pad - self.causal_shift
        #         end_2d = end_3d

        #         # 2D poses
        #         seq_2d = self.poses_2d[seq_i]
        #         low_2d = max(start_2d, 0)
        #         high_2d = min(end_2d, seq_2d.shape[0])
        #         pad_left_2d = low_2d - start_2d
        #         pad_right_2d = end_2d - high_2d
        #         if pad_left_2d != 0 or pad_right_2d != 0:
        #             self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        #         else:
        #             self.batch_2d[i] = seq_2d[low_2d:high_2d]

        #         if flip:
        #             # Flip 2D keypoints
        #             self.batch_2d[i, :, :, 0] *= -1
        #             self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

        #         # 3D poses
        #         if self.poses_3d is not None:
        #             seq_3d = self.poses_3d[seq_i]
        #             low_3d = max(start_3d, 0)
        #             high_3d = min(end_3d, seq_3d.shape[0])
        #             pad_left_3d = low_3d - start_3d
        #             pad_right_3d = end_3d - high_3d
        #             if pad_left_3d != 0 or pad_right_3d != 0:
        #                 self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        #             else:
        #                 self.batch_3d[i] = seq_3d[low_3d:high_3d]

        #             if flip:
        #                 # Flip 3D joints
        #                 self.batch_3d[i, :, :, 0] *= -1
        #                 self.batch_3d[i, :, self.joints_left + self.joints_right] = \
        #                         self.batch_3d[i, :, self.joints_right + self.joints_left]

        #         # Cameras
        #         if self.cameras is not None:
        #             self.batch_cam[i] = self.cameras[seq_i]
        #             if flip:
        #                 # Flip horizontal distortion coefficients
        #                 self.batch_cam[i, 2] *= -1
        #                 self.batch_cam[i, 7] *= -1

        #     if self.endless:
        #         self.state = (b_i + 1, pairs)
        #     if self.poses_3d is None and self.cameras is None:
        #         yield None, None, self.batch_2d[:len(chunks)]
        #     elif self.poses_3d is not None and self.cameras is None:
        #         yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
        #     elif self.poses_3d is None:
        #         yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
        #     else:
        #         yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
        
        # if self.endless:
        #     self.state = None
        # else:
        #     enabled = False

    def batch_num(self):
        return self.num_batches
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    # def next_epoch(self):
    #     enabled = True
    #     while enabled:
    #         start_idx, pairs = self.next_pairs()
    #         for b_i in range(start_idx, self.num_batches):
    #             chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
    #             for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
    #                 # start_2d = start_3d - self.pad - self.causal_shift
    #                 start_2d = start_3d
    #                 # end_2d = end_3d + self.pad - self.causal_shift
    #                 end_2d = end_3d

    #                 # 2D poses
    #                 seq_2d = self.poses_2d[seq_i]
    #                 low_2d = max(start_2d, 0)
    #                 high_2d = min(end_2d, seq_2d.shape[0])
    #                 pad_left_2d = low_2d - start_2d
    #                 pad_right_2d = end_2d - high_2d
    #                 if pad_left_2d != 0 or pad_right_2d != 0:
    #                     self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
    #                 else:
    #                     self.batch_2d[i] = seq_2d[low_2d:high_2d]

    #                 if flip:
    #                     # Flip 2D keypoints
    #                     self.batch_2d[i, :, :, 0] *= -1
    #                     self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

    #                 # 3D poses
    #                 if self.poses_3d is not None:
    #                     seq_3d = self.poses_3d[seq_i]
    #                     low_3d = max(start_3d, 0)
    #                     high_3d = min(end_3d, seq_3d.shape[0])
    #                     pad_left_3d = low_3d - start_3d
    #                     pad_right_3d = end_3d - high_3d
    #                     if pad_left_3d != 0 or pad_right_3d != 0:
    #                         self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
    #                     else:
    #                         self.batch_3d[i] = seq_3d[low_3d:high_3d]

    #                     if flip:
    #                         # Flip 3D joints
    #                         self.batch_3d[i, :, :, 0] *= -1
    #                         self.batch_3d[i, :, self.joints_left + self.joints_right] = \
    #                                 self.batch_3d[i, :, self.joints_right + self.joints_left]

    #                 # Cameras
    #                 if self.cameras is not None:
    #                     self.batch_cam[i] = self.cameras[seq_i]
    #                     if flip:
    #                         # Flip horizontal distortion coefficients
    #                         self.batch_cam[i, 2] *= -1
    #                         self.batch_cam[i, 7] *= -1

    #             if self.endless:
    #                 self.state = (b_i + 1, pairs)
    #             if self.poses_3d is None and self.cameras is None:
    #                 yield None, None, self.batch_2d[:len(chunks)]
    #             elif self.poses_3d is not None and self.cameras is None:
    #                 yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
    #             elif self.poses_3d is None:
    #                 yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
    #             else:
    #                 yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
            
    #         if self.endless:
    #             self.state = None
    #         else:
    #             enabled = False


class UnchunkedGenerator_Seq(Dataset):
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
    
    def __len__(self):
        return len(self.poses_2d)
    
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def batch_num(self):
        return self.num_batches

    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def __getitem__(self, index):
        batch_cam, batch_3d, batch_2d = self.cameras[index], self.poses_3d[index], self.poses_2d[index]
        # batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
        # batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
        # batch_2d = None if seq_2d is None else np.expand_dims(seq_2d, axis=0)
        # batch_2d = np.expand_dims(np.pad(seq_2d,
        #                 ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
        #                 'edge'), axis=0)
        if self.augment:
            # Append flipped version
            if batch_cam is not None:
                batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                batch_cam[1, 2] *= -1
                batch_cam[1, 7] *= -1
            
            if batch_3d is not None:
                batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                batch_3d[1, :, :, 0] *= -1
                batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

            batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
            batch_2d[1, :, :, 0] *= -1
            batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            # print(batch_2d.shape)
        return batch_cam, batch_3d, batch_2d
    

    # def next_epoch(self):
    #     for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
    #         batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
    #         batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
    #         batch_2d = None if seq_2d is None else np.expand_dims(seq_2d, axis=0)
    #         # batch_2d = np.expand_dims(np.pad(seq_2d,
    #         #                 ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
    #         #                 'edge'), axis=0)
    #         if self.augment:
    #             # Append flipped version
    #             if batch_cam is not None:
    #                 batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
    #                 batch_cam[1, 2] *= -1
    #                 batch_cam[1, 7] *= -1
                
    #             if batch_3d is not None:
    #                 batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
    #                 batch_3d[1, :, :, 0] *= -1
    #                 batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

    #             batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
    #             batch_2d[1, :, :, 0] *= -1
    #             batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
    #         # print(batch_2d.shape)
    #         yield batch_cam, batch_3d, batch_2d

class UnchunkedGenerator_Seq2Seq:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment

    def batch_num(self):
        return self.num_batches
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(np.pad(seq_3d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d


class ChunkedGenerator_Seq_CARD(Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_3d_infer, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert poses_3d_infer is None or len(poses_3d_infer) == len(poses_2d), (len(poses_3d_infer), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d_infer[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        # if cameras is not None:
        #     self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        # if poses_3d is not None:
        #     self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        # self.batch_2d = np.empty((batch_size, chunk_length, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_3d_infer = poses_3d_infer
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return len(self.pairs) * self.chunk_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        chunk = self.pairs[idx]
        seq_i, start_3d, end_3d, flip = chunk[0], chunk[1], chunk[2], chunk[3]
        start_2d = start_3d
        # end_2d = end_3d + self.pad - self.causal_shift
        end_2d = end_3d

        # 2D poses
        seq_2d = self.poses_2d[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            psd_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            psd_2d = seq_2d[low_2d:high_2d]

        if flip:
            # Flip 2D keypoints
            psd_2d[:, :, 0] *= -1
            psd_2d[:, self.kps_left + self.kps_right] = psd_2d[:, self.kps_right + self.kps_left]

        # 3D poses
        if self.poses_3d is not None:
            assert self.poses_3d[seq_i].shape[0] == self.poses_3d_infer[seq_i].shape[0], "infer poses {}, 3D poses {}".\
                format(self.poses_3d_infer[seq_i].shape[0], self.poses_3d[seq_i].shape[0])
            seq_3d = self.poses_3d[seq_i]
            seq_3d_infer = self.poses_3d_infer[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d = low_3d - start_3d
            pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                psd_3d = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                psd_3d_infer = np.pad(seq_3d_infer[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                psd_3d = seq_3d[low_3d:high_3d]
                psd_3d_infer = seq_3d_infer[low_3d:high_3d]

            if flip:
                # Flip 3D joints
                psd_3d[:, :, 0] *= -1
                psd_3d[:, self.joints_left + self.joints_right] = psd_3d[:, self.joints_right + self.joints_left]
                psd_3d_infer[:, :, 0] *= -1
                psd_3d_infer[:, self.joints_left + self.joints_right] = psd_3d_infer[:, self.joints_right + self.joints_left]

        # Cameras
        if self.cameras is not None:
            cam = self.cameras[seq_i]
            if flip:
                # Flip horizontal distortion coefficients
                cam[2] *= -1
                cam[7] *= -1

        return cam, psd_3d, psd_3d_infer, psd_2d

    def batch_num(self):
        return self.num_batches
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from arguments import parse_args
    from h36m_dataset import Human36mDataset
    from camera import *

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        from common.utils import deterministic_random
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = args.downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_camera_params, out_poses_3d, out_poses_2d
    
    args = parse_args()
    dataset_path = '/users/shuoyang67/projects/zoo/dataset/H36m/annot/data_3d_' + 'h36m' + '.npz'
    dataset = Human36mDataset(dataset_path)

    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load('/users/shuoyang67/projects/zoo/dataset/H36m/annot/data_2d_' + 'h36m' + '_' + "cpn_ft_h36m_dbb" + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = args.subjects_train.split(',')
    subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
    subjects_test = args.subjects_test.split(',')
    


    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)
    test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=0, causal_shift=0, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    loader = DataLoader(test_generator, batch_size=1, shuffle=False)

    for i, cam, batch, batch_2d in enumerate(loader()):
        print("batch id: {}".format(i)) 
        print('cam shape is {}'.format(cam.shape))
        print('batch shape is {}'.format(batch.shape))
        print('batch_2d shape is {}'.format(batch_2d.shape))