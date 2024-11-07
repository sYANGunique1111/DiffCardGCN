if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from common.arguments import parse_args
    from common.h36m_dataset import Human36mDataset
    from common.generators_dist import UnchunkedGenerator_Seq, ChunkedGenerator_Seq
    from common.camera import *

    def fetch(keypoints, dataset, subjects, action_filter=None, subset=1, parse_3d_poses=True):
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
                    # before = keypoints[subject][action][cam_idx].shape[0]
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                    # print("{}_{}_{} before : {}, after : {}, mocap : {}".format(subject, action, cam_idx, before, keypoints[subject][action][cam_idx].shape[0], mocap_length))

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

    cameras_train, poses_train, poses_train_2d = fetch(keypoints, dataset, subjects_train, action_filter, subset=args.subset)
    cameras_valid, poses_valid, poses_valid_2d = fetch(keypoints, dataset, subjects_test, action_filter)
    test_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
                                       pad=0, causal_shift=0, shuffle=True, augment=False,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    # test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
    #                                 pad=0, causal_shift=0, augment=False,
    #                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    loader = DataLoader(test_generator, batch_size=args.batch_size//args.stride, shuffle=False)

    print(len(loader))

    for i, (idx, cam, batch, batch_2d) in enumerate(loader):
        print("batch id: {}".format(idx)) 
        print('cam shape is {}'.format(cam.shape))
        print('batch shape is {}'.format(batch.shape))
        print('batch_2d shape is {}'.format(batch_2d.shape))