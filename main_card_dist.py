import os
import time
import copy
import random
import errno
import sys
import wandb
import numpy as np
from time import time
# from einops import rearrange 
import os.path as path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from common.loss import *
# from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar
from common.diffusionpose_card import *
# from common.generators_dist import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.generators_dist import ChunkedGenerator_Seq_CARD
# from common.log import Logger, savefig
# from common.utils import AverageMeter, lr_decay, save_ckpt
# from common.data_utils import fetch, read_3d_data, create_2d_data, read_3d_data_norm, create_2d_data_norm
from common.camera import *
# from common.loss import mpjpe, p_mpjpe, PCK_loss, AUC_loss
# from common import parse_args
from common.arguments import parse_args
from tqdm import tqdm

def load_infer_data(path):
    data = np.load(path, allow_pickle=True)
    poses = data.item()['positions_3d']
    errors = data.item()['infer error']
    return poses, errors

def fetch(keypoints, dataset, infer_dataset, subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_3d_infer = []
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

            # if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
            #     poses_3d = dataset[subject][action]['positions_3d']
            #     assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            #     for i in range(len(poses_3d)): # Iterate across cameras
            #         out_poses_3d.append(poses_3d[i])
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                poses_3d_infer = infer_dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    out_poses_3d_infer.append(poses_3d_infer[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_3d_infer, out_poses_2d


def runner(rank, args, train_data, test_data, joints_info):

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    lr = args.learning_rate
    joints_left, joints_right, kps_left, kps_right = joints_info
    model_pos = D3DP(args, joints_left, joints_right, is_train=True).cuda()
    model_pos = DDP(module=model_pos, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    min_loss = args.min_loss
    losses_3d_train = []
    losses_3d_pos_train = []
    losses_3d_diff_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_3d_depth_valid = []

    epoch = 0
    best_epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    # get training loader
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size//args.number_of_frames// args.world_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    valid_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=True)
    valid_loader = DataLoader(test_data, batch_size=args.batch_size//args.number_of_frames // args.world_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=valid_sampler)
    # train_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
    #                                    pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
    #                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    # train_generator_eval = UnchunkedGenerator_Seq(cameras_train, poses_train, poses_train_2d,
    #                                           pad=pad, causal_shift=causal_shift, augment=False)
    # if not args.nolog:
    #     writer.add_text(args.log+'_'+TIMESTAMP + '/Training Frames', str(train_generator_eval.num_frames()))
    if dist.get_rank() == args.reduce_rank:
        model_params = 0
        for parameter in model_pos.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
    

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_pos_train = 0
        epoch_loss_3d_diff_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        iteration = 0

        # Just train 1 time, for quick debug
        model_pos.train()
        model_pos.module.change_state_to('train')
        if dist.get_rank() == args.reduce_rank:
            bar = Bar('Train', max=len(train_loader))
        for _, inputs_3d, inputs_3d_infer, inputs_2d in train_loader:
            # if notrain:break
            # notrain=True

            # if cameras_train is not None:
            #     cameras_train = torch.from_numpy(cameras_train.astype('float32'))
            # inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            # inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            # inputs_3d = batch_3d
            # inputs_2d = batch_2d
          
            # inputs_3d = inputs_3d.cuda()
            # inputs_2d = inputs_2d.cuda()
            # if cameras_train is not None:
            #     cameras_train = cameras_train.cuda()
            # inputs_traj = inputs_3d[:, :, :1].clone()
            # inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d.cuda(), inputs_3d.cuda(), inputs_3d_infer.cuda())

            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d.cuda())

            # loss_total = loss_3d_pos
            
            loss_3d_pos.backward()

            # loss_total = torch.mean(loss_total)

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.cpu().detach().item()
            epoch_loss_3d_pos_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            optimizer.step()

            iteration += 1

            if dist.get_rank() == args.reduce_rank:
                bar.suffix = '({batch}/{size}) Batch loss: {data:.3f} mm'\
                    .format(batch=iteration , size=len(train_loader), data=loss_3d_pos.item()*1000)
                bar.next()

            

        if dist.get_rank() == args.reduce_rank:
            bar.finish()

        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_3d_pos_train.append(epoch_loss_3d_pos_train / N)
        # torch.cuda.empty_cache()

        # del inputs_3d, inputs_2d, inputs_3d_infer, predicted_3d_pos
        del predicted_3d_pos
        torch.cuda.empty_cache()
        
        # End-of-epoch evaluation
        with torch.no_grad():
            # model_pos_test_temp.load_state_dict(model_pos_train.state_dict(), strict=False)
            # model_pos_test_temp.eval()
            model_pos.module.change_state_to('test')
            epoch_loss_3d_valid = 0
            # epoch_loss_3d_depth_valid = 0
            # epoch_loss_traj_valid = 0
            # epoch_loss_2d_valid = 0
            # epoch_loss_3d_vel = 0
            N = 0
            # iteration = 0
            # if not args.no_eval:
            # Evaluate on test set
            if dist.get_rank() == args.reduce_rank:
                bar_test = Bar('Test', max=len(valid_loader))
            iteration_test = 0
            for _, inputs_3d, inputs_3d_infer, inputs_2d in valid_loader:

                ##### apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                # ##### convert size
                # inputs_3d_p = inputs_3d
                # inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                # inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

               
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d[:, :, 0] = 0


                predicted_3d_pos = model_pos(inputs_2d, inputs_3d, inputs_3d_infer,
                                                input_2d_flip=inputs_2d_flip)  # b, t, h, f, j, c

                predicted_3d_pos[:, :, :, :, 0] = 0

                error = mpjpe_diffusion(predicted_3d_pos.cpu(), inputs_3d.cpu())


                dist.reduce(error.cuda(), dst=args.reduce_rank, op=dist.ReduceOp.SUM)

                # if iteration == 0:
                #     epoch_loss_3d_valid = inputs_3d.shape[0] * inputs_3d.shape[1] * error.clone()
                # else:
                epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.cpu().detach() / args.world_size

                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                iteration_test += 1
                

                if dist.get_rank() == args.reduce_rank:
                    bar_test.suffix = '({batch_}/{size}) Batch loss: {data:.3f} mm'\
                        .format(batch_=iteration_test , size=len(valid_loader), data=error.item()/args.world_size*1000)
                    bar_test.next()

                # del inputs_3d, loss_3d_pos, predicted_3d_pos
                torch.cuda.empty_cache()
                # iteration += 1


            losses_3d_valid.append(epoch_loss_3d_valid / N)
    
        elapsed = (time() - start_time) / 60
        if dist.get_rank() == args.reduce_rank:
            print('\n')
            print('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 3d_pos_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_pos_train[-1] * 1000,
                losses_3d_valid[-1][0] * 1000
            ))

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            f = open(log_path, mode='a')
            f.write('[%d] time %.2f lr %f 3d_train %f 3d_pos_train %f 3d_pos_valid %f\n' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_pos_train[-1] * 1000,
                losses_3d_valid[-1][0] * 1000
            ))
            f.close()
            wandb.log({"train loss (mm)": losses_3d_pos_train[-1] * 1000, "Test loss (mm)": losses_3d_valid[-1][0] * 1000})


        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
                # 'min_loss': min_loss
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
        if losses_3d_valid[-1][0] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

            f = open(log_path, mode='a')
            f.write('best epoch\n')
            f.close()



def main(args):

    wandb.init(
    # Set the project where this run will be logged
    project="D3DP CARD DIST",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    },
)
    # dataset loading
    print('Loading dataset...')
    dataset_path = '/users/shuoyang67/projects/zoo/dataset/H36m/annot/data_3d_' + args.dataset + '.npz'
    infer_dataset, infer_errors = load_infer_data('/users/shuoyang67/projects/zoo/dataset/H36m/annot/data_3d_h36m_mixste.npy')
    print('infer error is {} mm'.format(infer_errors))
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

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
    keypoints = np.load('/users/shuoyang67/projects/zoo/dataset/H36m/annot/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    ###################
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
    # subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
    if not args.render:
        subjects_test = args.subjects_test.split(',')
    else:
        subjects_test = [args.viz_subject]


    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

 
    if not args.evaluate:
        print('** Note: reported losses are averaged over all frames.')
        print('** The final evaluation will be carried out after the last training epoch.')
        cameras_train, poses_train, poses_infer_train, poses_train_2d = fetch(keypoints, dataset, infer_dataset, subjects_train, action_filter, subset=args.subset)
        cameras_test, poses_test, poses_infer_test, poses_test_2d = fetch(keypoints, dataset, infer_dataset, subjects_test, action_filter)

        # # for test purpose
        # cameras_train, poses_train, poses_infer_train, poses_train_2d = [cameras_train[0]], [poses_train[0]], [poses_infer_train[0]], [poses_train_2d[0]]
        # cameras_test, poses_test, poses_infer_test, poses_test_2d = [cameras_test[0]], [poses_test[0]], [poses_infer_test[0]], [poses_test_2d[0]]


        train_data = ChunkedGenerator_Seq_CARD(args.batch_size//args.number_of_frames, cameras_train, poses_train, poses_infer_train, poses_train_2d, args.number_of_frames,
                                       pad=0, causal_shift=0, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        test_data= ChunkedGenerator_Seq_CARD(args.batch_size//args.number_of_frames, cameras_test, poses_test, poses_infer_test, poses_test_2d,args.number_of_frames,
                                              pad=0, causal_shift=0, shuffle=False, augment=False)
        
        print('INFO: Training on {} frames'.format(train_data.num_frames()))
        mp.spawn(runner, args=(args, train_data, test_data, [joints_left, joints_right, kps_left, kps_right]), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    # args, unknown = parse_args.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.evaluate != '':
        description = "Evaluate!"
    elif args.evaluate == '':
        
        description = "Train!"

    os.environ['MASTER_PORT'] = args.master_port
    os.environ['MASTER_ADDR'] = args.master_addr 
    print(description)
    print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    print(args)

    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    if args.checkpoint=='':
        args.checkpoint = args.log
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    main(args)