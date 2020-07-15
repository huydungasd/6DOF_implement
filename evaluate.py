import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc', 'cea'], help='Training dataset name (\'oxiod\' or \'euroc\' or \'cea\')')
    parser.add_argument('model', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model)

    window_size = 80
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu6.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu1.csv')

        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi6.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi1.csv')

    elif args.dataset == 'euroc':
        imu_data_filenames.append('MH_02_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_04_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_03_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_02_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_01_easy/mav0/imu0/data.csv')

        gt_data_filenames.append('MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_02_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_01_easy/mav0/state_groundtruth_estimate0/data.csv')

    elif args.dataset == 'cea':
        for i in range(38,42):
            imu_data_filenames.append(f'H:\\data\\0\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\0\\data_deep\\gt\\{i}.csv')
        for i in range(46,51):
            imu_data_filenames.append(f'H:\\data\\1\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\1\\data_deep\\gt\\{i}.csv')
        for i in range(49,54):
            imu_data_filenames.append(f'H:\\data\\2\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\2\\data_deep\\gt\\{i}.csv')
        for i in range(45,50):
            imu_data_filenames.append(f'H:\\data\\3\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\3\\data_deep\\gt\\{i}.csv')
        for i in range(47,52):
            imu_data_filenames.append(f'H:\\data\\4\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\4\\data_deep\\gt\\{i}.csv')
        for i in range(46,51):
            imu_data_filenames.append(f'H:\\data\\5\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\5\\data_deep\\gt\\{i}.csv')
        for i in range(47,52):
            imu_data_filenames.append(f'H:\\data\\6\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\6\\data_deep\\gt\\{i}.csv')
        for i in range(51,57):
            imu_data_filenames.append(f'H:\\data\\7\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\7\\data_deep\\gt\\{i}.csv')
        for i in range(54,60):
            imu_data_filenames.append(f'H:\\data\\8\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\8\\data_deep\\gt\\{i}.csv')
        for i in range(50,56):
            imu_data_filenames.append(f'H:\\data\\9\\data_deep\\imu\\{i}.csv')
            gt_data_filenames.append(f'H:\\data\\9\\data_deep\\gt\\{i}.csv')

    traj, xy_rmses, yz_rmses, zx_rmses = [], [], [], []
    for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
        if args.dataset == 'oxiod':
            gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'euroc':
            gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'cea':
            gyro_data, acc_data, pos_data, ori_data = load_cea_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
        
        if args.dataset == 'oxiod':
            [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)
        elif args.dataset == 'euroc':
            [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)
        elif args.dataset == 'cea':
            [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)

        print(yhat_delta_p.shape)
        gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        if args.dataset == 'oxiod':
            pred_trajectory = pred_trajectory[0:200, :]
            gt_trajectory = gt_trajectory[0:200, :]
        elif args.dataset == 'cea':
            pred_trajectory = pred_trajectory
            gt_trajectory = gt_trajectory

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))
        xy_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, [0, 1]] - gt_trajectory[:, [0, 1]], axis=-1))))
        yz_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, [1, 2]] - gt_trajectory[:, [1, 2]], axis=-1))))
        zx_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, [2, 0]] - gt_trajectory[:, [2, 0]], axis=-1))))

        print(f'Trajectory RMSE, sequence {cur_imu_data_filename}: {trajectory_rmse}\nx:{xy_rmse}\ty:{yz_rmse}\tz:{zx_rmse}')
        traj.append(trajectory_rmse)
        if trajectory_rmse > 0.06:
            print('\n\n\n')
        xy_rmses.append(xy_rmse)
        yz_rmses.append(yz_rmse)
        zx_rmses.append(zx_rmse)
    
    plt.figure()
    plt.boxplot([traj, xy_rmses, yz_rmses, zx_rmses], labels=['trajectory rmse', 'xy rmse', 'yz rmse', 'zx rmse'])
    plt.show()

if __name__ == '__main__':
    main()