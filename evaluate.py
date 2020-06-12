import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc', 'cea'], help='Training dataset name (\'oxiod\' or \'euroc\' or \'cea\')')
    parser.add_argument('model', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model)

    window_size = 200
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
        for i in range(65, 73):
            imu_data_filenames.append(f'data_deep/data1/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data1/gt/{i}.csv')
        for i in range(85, 95):
            imu_data_filenames.append(f'data_deep/data2/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data2/gt/{i}.csv')
        for i in range(90, 100):
            imu_data_filenames.append(f'data_deep/data3/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data3/gt/{i}.csv')
        for i in range(135, 151):
            imu_data_filenames.append(f'data_deep/data4/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data4/gt/{i}.csv')
        for i in range(135, 151):
            imu_data_filenames.append(f'data_deep/data5/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data5/gt/{i}.csv')
        for i in range(25, 30):
            imu_data_filenames.append(f'data_deep/data6/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data6/gt/{i}.csv')
        for i in range(100, 120):
            imu_data_filenames.append(f'data_deep/data7/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data7/gt/{i}.csv')
        for i in range(130, 144):
            imu_data_filenames.append(f'data_deep/data8/imu/{i}.csv')
            gt_data_filenames.append(f'data_deep/data8/gt/{i}.csv')

    traj, x_rmses, y_rmses, z_rmses = [], [], [], []
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

        gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        if args.dataset == 'oxiod':
            pred_trajectory = pred_trajectory[0:200, :]
            gt_trajectory = gt_trajectory[0:200, :]
        elif args.dataset == 'cea':
            pred_trajectory = pred_trajectory
            gt_trajectory = gt_trajectory

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))
        x_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 0] - gt_trajectory[:, 0], axis=-1))))
        y_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 1] - gt_trajectory[:, 1], axis=-1))))
        z_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory[:, 2] - gt_trajectory[:, 2], axis=-1))))

        print(f'Trajectory RMSE, sequence {cur_imu_data_filename}: {trajectory_rmse}\nx:{x_rmse}\ty:{y_rmse}\tz:{z_rmse}')
        traj.append(trajectory_rmse)
        x_rmses.append(x_rmse)
        y_rmses.append(y_rmse)
        z_rmses.append(z_rmse)
    
    plt.figure()
    plt.boxplot([traj, x_rmses, y_rmses, z_rmses], labels=['trajectory rmse', 'x rmse', 'y rmse', 'z rmse'])
    plt.show()

if __name__ == '__main__':
    main()