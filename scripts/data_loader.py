from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
import numpy as np
import csv
from bisect import bisect_left


def explore_data(bag_path):
    with AnyReader([bag_path]) as reader:
        print(f"{'TOPIC':<30} | {'MSG TYPE':<30} | {'COUNT':<10}")
        print("-" * 75)
        
        for connection in sorted(reader.connections, key=lambda x: x.topic):
            print(f"{connection.topic:<30} | {connection.msgtype:<30} | {connection.msgcount:<10}")

POST_TIMESTAMPS = ''
ODOM_TIMESTAMPS = ''

def find_nearest_pose(target_time,data,ts):
    idx = bisect_left(ts, target_time)
    
    if idx == 0:
        return data[0]
    if idx == len(data):
        return data[-1]

    before = data[idx - 1]                                      # Comparing the two closest values (the one before and the one at idx)
    after = data[idx]
    
    if abs(before['timestamp'] - target_time) < abs(after['timestamp'] - target_time):
        return before
    else:
        return after
    

def extract_data(bag_path,data_dir):
    
    img_data_log = []                                                   # Initializing lists to store data rows
    imu_data_log = []  
    odometry_data_log = []
    pose_data_log = []
    cam_id = ''


    with AnyReader([bag_path]) as reader:
        
        for connection, timestamp, rawdata in reader.messages():
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.msgtype == 'sensor_msgs/msg/Image':

                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                
                cam_id = 0 if connection.topic =='/snappy_cam/stereo_l' else 1
                # Please comment this line after images are saved. Optimization will folow in next commit. 
                if cam_id == 0:
                    cv2.imwrite(f"{data_dir}/frame_{timestamp}.jpg", img)

                img_data_log.append({
                    'timestamp':timestamp,
                    'cam_id':cam_id,
                    'image':img,
                    'path':f'{data_dir}/frame_{timestamp}.jpg'})
            elif connection.msgtype == 'sensor_msgs/msg/Imu':
                imu_data_log.append({
                    'timestamp': timestamp,
                    'acc_x':msg.linear_acceleration.x,
                    'acc_y':msg.linear_acceleration.y,
                    'acc_z':msg.linear_acceleration.z,
                    'gyro_x':msg.angular_velocity.x,
                    'gyro_y':msg.angular_velocity.y,
                    'gyro_z':msg.angular_velocity.z
                })
            elif connection.msgtype =='nav_msgs/msg/Odometry':
                odometry_data_log.append({
                    'timestamp':timestamp,
                    'linear_velocity':[msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z],
                    'rotation_velocity':[msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z]
                    })
            elif connection.msgtype =='geometry_msgs/msg/PoseStamped':
                pose_data_log.append({
                    'timestamp':timestamp,
                    'position':[msg.pose.position.x,msg.pose.position.y,msg.pose.position.z],
                    'orientation':[msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w]
                })
            else:
                print('Unknown object detected: ',msg)



    print('----------- NUMBER OF DATA POINTS -----------')
    print(f'Images:{len(img_data_log)} | IMU:{len(imu_data_log)} | Odometry:{len(odometry_data_log)} | Pose:{len(pose_data_log)}')
    data_synchronisation(img_data_log,imu_data_log,odometry_data_log,pose_data_log)


def data_synchronisation(img_data,imu_data,odometry_data,pose_data):
    dataset = []
    time_offset_l = -0.016684572091862235
    time_offset_r = -0.016591431247074982
    time_interval = 100000000

    for img_entry in img_data:

        end_timestamp = img_entry['timestamp'] + time_offset_l if img_entry['cam_id']==0 else img_entry['timestamp'] + time_offset_r
        start_timestamp = end_timestamp - time_interval

        global POST_TIMESTAMPS, ODOM_TIMESTAMPS
        POST_TIMESTAMPS= [p['timestamp'] for p in pose_data]
        ODOM_TIMESTAMPS = [o['timestamp'] for o in odometry_data]
        best_pose = find_nearest_pose(img_entry['timestamp'],pose_data,POST_TIMESTAMPS)
        best_odometry = find_nearest_pose(img_entry['timestamp'],odometry_data,ODOM_TIMESTAMPS)
        dataset.append({
            'timestamp': img_entry['timestamp'],
            'cam_id':img_entry['cam_id'],
            'img_path': img_entry['path'],
            'imu_start_ts':start_timestamp,
            'imu_end_ts':end_timestamp,
            'pos_x': best_pose['position'][0], 'pos_y': best_pose['position'][1], 'pos_z': best_pose['position'][2],
            'ori_w': best_pose['orientation'][3],
            'vel_x': best_odometry['linear_velocity'][0], 'vel_y': best_odometry['linear_velocity'][1], 'vel_z': best_odometry['linear_velocity'][2]
        })

    if dataset:                                                                             # Saving to CSV
        with open('data/drone_dataset.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
            writer.writeheader()
            writer.writerows(dataset)
        print(f"Dataset created with {len(dataset)} synchronized rows.")
    
    if imu_data:
        with open('data/imu_data.csv', 'w', newline='') as f:
            keys = imu_data[0].keys()
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            
            dict_writer.writeheader()
            dict_writer.writerows(imu_data) 
        print(f"Saved {len(imu_data)} IMU messages to imu_data.csv")