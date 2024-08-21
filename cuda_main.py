import math
import open3d as o3d
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import glob
import time
import datetime
import os
from natsort import natsorted


INPUT_BG_PATH = "/workspace/bind_data/ws_pcd/IA_ronbun/scans_2024-07-25_11-33-29_bg_init_cropped.pcd"
INPUT_TRANCE = "/workspace/bind_data/ws_pcd/IA_ronbun/transformation.npy"
INPUT_CURRENT_DIR = "/workspace/bind_data/ws_pcd/temp"

SAVE_RESULT_DIR = "/workspace/bind_data/result_pcd"

DEVICE = o3d.core.Device("CUDA:0")
DTYPE = o3d.core.float32


# ----- 制御用パラメータ ---------------------------
VOXEL_READ_BACK = 0.5
VOXEL_READ_CURRENT = 0.5

MIN_BOUND = np.array([0, -20, -20])
MAX_BOUND = np.array([45, 20, 20])

ICP_THD = 1.2

DISTANCE_THD = 1.0

DBSCAN_EPS = 1.2
DBSCAN_MIN_POINTS = 2

CLUSTER_MIN_Z = 2.0
CLUSTER_MAX_Z = 1.0
# ------------------------------------------------


# ----- 動作モード切り替え用パラメータ -----------------------------------------------------------
CALC_TIME_MODE = True
# -------------------------------------------------------------------------------------------


def get_device_time(cmd):
    tmp_time = datetime.datetime.now()

    # datetime.datetime
    date_time = tmp_time + datetime.timedelta(hours=9)
    # float
    unix_time = date_time.timestamp()
    # convert(no-milliseconds)
    conv_str = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    # convert(milliseconds)
    conv_str_milli = date_time.strftime("%Y-%m-%d_%H-%M-%S_%f")

    if cmd == "date_time":
        return date_time
    elif cmd == "unix_time":
        return unix_time
    elif cmd == "conv_str":
        return conv_str
    elif cmd == "conv_str_milli":
        return conv_str_milli


def init_cuda():
    print("=== [init_cuda] ===")
    print(" :: Start cuda access...")
    temp_time = time.time()

    pcd_dummy = o3d.t.geometry.PointCloud(DEVICE)
    pcd_dummy.point.positions = o3d.core.Tensor(np.empty((1, 3)), DTYPE, DEVICE)

    print(" :: Success! (time: {})".format(time.time() - temp_time))


def crop_current(pcd_t):
    bbox = o3d.t.geometry.AxisAlignedBoundingBox(
        min_bound = o3d.core.Tensor(MIN_BOUND,  DTYPE, DEVICE),
        max_bound = o3d.core.Tensor(MAX_BOUND,  DTYPE, DEVICE)
    )
    pcd_t_crop = pcd_t.crop(bbox)
    
    return pcd_t_crop


def icp_registration(pcd_t_source, pcd_t_target, init_matrix):
    print(" :: Point-to-point ICP registration is applied on original point")
    print("    clouds to refine the alignment. This time we use a strict")
    print("    distance threshold %.3f." % ICP_THD)

    result = o3d.t.pipelines.registration.icp(
        pcd_t_source, pcd_t_target, ICP_THD, 
        init_source_to_target = init_matrix,
        estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result.transformation


def distance(pcd_t_source, pcd_t_target):
    pcd_source = pcd_t_source.to_legacy()
    pcd_target = pcd_t_target.to_legacy()

    dists = pcd_source.compute_point_cloud_distance(pcd_target)
    dists = np.asarray(dists)
    ind = np.where(dists > DISTANCE_THD)[0]

    ind_cuda = o3d.core.Tensor(ind).cuda()
    pcd_t_distance = pcd_t_source.select_by_index(ind_cuda)

    return pcd_t_distance


def clustering(pcd_t):
    clustering_time = time.time()
    
    labels = pcd_t.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False)
    max_label = labels.max().item()
    print(" :: Point cloud has {0} clusters".format(max_label + 1))
    
    clustering_time = time.time() - clustering_time
    
    prediction_time = time.time()

    pcd_xyz = pcd_t.point.positions.cpu().numpy().copy()
    np_labels = labels.cpu().numpy().copy()
    
    temp_pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CUDA:0"))
    # bbox_list = np.empty((0, 3, 3), dtype=float)

    if max_label >= 0:
        for i in range(max_label + 1):
            clus_xyz = pcd_xyz[np_labels == i]
            temp_pcd.point.positions = o3d.core.Tensor(clus_xyz, DTYPE, DEVICE)
            
            temp_min = temp_pcd.get_min_bound()
            temp_max = temp_pcd.get_max_bound()
            if temp_min[2] >= CLUSTER_MIN_Z or temp_max[2] <= CLUSTER_MAX_Z:
                print(" :: Cluster {0} is removed by noise determination".format(i))
                np_labels[np_labels == i] = -1
                continue
            
            temp_center = temp_pcd.get_center()
            temp_max= temp_pcd.get_max_bound()
            
            # bbox_list = np.vstack((bbox_list, [np.array([temp_min.cpu().numpy().copy(), temp_center.cpu().numpy().copy(), temp_max.cpu().numpy().copy()])]))
    
    if not CALC_TIME_MODE:
        # クラスタに着色
        colors = plt.get_cmap("tab20")(labels.cpu().numpy() / (max_label if max_label > 0 else 1))
        colors = o3d.core.Tensor(colors[:, :3], DTYPE, DEVICE)
        colors[np_labels < 0] = 0
        pcd_t.point.colors = colors
        
        # クラスタのみを抽出
        ind = np.where(np_labels >= 0)[0]
        ind_cuda = o3d.core.Tensor(ind).cuda()
        pcd_t = pcd_t.select_by_index(ind_cuda)
    
    prediction_time = time.time() - prediction_time
    
    return pcd_t, clustering_time, prediction_time


def main():
     #  ========== 保存用パスの作成 ==========
    result_path = SAVE_RESULT_DIR + "/" + str(get_device_time("conv_str")) + "/"
    os.makedirs(result_path)
    print(" :: [Debug] result file save to: {0}".format(result_path))
    
    #  ========== CUDA access ==========
    init_cuda()
    
    #  ========== Load BG & tranceformation ==========
    print("=== [prepare] ===" )
    print(" :: Loading and downsampling BG point cloud..." )
    pcd_t_back = o3d.t.io.read_point_cloud(INPUT_BG_PATH)
    pcd_t_back = pcd_t_back.cuda()
    pcd_t_back = pcd_t_back.voxel_down_sample(VOXEL_READ_BACK)
    print(" :: Loading tranceformation..." )
    result_trance = o3d.core.Tensor(np.load(INPUT_TRANCE), DTYPE, DEVICE)
    print(" :: Success!")
    

    #  ========== メインのループ ==========
    pcd_path_list = natsorted(glob.glob(INPUT_CURRENT_DIR + "/*.pcd"))

    for pcd_path in pcd_path_list:
        print("\n********** [ {0} ] **********".format(pcd_path.split("/")[-1]))
        
        pcd_t_current = o3d.t.io.read_point_cloud(pcd_path)
        pcd_t_current = pcd_t_current.cuda()
        
        #  ========== Prepare (Tensor, CUDA) ==========
        print("=== [prepare] ===")
        print(" :: Downsampling and Cropping Current point cloud..." )
        prepare_time = time.time()
        pcd_t_current = pcd_t_current.voxel_down_sample(VOXEL_READ_CURRENT)
        pcd_t_current = crop_current(pcd_t_current)
        prepare_time = time.time() - prepare_time
        print(" :: Success! (time: {0})".format(str(prepare_time)))
        
        #  ========== ICP resistration (Tensor, CUDA) ==========
        print("=== [icp_registration] ===")
        print(" :: Start icp registration...")
        icp_time = time.time()
        result_trance = icp_registration(pcd_t_current, pcd_t_back, result_trance)
        icp_time = time.time() - icp_time
        print(" :: Success! (time: {0})".format(str(icp_time)))
        
        if not CALC_TIME_MODE:
            print(" :: [Debug] Rotate point cloud...")
            pcd_t_current = pcd_t_current.transform(result_trance)

            print(" :: [Debug] Debug PCD save...")
            pcd_t_current.paint_uniform_color([1, 0.706, 0])   # yellow
            pcd_t_back.paint_uniform_color([0, 0.651, 0.929])  # blue
            pcd_t_icp = pcd_t_current + pcd_t_back
            debug_icp_path = result_path + pcd_path.split("/")[-1].rsplit('.', 1)[0] + "_debug_icp.pcd"
            o3d.t.io.write_point_cloud(debug_icp_path, pcd_t_icp)
            print(" :: [Debug] Success! (save as: {0})".format(debug_icp_path))
        
        #  ========== Rotation (Tensor, CUDA) ==========
        print("=== [rotation] ===")
        rotaion_time = time.time()
        pcd_t_current.transform(result_trance)
        rotaion_time = time.time() - rotaion_time
        print(" :: Success! (time: {0})".format(str(rotaion_time)))
        
        
        #  ========== Distance (Tensor, CUDA) ==========
        print("=== [distance] ===")
        print(" :: Start compute distance...")
        distance_time = time.time()
        pcd_t_distance = distance(pcd_t_current, pcd_t_back)
        distance_time = time.time() - distance_time
        print(" :: Success! (time: {0})".format(str(distance_time)))
        
        if not CALC_TIME_MODE:
            print(" :: [Debug] Debug PCD save...")
            debug_distance_path = result_path + pcd_path.split("/")[-1].rsplit('.', 1)[0] + "_debug_distance.pcd"
            o3d.t.io.write_point_cloud(debug_distance_path, pcd_t_distance)
            print(" :: [Debug] Success! (save as: {0})".format(debug_distance_path))
        
        #  ========== Clustering (Tensor, CPU) ==========
        print("=== [clustering] ===")
        print(" :: Start dbscan clustering...")
        pcd_t_cluster, clustering_time, prediction_time = clustering(pcd_t_distance)
        print(" :: Success! (time: {0}, {1})".format(str(clustering_time), str(prediction_time)))
        
        if not CALC_TIME_MODE:
            print(" :: [Debug] Debug PCD save...")
            debug_cluster_path = result_path + pcd_path.split("/")[-1].rsplit('.', 1)[0] + "_debug_cluster.pcd"
            o3d.t.io.write_point_cloud(debug_cluster_path, pcd_t_cluster)
            print(" :: [Debug] Success! (merge cluster save as: {0})".format(debug_cluster_path))
        
        # ===== 処理時間をCSVに記録 =====
        if CALC_TIME_MODE:
            with open(result_path + "log_time.csv", mode="a") as f:
                f.write(str(prepare_time))
                f.write("," + str(icp_time))
                f.write("," + str(rotaion_time))
                f.write("," + str(distance_time))
                f.write("," + str(clustering_time))
                f.write("," + str(prediction_time) + "\r\n")



if __name__ == '__main__':
    main()
