import math
import open3d as o3d
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import time


INPUT_BG_PATH = "/workspace/bind_data/ws_pcd/ronbun/scans_2024-07-25_11-33-29_bg_init.pcd"
INPUT_INIT_PATH = "/workspace/bind_data/2024-07-27_15-53-33_1Hz/12110_949.pcd"

SAVE_CROPPED_BG_PATH = INPUT_BG_PATH.split(".pcd")[0] + "_cropped" + ".pcd"
SAVE_GR_MARGE_PATH = os.path.join(os.path.dirname(INPUT_BG_PATH), "debug_gr.pcd")
SAVE_GR_RESULT = os.path.join(os.path.dirname(INPUT_BG_PATH), "transformation")

DEVICE = o3d.core.Device("CUDA:0")
DTYPE = o3d.core.float32


# ----- 制御用パラメータ ------------------------------------------
LATLON_ORIGIN = (34.978577, 135.963469)  # 原点位置の緯度経度
LATLON_POINT = (34.978521, 135.963370)   # 移動ポイントの緯度経度

ODOMETRY_YAW = np.deg2rad(-60)     # 回転角度
CROP_AREA = np.array([50, 50, 50]) # 切り取りサイズ

VOXEL_SAVE_CROP = 0.1 # 背景点群のクロップ保存

VOXEL_GLOBAL_REG_TARGET = 0.4 # 背景点群(legacy)
VOXEL_GLOBAL_REG_SOURCE = 0.4 # init点群(legacy)
VOXEL_GLOBAL_REG_THD = 0.2
ITERATIONS_GLOBAL_REG = 100000000

VOXEL_SAVE_BACK = 0.1 # 背景点群(tensor, debug_gr保存用)
VOXEL_SAVE_INIT = 0.1 # init点群(tensor, debug_gr保存用)
# ---------------------------------------------------------------



def init_cuda():
    print("=== [init_cuda] ===")
    print(" :: Start cuda access...")
    temp_time = time.time()

    pcd_dummy = o3d.t.geometry.PointCloud(DEVICE)
    pcd_dummy.point.positions = o3d.core.Tensor(np.empty((1, 3)), DTYPE, DEVICE)

    print(" :: Success! (time: {})".format(time.time() - temp_time))


def gps_distance():
    pole_radius = 6356752.314245   # 極半径(m)
    equator_radius = 6378137.0     # 赤道半径(m)

    lat_origin = math.radians(LATLON_ORIGIN[0])
    lon_origin = math.radians(LATLON_ORIGIN[1])
    lat_point = math.radians(LATLON_POINT[0])
    lon_point = math.radians(LATLON_POINT[1])

    lat_difference = lat_point - lat_origin       # 緯度差
    lon_difference = lon_point - lon_origin       # 経度差
    lat_average = (lat_origin + lat_point) / 2    # 平均緯度

    e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) / math.pow(equator_radius, 2)  # 第一離心率^2
    w = math.sqrt(1- e2 * math.pow(math.sin(lat_average), 2))  # 子午線・卯酉線曲率半径の分母
    m = equator_radius * (1 - e2) / math.pow(w, 3)             # 子午線曲率半径
    n = equator_radius / w                                     # 卯酉線曲半径

    x = n * lon_difference * math.cos(lat_average)
    y = m * lat_difference

    return x,y


def crop_background(x_gps, y_gps, pcd_t):
    origin_center = np.array([x_gps, y_gps, 0]) # 原点座標 [x, y, z]
    theta = math.radians(90) + ODOMETRY_YAW     # 回転角

    # LiDARの向きに合わせて、Z軸周りに回転する回転行列を作成
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    
    # LiDARの前方だけをROIとする、クロップ領域の中心位置を計算
    crop_center = np.array([origin_center[0] + (CROP_AREA[0] * np.cos(theta))/2, \
                            origin_center[1] + (CROP_AREA[1] * np.sin(theta))/2, \
                            origin_center[2]])
    
    bbox = o3d.t.geometry.OrientedBoundingBox(
        center = o3d.core.Tensor(crop_center,  DTYPE, DEVICE),
        rotation = o3d.core.Tensor(rotation_matrix,  DTYPE, DEVICE),
        extent = o3d.core.Tensor(CROP_AREA,  DTYPE, DEVICE)
    )

    cropped_pcd_t_cuda = pcd_t.crop(bbox)

    return cropped_pcd_t_cuda


def global_registration_fpfh(pcd, voxel_size):
    print(" :: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(" :: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(" :: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh


def global_registration_prepare(pcd_source, pcd_target):
    print(" :: Load two point clouds and disturb initial pose.")
    source = pcd_source
    target = pcd_target

    source_down, source_fpfh = global_registration_fpfh(source, VOXEL_GLOBAL_REG_SOURCE)
    target_down, target_fpfh = global_registration_fpfh(target, VOXEL_GLOBAL_REG_TARGET)

    return source_down, target_down, source_fpfh, target_fpfh


def global_registration_execute(pcd_source_down, pcd_target_down, source_fpfh, target_fpfh, voxel_size, iterations):
    distance_threshold = voxel_size * 1.5
    print(" :: RANSAC registration on downsampled point clouds.")
    print("    Since the downsampling voxel size is %.3f," % voxel_size)
    print("    we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_source_down, pcd_target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, 0.999))
    
    return result


def main():
    #  ========== CUDA access ==========
    init_cuda()


    #  ========== Preprocessing of background points (Tensor, CUDA) ==========
    # GPS座標系を合わせる
    print("=== [gps_distance] ===")
    print(" :: Calculating crop area...")
    x_gps, y_gps = gps_distance()
    print(" :: Success! (X asis: {0}, Y asis: {1})".format(x_gps, y_gps))

    # 背景点群を読み込み
    print("=== [crop_background] ===" )
    print(" :: Loading and downsampling point cloud..." )
    pcd_t_back = o3d.t.io.read_point_cloud(INPUT_BG_PATH)
    pcd_t_back = pcd_t_back.cuda()

    # 背景をROIでクロップ
    print(" :: Cropping in progress..." )
    pcd_t_back = crop_background(x_gps, y_gps, pcd_t_back)
    pcd_t_back_crop = pcd_t_back.voxel_down_sample(VOXEL_SAVE_CROP)
    o3d.t.io.write_point_cloud(SAVE_CROPPED_BG_PATH, pcd_t_back_crop)
    print(" :: Success! (save as: {0})".format(SAVE_CROPPED_BG_PATH))


    #  ========== Global Registration (No Tensor) ==========
    # 初期化用の点群を読み込み
    print("=== [global_registration] ===")
    print(" :: Loading init point cloud...")
    pcd_t_init = o3d.t.io.read_point_cloud(INPUT_INIT_PATH)
    pcd_t_init = pcd_t_init.cuda()
    pcd_init = o3d.geometry.PointCloud()
    pcd_init.points = o3d.utility.Vector3dVector(pcd_t_init.point.positions.cpu().numpy().copy())

    # 背景用点群を準備
    print(" :: Loading background point cloud...")
    pcd_back = o3d.geometry.PointCloud()
    pcd_back.points = o3d.utility.Vector3dVector(pcd_t_back.point.positions.cpu().numpy().copy())

    # グローバルレジストレーション処理
    print(" :: Start global registration...")
    gr_time = time.time()
    source_down, target_down, source_fpfh, target_fpfh = global_registration_prepare(pcd_init, pcd_back)
    result_gr = global_registration_execute(source_down, target_down, source_fpfh, target_fpfh, VOXEL_GLOBAL_REG_THD, ITERATIONS_GLOBAL_REG)
    print(" :: Save transformation...")
    
    # 回転行列を保存
    np.save(SAVE_GR_RESULT, result_gr.transformation)
    print(" :: Success! (time: {0}, save as: {1})".format(str(time.time() - gr_time), SAVE_GR_RESULT))


    #  ========== 位置合わせ後の点群を保存 ==========
    # 保存用にダウンサンプリング
    print(" :: [Debug] Rotate PCD Save...")
    pcd_t_back = pcd_t_back.voxel_down_sample(VOXEL_SAVE_BACK)
    pcd_t_init = pcd_t_init.voxel_down_sample(VOXEL_SAVE_INIT)
    
    # 回転を適応
    pcd_t_init_rotate = copy.deepcopy(pcd_t_init)
    pcd_t_init_rotate = pcd_t_init_rotate.transform(result_gr.transformation)
    
    # 着色して保存
    pcd_t_init_rotate.paint_uniform_color([1, 0.706, 0])   # yellow
    pcd_t_back.paint_uniform_color([0, 0.651, 0.929])      # blue
    pcd_t_gr_marge = pcd_t_init_rotate + pcd_t_back
    o3d.t.io.write_point_cloud(SAVE_GR_MARGE_PATH, pcd_t_gr_marge)
    print(" :: [Debug] Success! (save as: {0})".format(SAVE_GR_MARGE_PATH))



if __name__ == '__main__':
    main()
