===== !!! IA研で使用　!!! =====
##### 【2024/07　Droneデータ】
INPUT_INIT_PATH = "/workspace/bind_data/2024-07-27_15-53-33_1Hz/12110_949.pcd"
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