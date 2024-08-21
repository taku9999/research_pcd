##### 2024-07-27_16-33-43_5Hz 【5Hz human_100】
# ----- 制御用パラメータ ---------------------------
VOXEL_READ_BACK = 0.5
VOXEL_READ_CURRENT = 0.5

MIN_BOUND = np.array([0, -20, -20])
MAX_BOUND = np.array([45, 20, 20])

ICP_THD = 0.25

DISTANCE_THD = 0.5

DBSCAN_EPS = 0.75
DBSCAN_MIN_POINTS = 2

CLUSTER_MIN_Z = 2.0
CLUSTER_MAX_Z = 1.0
# ------------------------------------------------


===== !!! IA研で使用　!!! =====
##### 2024-07-27_16-33-43_5Hz 【5Hz car_100】
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