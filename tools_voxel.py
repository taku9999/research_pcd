import open3d as o3d
import numpy as np
import time


INPUT_PATH = "/workspace/bind_data/ws_pcd/ronbun/debug_gr.pcd"
OUTPUT_PATH = INPUT_PATH.split(".pcd")[0] + "_voxel" + ".pcd"

# ----- 制御用パラメータ ----------
VOXEL_SIZE = 0.2 # 書き出し時
# ------------------------------


def common_voxel_downsample(common_pcd):
    pcd_voxel = common_pcd.voxel_down_sample(VOXEL_SIZE)

    return pcd_voxel


def main():
    print(" :: Loading point cloud...")
    point_cloud_tensor = o3d.t.io.read_point_cloud(INPUT_PATH)
    print(point_cloud_tensor)
    
    print(" :: Downsample with a voxel size %.3f." % VOXEL_SIZE)
    point_cloud_tensor = common_voxel_downsample(point_cloud_tensor)
    print(point_cloud_tensor)

    o3d.t.io.write_point_cloud(OUTPUT_PATH, point_cloud_tensor)
    print(" :: Success! (save as: {0})".format(OUTPUT_PATH))



if __name__ == "__main__":
    main()
