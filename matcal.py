import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from utils import  save_obj,read_VD, read_point, winding_number, Delaunay_triangulation,gaussian,loss_func , load_ply_points_normal,find_points_within_radius,find_overlapping_points,cal_angle,cal_degree,save_ma,save_spheres,load_ply_points_normal_sample,save_spheres1,normlize_point
import time

from mpl_toolkits.mplot3d import Axes3D
from medial_axis_approx import connect_method3,approx_medial_axis2,pole_cluster_lambda_torch,farthest_sphere_sampling_torch, approx_medial_axis_torch,approx_medial_axis, write_obj, label_pole,classfication,cal_voronoi,greedy_selection,pole_cluster_lambda,cluster,dist,connect_method2,connect_method1,cluster1,find_faces_and_lines,calculate_mat_error,find_boundary_points,project_point_onto_line,project_point_onto_triangle,calculate_barycentric_coordinates,project_point_onto_triangle_para,project_point_onto_line_para,findnewcenter,save_skel_mesh,farthest_sphere_sampling,figjj
import time
import polyscope as ps
import open3d as o3d
import numpy as np
from solver import VoronoiRBFReconstructor,compactly_supported_rbf
from sklearn.metrics.pairwise import euclidean_distances



class PairCenter:
    def __init__(self, a, b, number):
        self.a = a
        self.b = b
        self.number = number

IN=-1
OUT=1
SUR=0

# Initialize polyscope
# ps.init()
real_name = 'flower'

dilation = 0.0
inner_points = "random"
max_time_SCP = 1000 # in second
N=30 #中心数量 c                                                                                                                                                                                                                                       
mesh_scale = 0.5
# 假设你有一个包含10000个点的点云，每个点都有一个位置和一个法向量
waiter = 0.4

point_set = load_ply_points_normal('flower_new.ply')

# point_set = load_ply_points_normal('./output2/%s.ply'%real_name)

points1 = np.array(point_set[...,0:3])
normals=np.array(point_set[...,3:6])
import open3d as o3d

points=points1


min_coords = np.min(points, axis=0)
max_coords = np.max(points, axis=0)
# normalize mesh
normalized_normal=normlize_point(normals)

bbmin = points.min(0)
bbmax = points.max(0)
center = (bbmin + bbmax) * 0.5

scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
normalized_points = points

point_cloud =o3d.geometry.PointCloud()
# pts_3d = [[x, y, 0] for x, y in normalized_points]
point_cloud.points = o3d.utility.Vector3dVector(normalized_points)

# 构建八叉树
kd_tree =  o3d.geometry.KDTreeFlann(point_cloud)



save_obj("./output5/%s_pc_input.obj"%real_name, normalized_points)

points=np.array(normalized_points)
normals1=np.array(normalized_points)

#pole_cal
poles,poles_radii,pole_site,v_verts,all_pairs,inner_verts=approx_medial_axis(normalized_points, 1, 0, 0.05, False,normals,"./output5/%s_pc_input_POLE.obj"%real_name,"./output5/%s_pc_input_VORONOI.obj"%real_name)


weight_points=[]
pole_label1,inner_poles1,outer_poles1=label_pole(poles,kd_tree,normals,point_cloud)

#pole_filter
reminingpole,remainingradii,remaining_site=pole_cluster_lambda(np.array(poles),np.array(poles_radii),kd_tree,point_cloud,0.002,0.2,pole_site)
#pole_label
pole_label,inner_poles,outer_poles=label_pole(reminingpole,kd_tree,normals,point_cloud) #经过filter的


aa=np.array(reminingpole)[inner_poles]
bb=np.array(remainingradii)[inner_poles]
#center_select
cc,dd=farthest_sphere_sampling(aa,bb,normalized_points,N)

cc=np.array(cc)
 #optimize center
centers = [
  np.array(reminingpole)[outer_poles],  # External centers (not optimized)
    cc   # Internal centers (optimized)
 ]
 radii = [
    np.array(remainingradii)[outer_poles],  # External radii (not optimized)
     dd   # Internal radii (optimized)
 ]
constraints_inner = poles[inner_poles1]  # Inner poles
values_inner = poles_radii[inner_poles1] * -1  # Negative values for inner poles

constraints_outer = poles[outer_poles1]  # Outer poles
 values_outer = poles_radii[outer_poles1] * 1  # Positive values for outer poles

constraints_normalized = normalized_points  # Normalized points
values_normalized = np.zeros(len(normalized_points))  # Zero values for normalized points


 constraints = np.vstack((constraints_inner, constraints_outer, constraints_normalized))
 values = np.concatenate((values_inner, values_outer, values_normalized))

reconstructor = VoronoiRBFReconstructor(centers, radii, constraints, values, compactly_supported_rbf)
reconstructor.optimize_centers_and_weights()

final_center=reconstructor.internal_centers.detach().cpu().numpy()
final_radii=reconstructor.internal_radii.detach().cpu().numpy()

#connnect method
temppair=connect_method3(final_center,final_radii,all_pairs,v_verts)


line=[]
with open("./output5/%s_final.obj"%real_name, 'w') as out2:
        out2.write(f"{len(cc)} {len(temppair)} {0}\n")
        for iii in range(len(cc)):
            out2.write(f"v {cc[iii][0]} {cc[iii][1]} {cc[iii][2]}\n")

        for pair in temppair:
           
            line.append([pair[0],pair[1]])
            out2.write(f"l {pair[0]+1} {pair[1]+1}\n")                           


faces,lines=find_faces_and_lines(line)
faces=np.array(faces)
lines=np.array(lines)
save_ma("./output5/%s_face_final.ma"%real_name,final_center,final_radii,lines,faces)

print("所有采样点数的结果和.ma文件已成功生成。")
min_sphere2pts,min_pts2sphere=calculate_mat_error(cc,dd,faces,line,normalized_points)

print(np.max(min_pts2sphere))
print(np.max(min_sphere2pts))
# temp_radii=bb
save_skel_mesh(cc,faces, lines, "./output5/%s_face_final.obj"%real_name, "./output5/%s_edge_final.obj"%real_name)
