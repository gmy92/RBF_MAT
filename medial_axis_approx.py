#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:49:38 2020

@author: cfillmor
"""
import torch

import time

import scipy.spatial as spa
import numpy as np
import itertools as it

import trimesh
import json
import open3d as o3d
import math

from scipy.spatial import distance,ConvexHull
from sklearn.cluster import KMeans
from scipy.linalg import svd
from utils import  save_spheres,save_obj
from sklearn.neighbors import NearestNeighbors
import polyscope as ps
def is_coplanar(points):
    # Subtract mean from points
    points = points - np.mean(points, axis=0)
    # Perform singular value decomposition
    _, s, _ = svd(points)
    # If the smallest singular value is very close to zero, the points are coplanar
    return np.abs(s[-1]) < 1e-10

def compute_volume_or_area(vertices):
    if is_coplanar(vertices):
        # The points are coplanar, compute the area of the polygon
        # Note: This assumes the points form a convex polygon in counter-clockwise order
        area = 0.5 * np.abs(np.dot(vertices[:,0], np.roll(vertices[:,1], 1)) - np.dot(vertices[:,1], np.roll(vertices[:,0], 1)))
        return area
    else:
        # The points are not coplanar, compute the volume of the convex hull
        hull = ConvexHull(vertices, qhull_options='Q12 Pp')
        return hull.volume
IN=-1
OUT=1
UNKOWN=-3
def write_obj(path, verts, simps):
	dedup_tris = set([tuple(list(i)) for i in simps])
	dedup_tris = np.array(list(dedup_tris))

	with open(path,'w') as f:
		f.write("# Blender v2.79 (sub 0) OBJ File: 'test.blend'\n# www.blender.org\no Test.001\n")
		for v in verts:
			f.write('v ' + ' '.join([str(i) for i in v]) + '\n')
		f.write("s off\n")
		for s in simps:
			f.write('f ' + ' '.join([str(i+1) for i in s]) + '\n')
	pass

def vect(p,q):
    return ([ (p[i] - q[i]) for i in range(len(p))])

def dist(p,q):
   
    c=np.sqrt( np.dot(vect(p,q), vect(p,q)) )
    return c

def angle(v,w):
    return np.arccos( np.dot(v,w) / (np.sqrt(np.dot(v,v))*np.sqrt(np.dot(w,w) ) ) )
import torch
import itertools as it
from scipy.spatial import Voronoi

def torch_cdist_batch(vert, center, radii, batch_size=1000):
    """
    使用 PyTorch 实现分批的 cdist 距离计算，并返回最近中心的索引。
    """
    n_vert = vert.size(0)
    nearest_center_indices = []

    for start in range(0, n_vert, batch_size):
        end = min(start + batch_size, n_vert)

        # 当前批次计算距离
        dist_batch = torch.cdist(vert[start:end], center) - radii**2
        nearest_center_indices.append(torch.argmin(dist_batch, dim=1))

    return torch.cat(nearest_center_indices)

def connect_method3_torch(center, radii, all_pairs, vert, device='cuda'):
    """
    基于 Torch 的连接方法。
    """
    vert = torch.tensor(vert, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    radii = torch.tensor(radii, dtype=torch.float32, device=device)

    # 计算每个点属于的中心索引
    nearest_center_indices = torch_cdist_batch(vert, center, radii, batch_size=vert.size(0) // 10)
    point_to_cloud = {tuple(vert[i].cpu().numpy()): nearest_center_indices[i].item() for i in range(len(vert))}

    connected_clouds = set()

    # 遍历 all_pairs，找到点集之间的连接
    for idx1, idx2 in all_pairs:
        point1 = vert[idx1]
        point2 = vert[idx2]

        cloud1 = point_to_cloud[tuple(point1.cpu().numpy())]
        cloud2 = point_to_cloud[tuple(point2.cpu().numpy())]

        if cloud1 != cloud2:
            connected_clouds.add((min(cloud1, cloud2), max(cloud1, cloud2)))

    return list(connected_clouds)

def approx_medial_axis_torch(pts, MAX, z_max, alpha, plot, normal, oppath, oppath1, device='cuda:0'):
    """
    使用 PyTorch 优化的近似中轴计算。
    """
    # 转换点云数据为 PyTorch 张量
    pts = torch.tensor(pts, dtype=torch.float32, device=device)

    # Voronoi 计算
    voronoi = Voronoi(pts.cpu().numpy())
    v_verts = torch.tensor(voronoi.vertices, dtype=torch.float32, device=device)

    # 计算 poles 和 pole_radii
    pole_rad = []
    for i in range(len(pts)):
        pole_rad.extend(pole(i, pts.cpu().numpy(), voronoi))
    
    poles_array = np.array([item[0] for item in pole_rad], dtype=np.float32)
    poles_radii_array = np.array([item[1] for item in pole_rad], dtype=np.float32)
    pole_site_array = np.array([item[2] for item in pole_rad], dtype=np.int64)

    # 转换为 PyTorch 张量
    poles = torch.from_numpy(poles_array).to(device)
    poles_radii = torch.from_numpy(poles_radii_array).to(device)
    pole_site = torch.from_numpy(pole_site_array).to(device)
    unique_poles, unique_indices = torch.unique(poles, dim=0, return_inverse=True, return_counts=False)


    # # 按 poles_radii 对 unique_indices 排序
    # unique_radii = poles_radii[unique_indices]
    # sorted_indices = unique_indices[torch.argsort(unique_radii)]

    # 根据排序后的索引提取最终结果
    poles = poles[unique_indices]
    poles_radii = poles_radii[unique_indices]
    pole_site = pole_site[unique_indices]
    print(poles_radii.shape)

    # 构建点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts.cpu().numpy())

    # 构建 KDTree
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    # 标记 voronoi vertices 为内极点或外极点
    pole_label2, inner_poles2, outer_poles2 = label_pole(v_verts.cpu().numpy(), kd_tree, normal, point_cloud)
    inner_verts = v_verts[torch.tensor(inner_poles2, dtype=torch.long, device=device)]

    # 构建 inner_verts 索引
    inner_verts_set = {tuple(pt.cpu().numpy()) for pt in inner_verts}
    inner_verts_index = {pt: idx for idx, pt in enumerate(inner_verts_set)}

    # 生成 all_pairs
    all_pairs = set()
    for ridge in voronoi.ridge_vertices:
        if -1 not in ridge:
            for c in it.combinations(ridge, 2):
                point1 = tuple(voronoi.vertices[c[0]])
                point2 = tuple(voronoi.vertices[c[1]])

                # 检查是否都在 inner_verts 中
                if point1 in inner_verts_index and point2 in inner_verts_index:
                    idx1 = inner_verts_index[point1]
                    idx2 = inner_verts_index[point2]

                    if idx1 != idx2:
                        all_pairs.add(frozenset([idx1, idx2]))

    return poles.cpu().numpy(), poles_radii.cpu().numpy(), pole_site.cpu().numpy(), inner_verts.cpu().numpy(), list(all_pairs), v_verts.cpu().numpy()

def single_pole(pindex, pts, voronoi):
    p = pts[pindex]
    rindex = voronoi.point_region[pindex]
    reg = voronoi.regions[rindex]
    if -1 in reg:
        reg.remove(-1)
    r, index = max([(dist(p,j),i) for i,j in enumerate(voronoi.vertices[reg])])
   
    return [[voronoi.vertices[reg][index],r,pindex]]

def double_pole(pindex, pts, voronoi):
    p = pts[pindex]
    rindex = voronoi.point_region[pindex]
    reg = voronoi.regions[rindex]
    if -1 in reg:
        reg.remove(-1)
    
    p1,r1 ,pid= single_pole(pindex, pts, voronoi)[0]
    
    v1 = vect(p1,p)
    good_points = [ i for i in voronoi.vertices[reg] if angle(vect(i,p),v1) > np.pi/2]
    
    r2, index2 = max([(dist(p,j),i) for i,j in enumerate(good_points)])
    p2 = good_points[index2]
    return [[p1,r1,pid], [p2,r2,pid]]

def pole(pindex, pts, voronoi):
    rindex = voronoi.point_region[pindex]
    reg = voronoi.regions[rindex]
    if -1 in reg:
        return single_pole(pindex, pts, voronoi)
    else:
        return double_pole(pindex, pts, voronoi)

def find_neighbours(pindex, delaunay):
    return delaunay.vertex_neighbor_vertices[1][delaunay.vertex_neighbor_vertices[0][pindex]:delaunay.vertex_neighbor_vertices[0][pindex+1]]

def pt_at_angle(theta, x0, y0, r):
    a=r
    b=r
    
    x = a*np.cos(theta) + x0
    y = b*np.sin(theta) + y0
    z = 0.2
    return [x,y,z]

def triangulate(face):
    tris = []
    for i in range(1,len(face)-1):
        tris += [[face[0], face[i], face[i+1]]]
    #tris += [[face[0],face[-1],face[1]]]
    return tris

def delaunay_tri_2_voronoi_edge(triangle,ridge_dict):
    '''
    returns a vornoi edge with vertices indexed by voronoi vertices
    '''
    edges = [i for i in it.combinations(triangle,2)]
    dual_voro_faces = [ridge_dict[tuple(sorted(i))] for i in edges]
    return np.intersect1d(dual_voro_faces[0], dual_voro_faces[1], dual_voro_faces[2])
def kmeans(data, k, max_iters=100):
    # 初始化簇中心，随机选择 k 个数据点作为初始簇中心
    centroids = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(max_iters):
        # 分配每个数据点到最近的簇
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # 更新簇中心为每个簇的平均值
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 如果新簇中心与旧簇中心相同，算法收敛
        if np.array_equal(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def distance_to_centers(data, centers):
    # 计算每个数据点到簇中心的距离
    return np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

def measure_quality(data, labels, centers):
    # 计算整个点云的衡量值
    total_distance = np.sum(distance_to_centers(data, centers))
    num_clusters = len(centers)
    return total_distance / num_clusters

def incremental_kmeans(data, initial_k, max_iters=100):
    current_k = initial_k
    centers, labels = kmeans(data, initial_k, max_iters)
    
    while True:
        new_k = current_k + current_k // 2  # 增加 M/4, M/8, ... 个中心
        new_centers, new_labels = kmeans(data, new_k, max_iters)
        
        # 计算整个点云的衡量值
        current_quality = measure_quality(data, labels, centers)
        new_quality = measure_quality(data, new_labels, new_centers)
        
        # 如果质量提高，则更新中心和标签
        if new_quality < current_quality:
            current_k = new_k
            centers = new_centers
            labels = new_labels
        else:
            break
    
    return centers, labels
class PairCenter:
    def __init__(self, a, b, number):
        self.a = tuple(a)  # 转换为不可变类型以便于哈希
        self.b = tuple(b)
        self.number = number

    def __eq__(self, other):
        # 两个 PairCenter 被认为相等的条件：两点相同，无论顺序
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __hash__(self):
        # 哈希值根据点的内容计算，无论顺序
        return hash(frozenset([self.a, self.b]))

def classfication(center2,center_label1,poles1):
    pole_label=[]
    for i in range(len(poles1)):
        distances = distance.cdist([poles1[i]], center2)[0] 
        result_index=[] # 计算tt[i]到tt中所有点的距离
        if np.any(np.all(center2 == poles1[i], axis=1)):
    # If it is, set the distance to infinity
            distances[np.all(center2 == poles1[i], axis=1)] = np.inf
            result_index = np.argmin(distances) # Index of the second smallest distance
        else:
    # If it is not, find the index of the minimum distance
            result_index = np.argmin(distances)
        pole_label.append(center_label1[result_index])
       
    return pole_label
def figjj(center,vert,all_pairs):
    pole_label=[]
    for i in range(len(center)):
        pole_label.append(i)
    # line=[]
    # for pair in all_pairs:
    #     a_number = np.where((np.array(pair.a )== vert).all(axis=1))[0]
    #     b_number =np.where((np.array(pair.b) == vert).all(axis=1))[0]
    #     line.append([a_number[0],b_number[0]])
    
    ps.set_program_name("important app")
    ps.set_verbosity(0)
    ps.set_use_prefs_file(False)

    # initialize
    ps.init()
    sphere_mesh=trimesh.load("./output3/octopus_pc_inner_selected_sphere.obj")
 
    ps_mesh = ps.register_surface_mesh("my mesh", sphere_mesh.vertices, sphere_mesh.faces)
    
    array=np.random.rand(100, 3)
    class_pole=classfication(center,pole_label,vert)
    for i in range(len(center)):
        label_tt=np.where(np.array(class_pole) == i)[0]
        center_temp=np.array(vert)[label_tt]
        
        ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp)
        
            


        vals = np.tile(array[i], (len(center_temp), 1))
        ps_cloud.add_color_quantity("rand colors", vals)
        ps_cloud.set_enabled(False)
      
  
       
        # # points = np.random.rand(100, 3)
  
    ps.show() 
    tt=1
      
       
        # # # # visualize!


    # faces=np.array(faces)
    # lines=np.array(lines)
    # A,B=lines[0][0],lines[0][1]
    # pole_label=[]
    # for i in range(len(center)):
    #     pole_label.append(i)

    
    # ps.init
    # # for i in range(len(center)):

    # graph = {}
    # temp_lines=[]
    # for pair in all_pairs:
    #         a_number = np.where((pair.a == vert).all(axis=1))[0]
    #         b_number =np.where((pair.b == vert).all(axis=1))[0]
    #         temp_lines.append((a_number[0],b_number[0]))
   
    # for v1, v2 in temp_lines:
    # # 将v2添加到v1的邻接列表中
    #     if v1 in graph:
    #         graph[v1].append(v2)
    #     else:
    #         graph[v1] = [v2]

    #     # 将v1添加到v2的邻接列表中
    #     if v2 in graph:
    #         graph[v2].append(v1)
    #     else:
    #         graph[v2] = [v1]
    # label_tt1=np.where(np.array(class_pole) == int(A))[0]
        
        
    # center_temp1=np.array(vert)[label_tt1]
    # label_tt2=np.where(np.array(class_pole) == int(B))[0]
    # center_temp2=np.array(vert)[label_tt2]
    # temp_lines1=[]
    # combined_array = np.concatenate((center_temp1, center_temp2))
   

    # for k in range(len(center_temp1)):
    #     for j in range(len(center_temp2)):
    #         flg=0
        
    #         a= np.where((center_temp1[k] == vert).all(axis=1))[0][0]
    #         b= np.where((center_temp2[j]== vert).all(axis=1))[0][0]
            
    #         if a in graph:

    #             connected_vertices = graph[a]
    #             if b in connected_vertices:
    #                 flg=1
    #         elif b in graph:
    #                 connected_vertices = graph[b]  
    #                 if a in   connected_vertices:
    #                     flg=2
                    
    #         if flg == 1 or flg == 2:
    #                 temp_p = PairCenter(center_temp1[k], center_temp2[j], -1)
    #                 if temp_p in temp_lines1:
    #                     print("Found!", temp_p.a, temp_p.b)
    #                     continue
    #                 else:
    #                     temp_p.number = 0
    #                     temp_lines1.append(temp_p)
        
    # # temp_lines1= [list(x) for x in set(tuple(x) for x in temp_lines1)]
    # cclines=[]
    # for pair in temp_lines1:
    #     a_number = np.where((np.array(pair.a )== combined_array).all(axis=1))[0]
    #     b_number =np.where((np.array(pair.b) == combined_array).all(axis=1))[0]
    #     cclines.append([a_number[0],b_number[0]])
        
    # faces,lines=find_faces_and_lines(cclines)
    # save_obj("pole_1.obj",center_temp1)
    # save_obj("pole_2.obj",center_temp2)
    # with open("./output3/pole_final.obj", 'w') as out2:
    #     out2.write(f"{len(combined_array)} {len(temppair)} {0}\n")
    #     for iii in range(len(combined_array)):
    #         out2.write(f"v {combined_array[iii][0]} {combined_array[iii][1]} {combined_array[iii][2]}\n")

    #     for pair in temp_lines1:
    #         a_number = np.where((np.array(pair.a )== combined_array).all(axis=1))[0]
    #         b_number =np.where((np.array(pair.b) == combined_array).all(axis=1))[0]
    #         cclines.append([a_number[0],b_number[0]])
    #         out2.write(f"l {a_number[0]+1} {b_number[0]+1}\n") 
    
    # return A,B
def batch_cdist(vert, center, radii, batch_size=1000):
    n_vert = len(vert)
    nearest_center_indices = []

    for start in range(0, n_vert, batch_size):
        end = min(start + batch_size, n_vert)
        
        # Compute distances for the current batch
        dist_batch = distance.cdist(vert[start:end], center)
        
        # Subtract the squared radii
        radii_squared = (radii).reshape(-1, 1).T
        dist_batch -= radii_squared
        
        # Find the nearest center for each point
        nearest_center_indices_batch = np.argmin(np.abs(dist_batch), axis=1)
        nearest_center_indices.append(nearest_center_indices_batch)
    
    # Concatenate all the batch results into a single array
    return np.concatenate(nearest_center_indices)

def connect_method3(center,radii,all_pairs,vert):
    vert_list = list(vert)

# Convert vert_list to a 2D numpy array
   
    vert = np.array(vert_list, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    radii=np.array(radii, dtype=np.float32)
    nearest_center_indices = batch_cdist(vert, center, radii, batch_size=int(len(vert) / 10))
    point_to_cloud = {tuple(vert[i]): nearest_center_indices[i] for i in range(len(vert))}

# 用于存储点集之间的连接关系
    connected_clouds = set()

    # 遍历 all_pairs，找到对应的点集索引，并记录连接关系
    for pair in all_pairs:
        idx1, idx2 = pair
    # 获取每一对点
        point1 = vert[idx1]  # Get the point corresponding to index idx1
        point2 = vert[idx2]  # Get the point corresponding to index idx2

        # 获取这些点属于的点集索引
        cloud1 = point_to_cloud[tuple(point1)]  # Map point1 to its cloud (using tuple if necessary)
        cloud2 = point_to_cloud[tuple(point2)]  # Map point2 to its cloud (using tuple if necessary)

        # Do something with cloud1 and cloud2
        if cloud1!=cloud2:
            connected_clouds.add((min(cloud1, cloud2), max(cloud1, cloud2)))

    # 转换为列表或其他形式
    connected_clouds = list(connected_clouds)
    return connected_clouds


def connect_method1(center,vert,all_pairs):
    pole_label=[]
    for i in range(len(center)):
        pole_label.append(i)

    class_pole=classfication(center,pole_label,vert)
    graph = {}
    temp_lines=[]
    for pair in all_pairs:
            a_number = np.where((pair.a == vert).all(axis=1))[0]
            b_number =np.where((pair.b == vert).all(axis=1))[0]
            temp_lines.append((a_number[0],b_number[0]))
   
   
    temp_lines= [list(x) for x in set(tuple(x) for x in temp_lines)]
    

    


    for v1, v2 in temp_lines:
    # 将v2添加到v1的邻接列表中
        if v1 in graph:
            graph[v1].append(v2)
        else:
            graph[v1] = [v2]

        # 将v1添加到v2的邻接列表中
        if v2 in graph:
            graph[v2].append(v1)
        else:
            graph[v2] = [v1]
   
    all_temp_pairs=[]
   
    for i in range(len(center)):
        distances = distance.cdist([center[i]], center)[0] 
        distances[i] = np.inf 
        label_tt=np.where(np.array(class_pole) == i)[0]
        
        
        
        center_temp=np.array(vert)[label_tt]
        if len(center_temp)==0:
            center_temp=np.array(center)[i]
            center_temp=center_temp.reshape((1, 3))
        #  save_obj("./output3/%s_pc_class_pole.obj"%i, np.array(center_temp))
        nearest_index = np.argpartition(distances, 8)[:8]
        #  for j in range(len(nearest_index)):
            
        #      label_tt=np.where(np.array(class_pole) == nearest_index[j])[0]
        #      pole_temp=np.array(tt)[label_tt]
        #      save_obj("./output3/%s_pc_class_pole.obj"%nearest_index[j], np.array(pole_temp))
        for j in range(len(nearest_index)):
            temp_p = PairCenter(center[i], center[nearest_index[j]], -1)
            temp_b = PairCenter(center[nearest_index[j]],center[i], -1)
            if  temp_p in all_temp_pairs or temp_b in all_temp_pairs :
                print("Found!", temp_p.a, temp_p.b)
                break
            label_tt=np.where(np.array(class_pole) == nearest_index[j])[0]
            pole_temp=np.array(vert)[label_tt]
            if len(pole_temp)==0:
                pole_temp=np.array(center)[nearest_index[j]]
                pole_temp=pole_temp.reshape((1,3))
            if (len(pole_temp)==1 or len(center_temp)==1):
                if len(pole_temp)==1:
                    for pp in center_temp:
                       
                        flg=0
                        a=np.where((pole_temp[0] == vert).all(axis=1))[0][0]
                        b= np.where((pp== vert).all(axis=1))[0][0]
                        
                        if a in graph:
      
                                connected_vertices = graph[a]
                                if b in connected_vertices:
                                    flg=1
                        elif b in graph:
                                    connected_vertices = graph[b]  
                                    if a in   connected_vertices:
                                        flg=2
                            
                        if flg == 1 or flg == 2:
                                temp_p = PairCenter(center[i], center[nearest_index[j]], -1)
                                if temp_p in all_temp_pairs:
                                    print("Found!", temp_p.a, temp_p.b)
                                    continue
                                else:
                                    temp_p.number = 0
                                    all_temp_pairs.append(temp_p)
                                break
                        
                                        
                else:
                        for pp in pole_temp:
                            
                            flg=0
                            a=np.where((center_temp[0] == vert).all(axis=1))[0][0]
                            b= np.where((pp== vert).all(axis=1))[0][0]
                           
                            if a in graph:
      
                                connected_vertices = graph[a]
                                if b in connected_vertices:
                                    flg=1
                            elif b in graph:
                                    connected_vertices = graph[b]  
                                    if a in   connected_vertices:
                                        flg=2
                                
                            if flg == 1 or flg == 2:
                                    temp_p = PairCenter(center[i], center[nearest_index[j]], -1)
                                    if temp_p in all_temp_pairs:
                                        print("Found!", temp_p.a, temp_p.b)
                                        continue
                                    else:
                                        temp_p.number = 0
                                        all_temp_pairs.append(temp_p)
                                    break
                           

                            
                    
                
            else:
            
                
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(center_temp)
                #  save_obj("./output3/%s_pc_class_pole.obj"%nearest_index[j], np.array(pole_temp))

        # 找到points1中每个点在points2中的最近邻
                distances, indices = nbrs.kneighbors(pole_temp)

        # 找到距离最近的十对点
                nearpoint=[]
                nearest_pairs = sorted(range(len(distances)), key=lambda i: distances[i])[:12]
                #  for k in nearest_pairs:
                #     nearpoint.append(pole_temp[k])
                #     nearpoint.append(center_temp[indices[k][0]])
                #  save_obj("./output3/%s_pc_near_point.obj"%nearest_index[j], np.array(nearpoint))
                flg=0

                for k in nearest_pairs:
                    
                    a= np.where((pole_temp[k] == vert).all(axis=1))[0][0]
                    b= np.where((center_temp[indices[k][0]]== vert).all(axis=1))[0][0]
                    
                    if a in graph:
      
                        connected_vertices = graph[a]
                        if b in connected_vertices:
                            flg=1
                    elif b in graph:
                            connected_vertices = graph[b]  
                            if a in   connected_vertices:
                                flg=2
                            
                    if flg == 1 or flg == 2:
                            temp_p = PairCenter(center[i], center[nearest_index[j]], -1)
                            if temp_p in all_temp_pairs:
                                print("Found!", temp_p.a, temp_p.b)
                                continue
                            else:
                                temp_p.number = 0
                                all_temp_pairs.append(temp_p)
                            break
   
 
                 
    return center,all_temp_pairs

def connect_method2(pts,inner_pole,kd_tree,normal,point_cloud):
    all_pairs=[]
    temp_one_pair=[]
    temp_point=[]
    voronoi = spa.Voronoi(pts)
    for ridge in voronoi.ridge_vertices:
        if -1 not in ridge:
        # ridge是一条边的两个顶点的索引
            for c in it.combinations(ridge, 2):
                point1 = voronoi.vertices[c[0]]  # 边的第一个顶点
                point2 = voronoi.vertices[c[1]]
                flg = 0
                a=np.array(point1)
                b=np.array(point2)
                if a in inner_pole or b in inner_pole:
                    if a in inner_pole:
                        if b in inner_pole:
                            flg = 1
                        else:
                            flg=3

                        
                    elif b in inner_pole:
                        if a in inner_pole:
                            flg = 2
                        else:
                            flg=4
                            
                
                if flg == 1 or flg == 2:
                    temp_p = PairCenter(a, b, -1)
                    if temp_p in all_pairs:
                        print("Found!", temp_p.a, temp_p.b)
                        continue
                    else:
                        temp_p.number = 0
                        all_pairs.append(temp_p)
                if flg==3 or flg==4:
                    
                   
                         temp_p = PairCenter(a, b, -1)
                         if temp_p in temp_one_pair:
                            print("Found!", temp_p.a, temp_p.b)
                            continue
                         else:
                            temp_point.append(a)
                            temp_point.append(b)
                           
                            temp_one_pair.append(temp_p)

    # temp_one_pair = [list(x) for x in set(tuple(x) for x in temp_one_pair)]
    temp_point = [list(x) for x in set(tuple(x) for x in temp_point)]
    temp_one_pair=np.array(temp_one_pair)
    temp_point=np.array(temp_point)
    pole_label2,inner_poles2,outer_poles2=label_pole(temp_point,kd_tree,normal,point_cloud) #pole label
    temp_lines=[]
    for pair in temp_one_pair:
            a_number = np.where((pair.a == temp_point).all(axis=1))[0]
            b_number =np.where((pair.b == temp_point).all(axis=1))[0]
            temp_lines.append((a_number[0],b_number[0]))
    temp_lines= [list(x) for x in set(tuple(x) for x in temp_lines)]
    
    graph = {}
    for v1, v2 in temp_lines:
    # 将v2添加到v1的邻接列表中
        if v1 in graph:
            graph[v1].append(v2)
        else:
            graph[v1] = [v2]

        # 将v1添加到v2的邻接列表中
        if v2 in graph:
            graph[v2].append(v1)
        else:
            graph[v2] = [v1]
    for i in range(len(outer_poles2)):
        connected_vertices = graph[outer_poles2[i]]
        if len(connected_vertices)>2:
            combinations = list(it.combinations(connected_vertices, 2))
            for iyyt in combinations:
                temp_p = PairCenter(temp_point[iyyt[0]], temp_point[iyyt[1]], -1)
                if temp_p in all_pairs:
                    print("Found!", temp_p.a, temp_p.b)
                    continue
                else:
                    temp_p.number = 0
                    all_pairs.append(temp_p)
    return all_pairs
def single_pole_gpu(pindex, pts, voronoi, device='cuda:0'):
    # 将输入数据移动到 GPU
    p = torch.tensor(pts[pindex], device=device)
    reg = voronoi.regions[voronoi.point_region[pindex]]
    if -1 in reg:
        reg = [i for i in reg if i != -1]

    reg_vertices = torch.tensor(voronoi.vertices[reg], device=device)
    distances = torch.norm(reg_vertices - p, dim=1)
    max_index = torch.argmax(distances)
    r = distances[max_index]

    return [[reg_vertices[max_index].cpu().numpy(), r.item(), pindex]]

def double_pole_gpu(pindex, pts, voronoi, device='cuda:0'):
    p = torch.tensor(pts[pindex], device=device)
    reg = voronoi.regions[voronoi.point_region[pindex]]
    if -1 in reg:
        reg = [i for i in reg if i != -1]

    reg_vertices = torch.tensor(voronoi.vertices[reg], device=device)

    # 第一个极点
    distances = torch.norm(reg_vertices - p, dim=1)
    max_index = torch.argmax(distances)
    p1 = reg_vertices[max_index]
    r1 = distances[max_index]

    # 方向向量
    v1 = p1 - p
    v1 = v1 / torch.norm(v1)

    # 计算角度并筛选点
    vectors = reg_vertices - p
    vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
    angles = torch.acos(torch.clamp(torch.sum(vectors * v1, dim=1), -1.0, 1.0))
    good_points_mask = angles > (torch.pi / 2)
    good_points = reg_vertices[good_points_mask]

    if good_points.size(0) > 0:
        good_distances = torch.norm(good_points - p, dim=1)
        max_index = torch.argmax(good_distances)
        p2 = good_points[max_index]
        r2 = good_distances[max_index]
        return [[p1.cpu().numpy(), r1.item(), pindex], [p2.cpu().numpy(), r2.item(), pindex]]
    else:
        return [[p1.cpu().numpy(), r1.item(), pindex]]

def pole_gpu(pindex, pts, voronoi, device='cuda:0'):
    reg = voronoi.regions[voronoi.point_region[pindex]]
    if -1 in reg:
        return single_pole_gpu(pindex, pts, voronoi, device=device)
    else:
        return double_pole_gpu(pindex, pts, voronoi, device=device)

def compute_pole_rad_gpu(pts, voronoi, device='cuda:0'):
    pole_rad = []
    for pindex in range(len(pts)):
        pole_rad.extend(pole_gpu(pindex, pts, voronoi, device=device))
    return pole_rad
def approx_medial_axis(pts, MAX, z_max, alpha, plot,normal,oppath,oppath1):
    
   
    voronoi = spa.Voronoi(pts[...,0:2])
    
    v_verts = voronoi.vertices
   
    pole_rad=compute_pole_rad_gpu(pts[...,0:2], voronoi)
    # pole_rad = [item for sublist in [pole(i, pts, voronoi) for i in range(len(pts))] for item in sublist]
    
    

    # 提取 poles、poles_radii 和 pole_site
    poles = np.array([i[0] for i in pole_rad])
    poles_radii = np.array([i[1] for i in pole_rad])
    pole_site = np.array([i[2] for i in pole_rad])

    # 去重 poles，保留最小半径对应的索引
    _, unique_indices = np.unique(poles, axis=0, return_index=True)
    unique_indices_sorted = sorted(unique_indices, key=lambda x: poles_radii[x])

    # 根据 unique_indices 提取最终的 poles, poles_radii, pole_site
    poles = poles[unique_indices_sorted]
    poles_radii = poles_radii[unique_indices_sorted]
    pole_site = pole_site[unique_indices_sorted]
    poles1 = np.hstack((poles, np.zeros((poles.shape[0], 1))))

   
    save_obj("flower_pc_poles.obj", poles1)

        
        

                
    all_pairs=[]
    point_cloud =o3d.geometry.PointCloud()
    
    
    point_cloud.points = o3d.utility.Vector3dVector(pts)

# 构建八叉树
    kd_tree =  o3d.geometry.KDTreeFlann(point_cloud)
    v_verts1 = np.hstack((v_verts, np.zeros((v_verts.shape[0], 1))))
    
   

    pole_label2,inner_poles2,outer_poles2=label_pole(v_verts1,kd_tree,normal,point_cloud)#voronoi vertices label
   
    inner_verts=np.array(v_verts)[inner_poles2]
    
    inner_verts = set(map(tuple, inner_verts))  # inner_verts 转为 tuple 集合
    all_pairs = set()  # 用集合存储 PairCenter
    inner_verts_index = {pt: idx for idx, pt in enumerate(inner_verts)}

    for ridge in voronoi.ridge_vertices:
        if -1 not in ridge:
            for c in it.combinations(ridge, 2):
                point1 = tuple(voronoi.vertices[c[0]])  # 转为 tuple
                point2 = tuple(voronoi.vertices[c[1]])  # 转为 tuple
                
                # 检查点是否在 inner_verts
                if point1 in inner_verts_index and point2 in inner_verts_index:
                    # 获取两个点在 inner_verts 中的索引
                    idx1 = inner_verts_index[point1]
                    idx2 = inner_verts_index[point2]

                    # 如果两点不是同一个点，则存储索引对
                    if idx1 != idx2:
                        all_pairs.add(frozenset([idx1, idx2]))
    
        
    pole_label1,inner_poles1,outer_poles1=label_pole(poles1,kd_tree,normal,point_cloud)#voronoi vertices label
    inner_poles=poles1[inner_poles1]
    temppair=connect_method3(inner_poles,np.array(poles_radii)[inner_poles1],all_pairs,v_verts1)
    line=[]
    with open("flowerinnerpole_final.obj", 'w') as out2:
            out2.write(f"{len(inner_poles)} {len(temppair)} {0}\n")
            for iii in range(len(inner_poles)):
                out2.write(f"v {inner_poles[iii][0]} {inner_poles[iii][1]} {inner_poles[iii][2]}\n")

            for pair in temppair:
            
                line.append([pair[0],pair[1]])
                out2.write(f"l {pair[0]+1} {pair[1]+1}\n")  
    #origin connect

    # for ridge in voronoi.ridge_vertices:
    #     if -1 not in ridge:
    #         for c in it.combinations(ridge, 2):
    #             point1 = tuple(voronoi.vertices[c[0]])  # 转为 tuple
    #             point2 = tuple(voronoi.vertices[c[1]])  # 转为 tuple
                
    #             # 检查点是否在 inner_verts
    #             is_a_inner = point1 in inner_verts
    #             is_b_inner = point2 in inner_verts

    #             if is_a_inner or is_b_inner:
    #                 flg = 1 if is_a_inner and is_b_inner else 2

    #                 if flg in (1, 2):
    #                     temp_p = PairCenter(point1, point2, -1)

    #                     # 如果 temp_p 不在集合中，添加进去
    #                     if temp_p not in all_pairs:
                           
    #                         temp_p.number = 0
    #                         all_pairs.add(temp_p)
    
    
    # cc_pair=[]
    # with open(oppath1, 'w') as out2:
    #     out2.write(f"{len(inner_verts)} {len(all_pairs)} {0}\n")
    #     for iii in range(len(inner_verts)):
    #         out2.write(f"v {inner_verts[iii][0]} {inner_verts[iii][1]} {inner_verts[iii][2]}\n")

    #     for pair in all_pairs:
    #         a_number = np.where((pair.a == inner_verts).all(axis=1))[0]
    #         b_number =np.where((pair.b == inner_verts).all(axis=1))[0]
    #         if len(a_number) ==0 or len(b_number) ==0:
    #             continue
    #         else:
    #             out2.write(f"l {a_number[0]+1} {b_number[0]+1}\n")
    #             cc_pair.append([a_number[0],b_number[0]])

    #     print("The number of edges:", len(all_pairs))
    #     print("Written to", oppath1)
    #     #vert connect
    # faces,lines=find_faces_and_lines(cc_pair)
# min_sphere2pts,min_pts2sphere=calculate_mat_error(pts,radius,faces,edges,sample_point)
# ave_pts=np.sum(min_pts2sphere)/len(min_pts2sphere)
# ave_sphere=np.sum(min_sphere2pts)/len(min_sphere2pts)
# print(np.max(min_pts2sphere))
# print(np.max(min_sphere2pts))
# print("ave_pts:",ave_pts)
# print("ave_sphere:",ave_sphere)
# print(len(min_sphere2pts))
    # save_skel_mesh(inner_verts,faces, lines, "./output3/voronoi_face_final.obj","./output3/vorono_edge_final.obj")
    
    # pole_radii=np.array(poles_radii)[inner_poles1]
    # tempair=connect_method3(inner_pole,pole_radii,all_pairs,inner_verts)
    # center,tempair=connect_method1(inner_pole,v_verts,all_pairs)
    # # tempair=connect_method2(pts,inner_pole,kd_tree,normal,point_cloud)
            
    # with open(oppath, 'w') as out2:
    #     out2.write(f"{len(inner_pole)} {len(tempair)} {0}\n")
    #     for iii in range(len(inner_pole)):
    #         out2.write(f"v {inner_pole[iii][0]} {inner_pole[iii][1]} {inner_pole[iii][2]}\n")

    #     for pair in tempair:
    #         a_number = np.where((pair.a == inner_pole).all(axis=1))[0]
    #         b_number =np.where((pair.b == inner_pole).all(axis=1))[0]
    #         if len(a_number) ==0 or len(b_number) ==0:
    #             continue
    #         else:
    #             out2.write(f"l {a_number[0]+1} {b_number[0]+1}\n")

    #     print("The number of edges:", len(tempair))
    #     print("Written to", oppath)
    # line=[]
    # for pair in tempair:
    #         a_number = np.where((pair.a == inner_pole).all(axis=1))[0]
    #         b_number =np.where((pair.b == inner_pole).all(axis=1))[0]
    #         if len(a_number) ==0 or len(b_number) ==0:
    #             continue
    #         else:
    #             line.append([a_number[0],b_number[0]])
    # graph={}
  
    # face,line=find_faces_and_lines(cc_pair)
    # face=np.array(face)
    # line=np.array(line)
    inner_verts=np.array(list(inner_verts))
    # save_skel_mesh(inner_verts,face, line, "./output3/face_final.obj", "./output3/edge_final.obj")
    # save_spheres("./output3/pc_inner_remianing_sphere.obj",np.array(inner_pole)[single_edge_vertices],np.array(inner_pole)[single_edge_vertices])
    # ps.init()
    # mesh=trimesh.load("./output3/pc_inner_remianing_sphere.obj")
    # ps_point=ps.register_point_cloud("pole",inner_pole)
    # ps_mesh=ps.register_surface_mesh("corner sphere",np.array(mesh.vertices),np.array(mesh.faces))
    # ps.show()
   
    inner_verts1 = np.hstack((inner_verts, np.zeros((inner_verts.shape[0], 1))))
    return poles1,poles_radii,pole_site,inner_verts1,all_pairs,v_verts1

def cal_voronoi(points):
    vor = spa.Voronoi(points)
    # 遍历每一个Voronoi区域
    pole=[]
    pole_radii=[]
    site=[]
    for i, region in enumerate(vor.point_region):
        
        if not -1 in vor.regions[region] and len(vor.regions[region]) > 0:
            # 获取区域的生成点和法向量
            point = vor.points[i]
           
            # 获取区域的所有顶点
            vertices = vor.vertices[vor.regions[region]]
        
            # 计算生成点到每一个顶点的距离
            distances = distance.cdist(vertices,[point])
        
            # 找出最远的顶点
            farthest_vertex = vertices[np.argmax(distances)]
        
            vector_1 = farthest_vertex - point
            # 将最远的顶点添加到poles列表中，并记录其半径
            pole.append(farthest_vertex)
            pole_radii.append(np.argmax(distances))
            site.append(point)
        
            
            # 计算每个顶点到生成点的向量
                # 将这个顶点添加到inner poles列表中，并记录其半径
            vectors = vertices - point
        
            
            # 计算每个向量和法向量的点积
            dot_products = np.einsum('ij,ij->i', vectors, np.tile(vector_1, (len(vectors), 1)))
        
            
            inner_distances = distances[dot_products < 0]
           
               
            second_pole = vertices[np.argmax(inner_distances)]
            pole.append(second_pole)
            pole_radii.append(np.argmax(inner_distances))
            site.append(point)
    return pole,pole_radii,site
          
def cal_radii(pole,point1,point2,radius):
    vector1 = point1 - pole
    vector2 = point2 - pole

# Calculate the dot product of the vectors
    dot_product = np.dot(vector1, vector2)

# Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

# Calculate the angle in radians
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

# Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    return radius*np.sin(angle_degrees/2)

def cluster_and_classification(poles,poles_radii,site,normal,points,LAMB):
    point_cloud =o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector2dVector(points)

# 构建八叉树
    
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    PC=[]
    PC_radii=[]
    pole_label=[]
    sample=[]
    sample_value=[]
    for i in range(len(poles)):
        indices = pcd_tree.search_radius(poles[i], poles_radii[i])
        distances = distance.cdist(point_cloud[indices],[poles[i]])
        sorted_indices = np.argsort(distances)[::-1]

# Get the indices of the two farthest points
        farthest_indices = sorted_indices[:2]
        ras=cal_radii(poles[i],point_cloud[farthest_indices[0]],point_cloud[farthest_indices[1]],poles_radii[i])
        if(ras)<LAMB:
            index = indices[np.argmin(distances)]
            cc=np.dot(poles[i]-point_cloud[index])*normal[index]
            if cc<0:
                sample.append((poles[i]-site[i])/2)
                sample_value.append(-dist(poles[i]-site[i])/2)
            else:
                sample.append((poles[i]-site[i])/2)
                sample_value.append(dist(poles[i]-site[i])/2)

            continue
        else:
            index = indices[np.argmin(distances)]
            cc=np.dot(poles[i]-point_cloud[index])*normal[index]
            if cc<0:
                PC.append(poles[i])
                pole_label.append(IN)
                PC_radii.append(dist(poles[i]-site[i]))
                sample.append((poles[i]-site[i])/2)
                sample_value.append(-dist(poles[i]-site[i])/2)
            else:
                PC.append(poles[i])
                pole_label.append(OUT)
                PC_radii.append(dist(poles[i]-site[i]))
                sample.append((poles[i]-site[i])/2)
                sample_value.append(dist(poles[i]-site[i])/2)
    return  PC,PC_radii,pole_label,sample,sample_value

def calculate_overlap_volume(center1, radius1, center2, radius2):
    # Calculate the distance between the centers of the spheres
    d = dist(center1,center2)
    if d >=(radius1+radius2): 
        return 0
    elif d+radius1<=radius2:
        return (4.0*math.pi*radius1*radius1*radius1)/3.0
    elif d+radius2<=radius1:
        return (4.0*math.pi*radius2*radius2*radius2)/3.0
    else:
    # Calculate the height of the spherical cap
        h1 = radius1-(radius1*radius1+d*d-radius2*radius2)/(2*d)
        h2 = radius2-(radius2*radius2+d*d-radius1*radius1)/(2*d)

        # Calculate the volume of the spherical cap
        volume = math.pi*h1*h1*(radius1-h1/3)+math.pi*h2*h2*(radius2-h2/3)

        return volume

def greedy_selection(PC,PC_radii,rho,N):
#     point_cloud =o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(PC)

# # 构建八叉树
#     center=[]
#     radius=[]
#     pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
#     tt=np.sort(PC_radii)
#     for i in range(len(PC)):
#         indices=pcd_tree.search_radius(PC[i], PC_radii[i])
#         if()
    sorted_indices = np.argsort(PC_radii)[::-1]
    poles = PC[sorted_indices]
    radii = PC_radii[sorted_indices]
    
    # Initialize the list of qualified poles
    qualified_poles = []
    qualified_poles_radii = []
    # Set the threshold of the overlapping rate


    # Select the pole with the largest radius
    p = poles[0]
    r_p = radii[0]
    qualified_poles.append(p)
    qualified_poles_radii.append(r_p)
   
    point_cloud =o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PC)

# 构建八叉树
    pcd_tree =  o3d.geometry.KDTreeFlann(point_cloud)
    bad_pole=[]
    bad_pole.append(p)
    
    # Disqualify the poles which maximal sphere intersect deeply the maximal sphere of the selected pole
    while len(qualified_poles) <=N:
        v1=qualified_poles[len(qualified_poles)-1]
        r_v =qualified_poles_radii[len(qualified_poles)-1]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(query=v1, radius=r_v)
        
        for  v in idx:
             
          
            if  len(np.intersect1d(PC[v], bad_pole, assume_unique = True)) != 0 :
                continue
                         
            else :
                
                overlap_volume = calculate_overlap_volume(v1, r_v, PC[v], PC_radii[v])
                
                if overlap_volume >rho * (4/3) * np.pi * r_v**3:
                    bad_pole.append(PC[v])        
       
        for j in range(len(poles)):
            
            if len(np.intersect1d(poles[j], bad_pole, assume_unique = True)) == 0 :
                p = poles[j]
                r_p = radii[j]
                qualified_poles.append(p)
                qualified_poles_radii.append(r_p)
                bad_pole.append(p)
    
                break

    qualified_poles = np.array(qualified_poles)
    qualified_poles_radii=np.array(qualified_poles_radii)
    return qualified_poles,qualified_poles_radii


    
        
       
def approx_medial_axis2(file, MAX, z_max, alpha, LAMBDA, exp_tris):
    with open(file) as f:
        pts,colours = json.load(f)
    pts = np.array(pts)
    
    voronoi = spa.Voronoi(pts)
    print("done voronoi")
    delaunay = spa.Delaunay(pts)
    print("done delaunay")
    
    dtris = set([])
    for i in delaunay.simplices:
        dtris.update([ j for j in it.combinations(i,3) if circumsphere_3d(pts[i])[1] < alpha])
    dtris = np.array([list(i) for i in dtris], dtype=np.int32)
    print("done alpha")
    
    '''
    new_dict = {tuple(sorted(tuple(i))):voronoi.ridge_dict[tuple(i)] for i in voronoi.ridge_points}
    inv_dict = {tuple(new_dict[i]):list(i) for i in new_dict}
    '''
    bad_v = []
    if MAX:
        bad_v += [i for i,j in enumerate(voronoi.vertices) if dist(j,[0,0,0])>MAX]
    if z_max:
        bad_v += [i for i,j in enumerate(voronoi.vertices) if (np.abs(j[2]) > z_max)]
    
    bad_v = list(set(bad_v))
    
    faces = []
    for i in voronoi.ridge_dict:
        if -1 in voronoi.ridge_dict[i]:
            continue
        elif dist(pts[i[0]], pts[i[1]]) > LAMBDA:
            if len(np.intersect1d(voronoi.ridge_dict[i], bad_v, assume_unique = True)) == 0:
                if exp_tris:
                    faces += triangulate(voronoi.ridge_dict[i])
                else:
                    faces += [voronoi.ridge_dict[i]] 
    print("done lambda + limit")
    
   
    
    return [pts, dtris, voronoi.vertices, faces]        

def calculate_surface_error(centers, center_radii, points):
    total_error = 0

    # Iterate through each center
    for i in range(len(centers)):
        min_distance = float('inf')

        # Iterate through each point
        for j in range(len(points)):
            # Calculate the absolute difference between the distance from the point to the center
            # and the corresponding center radius
            current_distance = np.abs(dist(points[j], centers[i]) - center_radii[i])

            # Update the minimum distance if the current distance is smaller
            if current_distance < min_distance:
                min_distance = current_distance

        # Add the minimum distance to the total error
        total_error += min_distance

    return total_error
def calculate_surface_error(centers, center_radii, points):
    total_error = 0

    for center, radius in zip(centers, center_radii):
        distances = np.abs(distance.cdist([center], points) - radius)
        total_error += np.min(distances)

    return total_error

def clamp(a, b, c):
    """
    :param a:
    :param b:
    :param c:
    :return: max num
    """
    if a < b:
        return b
    else:
        if a > c:
            return c
        else:
            return a

neighbornum = 20

MIN_VALUE = 0

EPSILON = 1e-6


def normalize(v):
    norm = np.linalg.norm(v, axis=3, keepdims=True)
    v = v / norm
    return v

def length(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def Approximately(a, b):
    return a - b < EPSILON


def determinant(matrix):
    return round(np.linalg.det(matrix))


def IsCoplanar(A, B, C, P):
    normal1 = np.cross((B - A), (C - A))
    normal1 = normalize(normal1)

    normal2 = np.cross((P - A), (C - A))
    normal2 = normalize(normal2)
    if (np.abs(np.dot(normal1, normal2)) - 1 <= 1e-12):
        return True
    else:
        return False


def IS_FLOAT_ZERO(g):
    """
    是否等于0
    :param :if f=0,return true
    :return:
    """
    if g.item() - 0.00000001 < EPSILON:
        return True
    else:
        return False


def SameSide(p1, p2, a, b):
    """
    是否都在同一侧
    """
    cp1 = np.cross(b - a, p1 - a)
    cp2 = np.cross(b - a, p2 - a)

    if np.dot(cp1, cp2) >= 1e-12:
        return True
    else:
        return False


def IsInsideAtrianle(v0, v1, v2, p):
    """
    点是否在一个三角形内部
    """
    if SameSide(p, v0, v1, v2) and SameSide(p, v1, v0, v2) and SameSide(p, v2, v0, v1):
        return True
    else:
        return False


def LerpCone(v1, v2, t):
    """
    cone内部插值的球
    :param v1: sphere v1
    :param v2: sphere v2
    :param t: float
    :return: return a new sphere between the cone
    """
    return v1 * t + v2 * (1 - t)


def LerpSlab(v1, v2, v3, t1, t2):
    return v1 * t1 + v2 * t2 + v3 * (1.0 - t1 - t2)

def SphereToConeNearestSphere(m, m1, m2):
    c1 = m1[:3]
    c2 = m2[:3]
    r1 = m1[3]
    r2 = m2[3]

    inversed = False
    if r1 > r2:
        c1, c2 = c2, c1
        r1, r2 = r2, r1
        inversed = True

    cq = m[:3]
    rq = m[3]

    c21 = c1 - c2
    cq2 = c2 - cq
    A = np.dot(c21, c21)
    D = 2 * np.dot(c21, cq2)
    F = np.dot(cq2, cq2)
    R1 = r1 - r2
    t = -(A * D - R1 * R1 * D) - np.sqrt(((D * D - 4 * A * F) * (R1 * R1 - A) * R1 * R1))

    if A * A - A * R1 * R1 < 0:
        t = 0.5
    else:
        t /= 2 * ((A * A - A * R1 * R1))
        t = np.clip(t, 0, 1)

    if inversed:
        t = 1 - t

    return t

def sphere2Slabnearestsphere(m, m1, m2, m3):
    c11 = m1
    c12 = m2
    c13 = m3

    inversed1 = False
    inversed2 = False

    if c11[3] > c13[3]:
        c11, c13 = c13, c11
        inversed1 = True

    if c12[3] > c13[3]:
        c12, c13 = c13, c12
        inversed2 = True

    c13c11 = (c11[:3] - c13[:3])
    c13c12 = (c12[:3] - c13[:3])
    cqc13 = (c13[:3] - m[:3])

    R1 = c11[3] - c13[3]
    R2 = c12[3] - c13[3]
    A = np.dot(c13c11, c13c11)
    B = 2.0 * np.dot(c13c11, c13c12)
    C = np.dot(c13c12, c13c12)
    D = 2.0 * np.dot(c13c11, cqc13)
    E = 2.0 * np.dot(c13c12, cqc13)
    F = np.dot(cqc13, cqc13)

    if R1 == 0 and R2 == 0:
        t1 = (B * E - 2.0 * C * D) / (4.0 * A * C - B * B + 1e-8)
        t2 = (B * D - 2.0 * A * E) / (4.0 * A * C - B * B + 1e-8)
    elif R1 != 0 and R2 == 0:
        H2 = -B / (2.0 * (C + 1e-8))
        K2 = -E / (2.0 * (C + 1e-8))
        W1 = np.power(2.0 * A + B * H2, 2) - 4.0 * R1 * R1 * (A + B * H2 + C * H2 * H2)
        W2 = 2.0 * (2.0 * A + B * H2) * (B * K2 + D) - 4.0 * R1 * R1 * (B * K2 + 2.0 * C * H2 * K2 + D + E * H2)
        W3 = np.power(B * K2 + D, 2) - 4.0 * R1 * R1 * (C * K2 * K2 + E * K2 + F)
        if W2 * W2 - 4.0 * W1 * W3 < 0:
            t1 = 1.0
        else:
            t1 = (-W2 - np.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1 + 1e-8)
        t2 = H2 * t1 + K2
    elif R1 == 0 and R2 != 0:
        H1 = -B / (2.0 * A + 1e-8)
        K1 = -D / (2.0 * A + 1e-8)
        W1 = np.power(2.0 * C + B * H1, 2) - 4.0 * R2 * R2 * (C + B * H1 + A * H1 * H1)
        W2 = 2.0 * (2.0 * C + B * H1) * (B * K1 + E) - 4.0 * R2 * R2 * (B * K1 + 2.0 * A * H1 * K1 + E + D * H1)
        W3 = np.power(B * K1 + E, 2) - 4.0 * R2 * R2 * (A * K1 * K1 + D * K1 + F)
        if W2 * W2 - 4.0 * W1 * W3 < 0:
            t2 = 1
        else:
            t2 = (-W2 - np.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1 + 1e-8)
        t1 = H1 * t2 + K1
    else:
        L1 = 2.0 * A * R2 - B * R1
        L2 = 2.0 * C * R1 - B * R2
        L3 = E * R1 - D * R2
        if L1 == 0 and L2 != 0:
            t2 = -L3 / (L2 + 1e-5)
            W1 = 4.0 * A * A - 4.0 * R1 * R1 * A
            W2 = 4.0 * A * (B * t2 + D) - 4.0 * R1 * R1 * (B * t2 + D)
            W3 = np.power(B * t2 + D, 2) - (C * t2 * t2 + E * t2 + F)
            if W2 * W2 - 4.0 * W1 * W3 < 0:
                t1 = 1
            else:
                t1 = (-W2 - np.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1 + 1e-5)
        elif L1 != 0 and L2 == 0:
            t1 = L3 / (L1 + 1e-5)
            W1 = 4.0 * C * C - 4.0 * R2 * R2 * C
            W2 = 4.0 * C * (B * t1 + E) - 4.0 * R2 * R2 * (B * t1 + E)
            W3 = np.power(B * t1 + E, 2) - (A * t1 * t1 + D * t1 + F)
            if W2 * W2 - 4.0 * W1 * W3 < 0:
                t2 = 1
            else:
                t2 = (-W2 - np.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1 + 1e-5)
        else:
            H3 = L2 / (L1 + 1e-5)
            K3 = L3 / (L1 + 1e-5)
            W1 = np.power((2.0 * C + B * H3), 2) - 4.0 * R2 * R2 * (A * H3 * H3 + B * H3 + C)
            W2 = 2.0 * (2.0 * C + B * H3) * (B * K3 + E) - 4.0 * R2 * R2 * (2.0 * A * H3 * K3 + B * K3 + D * H3 + E)
            W3 = np.power((B * K3 + E), 2) - 4.0 * R2 * R2 * (A * K3 * K3 + D * K3 + F)
            if W2 * W2 - 4.0 * W1 * W3 < 0:
                t2 = 1
            else:
                t2 = (-W2 - np.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1 + 1e-5)
            t1 = H3 * t2 + K3
    if (t1 + t2) < 1.0 and 0 <= t1 <= 1 and 0 <= t2 <= 1:
        mt = LerpSlab(c11, c12, c13, t1, t2)
        if inversed1 and not inversed2:
            t1 = 1 - t1 - t2
        elif not inversed1 and inversed2:
            t2 = 1 - t1 - t2
        elif inversed1 and inversed2:
            t1, t2 = t2, 1 - t1 - t2
        return mt
    else:
        t13 = SphereToConeNearestSphere(m, c11, c13)
        min_ball = LerpCone(c11, c13, t13)
        min_d = np.linalg.norm((min_ball[:3] - m[:3])) - (min_ball[3] + m[3])

        t2 = 0
        t1 = t13
        t23 = SphereToConeNearestSphere(m, c12, c13)
        min_ball = LerpCone(c12, c13, t23)
        d = np.linalg.norm((min_ball[:3] - m[:3])) - (min_ball[3] + m[3])

        if d < min_d:
            min_d = d
            t1 = 0
            t2 = t23
        t12 = SphereToConeNearestSphere(m, c11, c12)
        min_ball = LerpCone(c11, c12, t12)
        d = np.linalg.norm((min_ball[:3] - m[:3])) - (min_ball[3] + m[3])

        if d < min_d:
            min_d = d
            t1 = t12
            t2 = 1 - t12
        min_ball = LerpSlab(c11, c12, c13, t1, t2)
        return min_ball

def distance_to_surface_with_normal(center, center_radii, pts,normal):
    # 计算每个球距离其球面最近的点
    center_radii=center_radii.reshape(-1, 1)
    distances = np.linalg.norm(pts - center[:, np.newaxis], axis=2) - center_radii
    min_distances = np.min(distances, axis=1)
    ccdistance=[]
    tt=np.argmin(distances,axis=1)
   
    min_distances=np.power(min_distances,2)
    return min_distances

def distance_to_surface(center, center_radii, pts):
    # 计算每个球距离其球面最近的点
    center_radii=center_radii.reshape(-1, 1)
    distances = np.linalg.norm(pts - center[:, np.newaxis], axis=2) - center_radii
    min_distances = np.min(distances, axis=1)
    ccdistance=[]
    tt=np.argmin(distances,axis=1)
   
    # min_distances=np.power(min_distances,2)
    return min_distances
def calculate_mat_errorfordpc(center,center_radii,face,line,pts):
    min_sphere2pts=distance_to_surface(center,center_radii,pts)
    center_radii=center_radii.reshape(-1, 1)
   
    
  
    distance=np.linalg.norm(pts - center[:, np.newaxis], axis=2) - center_radii
    
    min_distances = np.min(distance, axis=0)
    # min_pts2sphere=np.power(min_distances,2)
    return min_sphere2pts,min_distances

def calculate_mat_error(center,center_radii,face,line,pts):
    samplepoint=pts[np.random.choice(np.arange(len(pts)),2000)]
    min_sphere2pts=distance_to_surface(center,center_radii,samplepoint)
    sphere = np.concatenate((center, center_radii.reshape(-1,1)), axis=1)
   
    pts_s=np.concatenate((samplepoint, np.zeros_like(np.ones((len(samplepoint), 1))).reshape(-1,1)), axis=1)
    min_pts2sphere=[]
    for i in range(len(pts_s)):
        mindis=1000000000
        for j in range(len(face)):
            minball=sphere2Slabnearestsphere(pts_s[i],sphere[face[j][0]],sphere[face[j][1]],sphere[face[j][2]])
            mindis_temp=np.linalg.norm((minball[:3] - pts_s[i][:3])) - (minball[3] + pts_s[i][3])
            if mindis_temp<mindis:
                mindis=mindis_temp
        for j in range(len(line)):
            t=SphereToConeNearestSphere(pts_s[i],sphere[line[j][0]],sphere[line[j][1]])
            min_ball = LerpCone(sphere[line[j][0]], sphere[line[j][1]], t)
            min_d = np.linalg.norm((min_ball[:3] - pts_s[i][:3])) - (min_ball[3] + pts_s[i][3])
            if min_d<mindis:
                mindis=min_d
        min_pts2sphere.append(mindis)
    min_pts2sphere=np.power(min_pts2sphere,2)
    return min_sphere2pts,min_pts2sphere
            
# center=np.random.rand(10,3)

# center_radii=np.random.rand(10,1)
# random_integers = np.random.randint(0, 10, 6)

# # 将它们组合成2*3的形式
# face = random_integers.reshape(2, 3)
# random_integers = np.random.randint(0, 10, 4)

# # 将它们组合成2*3的形式
# line = random_integers.reshape(2,2 )

# pts=np.random.rand(30,3)
# min_sphere2pts,min_pts2sphere=calculate_mat_error(center,center_radii,face,line,pts)
# def approx_medial_axis2(pts,normals, MAX, z_max, alpha, LAMBDA, exp_tris):
#     min_coords = np.min(pts, axis=0)
#     max_coords = np.max(pts, axis=0)
#     # normalize mesh

#     bbmin = pts.min(0)
#     bbmax = pts.max(0)
#     center = (bbmin + bbmax) * 0.5
    
#     voronoi = spa.Voronoi(pts)
#     print("done voronoi")
#     # delaunay = spa.Delaunay(pts)
#     # print("done delaunay")
    
#     # dtris = set([])
#     # for i in delaunay.simplices:
#     #     dtris.update([ j for j in it.combinations(i,3) if circumsphere_3d(pts[i])[1] < alpha])
#     # dtris = np.array([list(i) for i in dtris], dtype=np.int32)
#     # print("done alpha")
    
#     # '''
#     # new_dict = {tuple(sorted(tuple(i))):voronoi.ridge_dict[tuple(i)] for i in voronoi.ridge_points}
#     # inv_dict = {tuple(new_dict[i]):list(i) for i in new_dict}
#     # '''
#     # bad_v = []
#     # if MAX:
#     #     bad_v += [i for i,j in enumerate(voronoi.vertices) if dist(j,[0,0,0])>MAX]
#     # if z_max:
#     #     bad_v += [i for i,j in enumerate(voronoi.vertices) if (np.abs(j[2]) > z_max)]
    
#     # bad_v = list(set(bad_v))
    
#     # faces = []
#     # for i in voronoi.ridge_dict:
#     #     if -1 in voronoi.ridge_dict[i]:
#     #         continue
#     #     elif dist(pts[i[0]], pts[i[1]]) > LAMBDA:
#     #         if len(np.intersect1d(voronoi.ridge_dict[i], bad_v, assume_unique = True)) == 0:
#     #             if exp_tris:
#     #                 faces += triangulate(voronoi.ridge_dict[i])
#     #             else:
#     #                 faces += [voronoi.ridge_dict[i]] 
#     # print("done lambda + limit")
#     poles = []
    
#     pole_radii = []
    
#     sample_points = []
#     sample_values = []
#        # 遍历每一个Voronoi区域
#     for i, region in enumerate(voronoi.point_region):
        
#         if not -1 in voronoi.regions[region] and len(voronoi.regions[region]) > 0:
#             # 获取区域的生成点和法向量
#             point = voronoi.points[i]
            
#             vertices = voronoi.vertices[voronoi.regions[region]]
            
#             bad_v = []
#             good_v=[]
#             if MAX:
#                 distances = spa.distance.cdist(vertices, [center])
#                 bad_v += [y for x,y in enumerate(vertices) if (distances[x] > MAX)]
#                 good_v+=[y for x ,y in enumerate(vertices) if (distances[x] < MAX)]
#             if z_max:
#                 bad_v += [y for x,y in enumerate(vertices) if (np.abs(y[2]) > z_max)]
#                 good_v+=[y for x ,y in enumerate(vertices) if (np.abs(y[2]) < z_max)]
#             # bad_v = list(set(bad_v))
#             good_v=np.array(good_v)
           
#             tempv = []
           
#             for j in range(len(good_v)):
#                 if dist(point,good_v[j])>LAMBDA:
#                     if len(np.intersect1d(good_v[j], bad_v, assume_unique = True)) == 0:
#                         tempv.append(good_v[j])
#             # for j in voronoi.ridge_dict:
#             #     if -1 in voronoi.ridge_dict[j]:
#             #         continue
#             #     elif dist(pts[i[0]], pts[i[1]]) > LAMBDA:
#             #         if len(np.intersect1d(voronoi.ridge_dict[j], bad_v, assume_unique = True)) == 0:
#             #             if exp_tris:
#             #                 faces += triangulate(voronoi.ridge_dict[i])
#             #             else:
#             #                 faces += [voronoi.ridge_dict[i]] 
            
#                 # 获取区域的所有顶点
#                 # vertices = voronoi.vertices[voronoi.regions[region]]
            
#                 # 计算生成点到每一个顶点的距离
#             tempv=np.array(tempv)
#             if(len(tempv)<2):
#                 continue
#             else:
#                 distances = spa.distance.cdist(tempv,[point])
        
#             # 找出最远的顶点
#                 farthest_vertex = tempv[np.argmax(distances)]
            
#                 vector_1 = farthest_vertex - point
#                 # 将最远的顶点添加到poles列表中，并记录其半径
                
            
                
#                 # 计算每个顶点到生成点的向量
#                     # 将这个顶点添加到inner poles列表中，并记录其半径
#                 vectors = tempv - point
            
                
#                 # 计算每个向量和法向量的点积
#                 dot_products = np.einsum('ij,ij->i', vectors, np.tile(vector_1, (len(vectors), 1)))
           
               
#                 inner_distances = distances[dot_products < 0]
                

#                 second_pole = vertices[np.argmax(inner_distances)]
#                 poles.append(farthest_vertex)
#                 pole_radii.append(np.max(distances))

#                 poles.append(second_pole)
#                 pole_radii.append(np.max(inner_distances))
                    
                
                        
                   
    # '''
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    
    # plot_simp = a3.art3d.Poly3DCollection([ voronoi.vertices[i] for i in faces], alpha=0.1)
    # plot_simp.set_color('purple')
    # ax.add_collection3d(plot_simp)
    
    # ax.set_xlim3d(-4, 4)
    # ax.set_ylim3d(-4, 4)
    # ax.set_zlim3d(-4, 4)
    # '''
    
    # return poles,pole_radii 
def find_faces_and_lines(edges):
    # 创建一个字典来存储每个顶点连接的边
    graph = {}
    for edge in edges:
        for vertex in edge:
            if vertex not in graph:
                graph[vertex] = set()
            graph[vertex].add(edge[0] if vertex != edge[0] else edge[1])

    # 找出由三条边组成的面
    faces = []
    for vertex, connected_vertices in graph.items():
        for v1 in connected_vertices:
            for v2 in connected_vertices:
                if v1 != v2 and v2 in graph[v1]:
                    face = sorted([vertex, v1, v2])
                    if face not in faces:
                        faces.append(face)

    # 找出不属于任何面的边
    lines = [edge for edge in edges if not any(set(edge).issubset(face) for face in faces)]

    return faces, lines


def label_pole_no_kdtree_chunked(pole, normal, point_cloud, radius, chunk_size=1000, device='cuda'):
    """
    使用距离矩阵的分块实现，避免内存溢出，对 pole 点进行分类。
    """
    # 转换为 PyTorch 张量
    pole = torch.tensor(pole, dtype=torch.float32, device=device)
    normal = torch.tensor(normal, dtype=torch.float32, device=device)
    points = torch.tensor(point_cloud, dtype=torch.float32, device=device)

    # 初始化结果列表
    label_pole = []
    Inner_pole = []
    outer_pole = []

    # 分块处理查询点
    num_poles = pole.size(0)
    for start in range(0, num_poles, chunk_size):
        end = min(start + chunk_size, num_poles)
        current_pole = pole[start:end]

        # 计算当前块的距离矩阵
        distances = torch.cdist(current_pole, points)  # (chunk_size, N)

        # 筛选邻居点
        neighbor_mask = distances < radius  # (chunk_size, N)

        for i in range(current_pole.size(0)):
            # 获取邻居点索引
            neighbor_indices = torch.where(neighbor_mask[i])[0]

            if len(neighbor_indices) == 0:
                # 没有邻居点
                label_pole.append("OUT")
                outer_pole.append(start + i)
                continue

            # 获取邻居点坐标和法向量
            neighbor_points = points[neighbor_indices]
            neighbor_normals = normal[neighbor_indices]

            # 计算相对向量
            relative_vectors = current_pole[i] - neighbor_points  # (K, 3)

            # 检查所有邻居点的法向量点乘条件
            dot_products = torch.matmul(relative_vectors, neighbor_normals.T)  # (K, K)
            if torch.all(dot_products.diagonal() < 0):
                label_pole.append("IN")
                Inner_pole.append(start + i)
            else:
                label_pole.append("OUT")
                outer_pole.append(start + i)

    return label_pole, Inner_pole, outer_pole

def label_pole(pole, kd_tree, normal, point_cloud):
    """
    对 pole 点进行分类为内极点和外极点。

    参数：
    pole: 极点列表 (Nx3 array)
    kd_tree: 用于查询邻居点的 KD 树
    normal: 法向量列表 (Nx3 array)
    point_cloud: 点云对象，包含点的坐标

    返回：
    label_pole: 标记的极点类别列表 (IN/OUT)
    Inner_pole: 内极点的索引列表
    outer_pole: 外极点的索引列表
    """
    # 初始化结果列表
    label_pole = []
    Inner_pole = []
    outer_pole = []

    # 转换为 NumPy 数组以提高操作效率
    pole = np.array(pole)
    normal = np.array(normal)
    points = np.array(point_cloud.points)

    for i, p in enumerate(pole):
        # 搜索邻居点，返回最多 3 个点
        k, idx, _ = kd_tree.search_knn_vector_3d(p, 2)
        neighbor_points = points[idx]
        neighbor_normals = normal[idx]

        # 计算相对向量
        relative_vectors = p - neighbor_points

        # 检查所有邻居点的法向量点乘条件是否成立
        if np.all(np.dot(relative_vectors, neighbor_normals.T) < 0):
            label_pole.append("IN")
            Inner_pole.append(i)
        else:
            label_pole.append("OUT")
            outer_pole.append(i)

    return label_pole, Inner_pole, outer_pole




def pole_cluster_lambda(pole, pole_radii, kd_tree, point_cloud, lanb, alpha, site):
    remaining_pole = []
    remaining_pole_radii = []
    remaining_site = []
    point1=0
    point0=0
    point2=0
   
  
    for i, (current_pole, current_radius) in enumerate(zip(pole, pole_radii)):
        # 搜索在当前极点半径范围内的邻居
        k, idx, _ = kd_tree.search_radius_vector_3d(query=current_pole, radius=current_radius)
        
        # 获取邻居点到当前点的距离
        distances = np.linalg.norm(np.array(point_cloud.points)[idx] - current_pole, axis=1)

        # 确保包含边界点（距离刚好等于 current_radius 的点）
       
        k = len(idx)

        # 如果没有足够的邻居点，归入类别 0
        if k < 1:
            point0=point0+1          
            continue
        if k ==1:
            point1=point1+1          
            continue

        # 计算邻居点间的最大距离
        near_points = np.array(point_cloud.points)[idx]
        dis = distance.cdist(near_points, near_points)
        r = 0.5 * dis.max()

        # 根据条件分类
        if r / current_radius >= alpha and r >= lanb:
            remaining_pole.append(current_pole)
            remaining_pole_radii.append(current_radius)
            remaining_site.append(site[i])

    return remaining_pole,remaining_pole_radii, remaining_site
      
def batch_cdist_torch(x1, x2, batch_size, device='cuda'):
    """
    分批计算 torch.cdist
    x1: 第一组点 (N, D)
    x2: 第二组点 (M, D)
    batch_size: 分批大小
    """
    n = x1.shape[0]
    m = x2.shape[0]
    distances = []

    for i in range(0, n, batch_size):
        batch_x1 = x1[i:i + batch_size]
        batch_distances = torch.cdist(batch_x1, x2)
        distances.append(batch_distances)

    return torch.cat(distances, dim=0)
def pole_cluster_lambda_torch(pole, pole_radii, point_cloud, lanb, alpha, site, device='cuda'):
    """
    使用 PyTorch 优化 pole_cluster_lambda 函数
    """
    # 转换为 PyTorch 张量并移动到指定设备
    pole = torch.tensor(pole, dtype=torch.float32, device=device)
    pole_radii = torch.tensor(pole_radii, dtype=torch.float32, device=device)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32, device=device)
    site = torch.tensor(site, dtype=torch.int64, device=device)

    # 初始化结果列表
    remaining_pole = []
    remaining_pole_radii = []
    remaining_site = []

    # 计算每个极点到点云所有点的距离
    distances = batch_cdist_torch(pole, point_cloud, batch_size=2000, device=device)

    # 遍历每个极点
    for i, (current_pole, current_radius) in enumerate(zip(pole, pole_radii)):
        # 找到当前极点半径内的邻居点
        mask = distances[i] <= current_radius
        neighbor_points = point_cloud[mask]

        # 如果邻居点少于等于 1，跳过
        if neighbor_points.size(0) <= 1:
            continue

        # 计算邻居点之间的最大距离
        neighbor_distances = torch.cdist(neighbor_points, neighbor_points)
        r = 0.5 * torch.max(neighbor_distances)

        # 根据条件分类
        if r / current_radius >= alpha and r >= lanb:
            remaining_pole.append(current_pole)
            remaining_pole_radii.append(current_radius)
            remaining_site.append(site[i])

    # 将结果转换为 PyTorch 张量
    if remaining_pole:
        remaining_pole = torch.stack(remaining_pole)
        remaining_pole_radii = torch.stack(remaining_pole_radii)
        remaining_site = torch.stack(remaining_site)
    else:
        remaining_pole = torch.empty(0, 3, device=device)
        remaining_pole_radii = torch.empty(0, device=device)
        remaining_site = torch.empty(0, device=device)

    return remaining_pole, remaining_pole_radii, remaining_site   

def assign_labels(pole, pole_radii, center, center_radii):
    lables=[]
    
    for i in range (len(pole)):
        mindis=10000
        lable=1000000
        for j in range(len(center)):
            
            dis=np.dot(vect(center[j],pole[i]), vect(center[j],pole[i]))

        
            tt2=pow((center_radii[j]- pole_radii[i]),2)
            if dis<mindis:
                              
                mindis= dis
                lable=j

        lables.append(lable)

        
    # Calculate distances based on the specified distance metric


    # Assign labels based on the minimum distances
    
        
   
    return np.array(lables)

def update_centroids(pole,pole_radii,pole_volume, center,labels, k):
    newcenter=[]
    newradii=[]
    # Update centroids based on the mean of data points assigned to each cluster
    for i in range(k):
        tt=np.where(labels == i)

        # Calculate the weighted mean
        cluster_points = pole[tt]
        cluster_radii = pole_radii[tt]
        cluster_volume = pole_volume[tt]
        print(len(cluster_points))
        if len(cluster_points)==0:
            
            center_temp=center[i]
            
        else:
            total_weight=np.array([0,0,0])
            weighted_sum=0
            for j in range(len(cluster_points)):
                total_weight[0]+=cluster_points[j][0]*cluster_volume[j]/(cluster_radii[j]**5)
                total_weight[1]+=cluster_points[j][1]*cluster_volume[j]/(cluster_radii[j]**5)
                total_weight[2]+=cluster_points[j][2]*cluster_volume[j]/(cluster_radii[j]**5)
                weighted_sum+=cluster_volume[j] / (cluster_radii[j]**5)
            center_temp=total_weight/weighted_sum
            

            

            # Find the index of the closest pole to the updated centroid
        distances = distance.cdist(pole , [center_temp])
        closest_pole_index = np.argmin(distances)

        # Assign the coordinates and radius of the closest pole to the updated centroid
        newcenter.append(pole[closest_pole_index])
        newradii.append(pole_radii[closest_pole_index])

    return newcenter,newradii

def random_ini(pole,pole_radii,k):
    segment_lengths=[]
    total_length=0
    for i in range(len(pole_radii)):
        total_length+=(1.0)/(pole_radii[i]*pole_radii[i])
        segment_lengths.append((1.0)/(pole_radii[i]*pole_radii[i]))
    segment_lengths=np.array(segment_lengths/total_length)
    cumulative_lengths = np.cumsum(segment_lengths)


# 生成k个在(0, 1)之间的随机数，表示采样点的位置
    random_points = np.random.rand(k)

# 初始化一个数组，用于存储每个采样点属于哪个线段的索引
    point_indices = np.zeros(k, dtype=int)

# 确定每个采样点属于哪个线段
    for i in range(len(segment_lengths)):
        if i == 0:
            mask = (random_points >= 0) & (random_points <= cumulative_lengths[i])
        else:
            mask = (random_points > cumulative_lengths[i-1]) & (random_points <= cumulative_lengths[i])
        point_indices[mask] = i

  
    # total_number=0
    # for i in range(len(pole)):
    #     total_number+=1.0/(pole_radii[i]*pole_radii[i])
    # segment_probabilities = [(1.0/(length*length)) / total_number for length in pole_radii]
    # random_numbers = np.random.uniform(0, 1, size=k)
    # cumulative_probabilities = np.cumsum(segment_probabilities)
    # segment_indices = np.digitize(random_numbers*total_number, cumulative_probabilities)
    return point_indices
        
def cluster(pole,pole_radii,pole_volume,k,max_iters,real_name,kd_tree,normals,point_cloud):

    # centroids = random_ini(pole,pole_radii,k)
   
    # center=pole[centroids]
    # center_radii=pole_radii[centroids]
    center=[]
    save_spheres("./output3/%s_pc_ipole_sphere.obj"%real_name,pole,pole_radii)
    center_radii=[]
    for i in range(k):
        center.append(pole[i])
        center_radii.append(pole_radii[i])
    save_spheres("./output3/%s_pc_ini_sphere.obj"%real_name,center,center_radii)
    labels = assign_labels(pole,pole_radii ,center,center_radii)
   
    # ps.set_program_name("important app")
    # ps.set_verbosity(0)
    # ps.set_use_prefs_file(False)

    # # initialize
    # ps.init()
 
    # ps_mesh = ps.register_surface_mesh("my mesh", mesh_vertices, mesh_face)
    # ps_mesh1 = ps.register_surface_mesh("my mesh1", mesh_vertices1, mesh_face1)
   
    # for i in range(len(center)):
    #     label_tt=np.where(np.array(labels) == i)[0]
    #     center_temp=np.array(pole)[label_tt]
    #     if len(center_temp)<=1:
    #         ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp.reshape(-1,3))
            


    #         vals = np.tile(array[i], (len(center_temp), 1))
    #         ps_cloud.add_color_quantity("rand colors", vals)
    #         ps_cloud.set_radius(0.005)
  
       
    #     # # points = np.random.rand(100, 3)
    # ps_cloudq = ps.register_point_cloud("my points", pole)
    # ps.show() 
      
       
    #     # # # # visualize!
    #     ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp)
        


    #     vals = np.tile(array[i], (len(center_temp), 1))
    #     ps_cloud.add_color_quantity("rand colors", vals)
    #     ps_cloud.set_radius(0.005)
    # ps_cloudq = ps.register_point_cloud("my points", pole)  
    # ps.show()   
    #     save_obj("./output3/%s_pc_class_pole.obj"%i, np.array(center_temp))
    kmeans_3d = KMeans(n_clusters=k,max_iter=100)
    kmeans_3d.fit(pole)
    centroids_3d = kmeans_3d.cluster_centers_
    distances = distance.cdist(pole, centroids_3d, metric='euclidean')

#找到每个聚类中每个点到聚类中心的最小距离的索引
    closest_indices = np.argmin(distances, axis=0)
    

    
    # for _ in range(max_iters):
    #     # Assign each data point to the nearest centroid
    #     labels = assign_labels(pole,pole_radii ,center, center_radii)
        
    #     # Update centroids based on the mean of data points assigned to each cluster
       
        
    #     # Update centroids
    #     new_centroids,centroids_radii = update_centroids(pole,pole_radii,pole_volume,center, labels, k)
        
    #     # Check for convergence
    #     if np.array_equal(center, new_centroids):
    #         break
        
    #     center = new_centroids
    #     center_radii=centroids_radii
    #     distances = distance.cdist(pole, center, metric='euclidean')
    #     closest_indices = np.argmin(distances, axis=0)
    closest_indices = np.argmin(distances, axis=0)
    # save_spheres("./output3/%s_pc_inner_sphere.obj"%real_name,pole[closest_indices],pole_radii[closest_indices])
    labels1 = assign_labels(pole,pole_radii ,pole[closest_indices],pole_radii[closest_indices])
    # ps.set_program_name("important app")
    # ps.set_verbosity(0)
    # ps.set_use_prefs_file(False)

    # # initialize
    # ps.init()
    for i in range(len(pole[closest_indices])):
        label_tt=np.where(np.array(labels1) == i)[0]
        center_temp=np.array(pole)[label_tt]
       
        # # points = np.random.rand(100, 3)
     
      
       
        # # # # visualize!
        # if len(center_temp)<=1:
        #     ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp.reshape(-1,3))
            


        #     vals = np.tile(array[i], (len(center_temp), 1))
        #     ps_cloud.add_color_quantity("rand colors", vals)
        #     ps_cloud.set_radius(0.005)
    # ps_cloud1 = ps.register_point_cloud("my points1",pole)
    # ps_cloud2 = ps.register_point_cloud("my points",pole[closest_indices])
    pole_label1,inner_poles1,outer_poles1=label_pole(pole[closest_indices],kd_tree,normals,point_cloud)
    # ps_cloud = ps.register_point_cloud("my points2",np.array(pole[closest_indices])[inner_poles1])
    # ps.show()   
    save_spheres("./output3/%s_pc_ipole_sphere.obj"%real_name,pole[closest_indices][inner_poles1],pole_radii[closest_indices][inner_poles1])

    return labels, np.array(pole[closest_indices])[inner_poles1],np.array(pole_radii[closest_indices])[inner_poles1]
def choose_sphere(center,radii,pts,pole,pole_radii,K ):
   
    center_radii=radii.reshape(-1, 1)
    distances = np.linalg.norm(pts - center[:, np.newaxis], axis=2) - center_radii
    min_distances = np.min(distances, axis=0)
    max_k_indices = np.argsort(min_distances)[-K:]
    
    # 获取这些点的索引和坐标
    
    array=np.array(pts)[max_k_indices]
    distances_new = np.linalg.norm(array - np.array(pole)[:, np.newaxis], axis=2) - pole_radii
    min_distances = np.min(distances_new, axis=1)
    closest_indices = np.argmin(distances_new, axis=0)
    return closest_indices

    
def findnewcenter(centers,poles,lines,faces,pole_radii):
    point_cloud1 =o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(poles)

# 构建八叉树
    kd_tree_pole =  o3d.geometry.KDTreeFlann(point_cloud1)
    newcenters=[]
    newradii=[]
    for i in range(len(centers)):
        [k, idx, _] = kd_tree_pole.search_knn_vector_3d(centers[i],3)
    
    
        relevant_lines = [line for line in lines for j in idx if j in line]
        relevant_faces = [face for face in faces for j in idx if j in face]
        mindis=10000000
        center_temp=[]
        radii_temp=[]
        if len(relevant_lines)>0:
            for j in range(len(relevant_lines)):
                p,radii=project_point_onto_line(poles[relevant_lines[j][0]],poles[relevant_lines[j][1]],centers[i],pole_radii[relevant_lines[j][0]],pole_radii[relevant_lines[j][1]])
                dis=dist(p,centers[i])
                if dis<mindis:
                    center_temp=p
                    radii_temp=radii
                    
                    mindis=dis
        if len(relevant_faces)>0:
            for k in range(len(relevant_faces)):
                p,radii=project_point_onto_triangle(poles[relevant_faces[k][0]],poles[relevant_faces[k][1]],poles[relevant_faces[k][2]],centers[i],pole_radii[relevant_faces[k][0]],pole_radii[relevant_faces[k][1]],pole_radii[relevant_faces[k][2]])
                dis=dist(p,centers[i])
                if dis<mindis:
                    center_temp=p
                    radii_temp=radii
                    mindis=dis
        newcenters.append(center_temp)
        newradii.append(radii_temp)

        
    newcenters=np.array(newcenters).astype(np.float64)
    newradii=np.array(newradii).astype(np.float64)
    return  newcenters,newradii
def cluster1(pole,pole_radii,pole_volume,n,max_iters,real_name,kd_tree,normals,pts):
    K=int(n/4)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(pole)
    labels = kmeans.labels_
    pole_radii = np.expand_dims(pole_radii, axis=1)

    # 循环直到K=N
    p=-1
    while K < n:
        if K + (K // 2)<n:
            p=K//2
        else:
            p=n-K
        
        prev_centers = kmeans.cluster_centers_
        distances = distance.cdist(pole, prev_centers, metric='euclidean')

    #找到每个聚类中每个点到聚类中心的最小距离的索引
        closest_indices = np.argmin(distances, axis=0)
        center=np.array(pole)[closest_indices]
        radii=np.array(pole_radii)[closest_indices]
        K=K+p
       

        
        new_centers_indices = choose_sphere(center,radii,pts,pole,pole_radii,p)
        new_centers = np.array(pole)[new_centers_indices]
        prev_centers = center
       
        for _ in range(max_iters):
            kmeans = KMeans(n_clusters=K, init=np.vstack([prev_centers, new_centers]), n_init=1).fit(pole)
            if np.all(kmeans.cluster_centers_ == kmeans.cluster_centers_):
                break
            prev_centers = kmeans.cluster_centers_
            distances = distance.cdist(pole, prev_centers, metric='euclidean')

    #找到每个聚类中每个点到聚类中心的最小距离的索引
            closest_indices = np.argmin(distances, axis=0)
            prev_centers=np.array(pole)[closest_indices]
    
    distances = distance.cdist(pole, kmeans.cluster_centers_, metric='euclidean')

    #找到每个聚类中每个点到聚类中心的最小距离的索引
    closest_indices = np.argmin(distances, axis=0)
    # labels1 = assign_labels(pole,pole_radii ,pole[closest_indices],pole_radii[closest_indices])
    # array = np.random.rand(100, 3)
    # # ps.set_program_name("important app")
    # # ps.set_verbosity(0)
    # # ps.set_use_prefs_file(False)

    # # # initialize
    # # ps.init()
    # for i in range(len(pole[closest_indices])):
    #     label_tt=np.where(np.array(labels1) == i)[0]
    #     center_temp=np.array(pole)[label_tt]
    
        # # points = np.random.rand(100, 3)
    
    # ps_cloud = ps.register_point_cloud("my points1",pole)
    # # ps_cloud = ps.register_point_cloud("my points",pole[closest_indices])
    # pole_label1,inner_poles1,outer_poles1=label_pole(pole[closest_indices],kd_tree,normals,point_cloud)
    # ps_cloud = ps.register_point_cloud("my points2",np.array(pole[closest_indices])[inner_poles1])
        # # # # visualize!
        # if len(center_temp)<=1:
        #     ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp.reshape(-1,3))
            


        #     vals = np.tile(array[i], (len(center_temp), 1))
        #     ps_cloud.add_color_quantity("rand colors", vals)
        #     ps_cloud.set_radius(0.007)
        # ps_cloudq = ps.register_point_cloud("my points", pole)  
    # ps.show()   

    return labels, np.array(pole)[closest_indices],np.array(pole_radii)[closest_indices]

boundary_points = []  # 用于存储边界点
boundary_spheres = []  # 用于存储边界球
def distance_to_sphere(surface_point, centers, radii):
    # 计算点与每个中轴球的球面距离
    center_radii=radii.reshape(-1, 1)
    distances = np.linalg.norm(surface_point - centers[:, np.newaxis], axis=2) - center_radii
    return distances

# def find_boundary_points(centers, radii, surface_points, k):
#     # 初始化一个空的列表来存储边界点的索引
#     boundary_point_indices = []

#     # 对于每个表面点
#     for i, point in enumerate(surface_points):
#         # 计算点与中轴球的球面距离
#         distances = distance_to_sphere(point,centers, radii)
        
#         # 使用K最近邻算法计算K邻域内的点
#         knn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(surface_points)
#         indices = knn.kneighbors([point], return_distance=False)[0]
#         labels=np.argmin(distances, axis=1)
#         # 检查K邻域内的点的中轴球标记
#         for index in indices:
#             if np.any(np.argmin(distances, axis=1) != np.argmin(distance_to_sphere(surface_points[index], centers, radii), axis=1)):
#                 # 将当前点标记为边界点
#                 boundary_point_indices.append(i)
#                 break

#     # 边界点所属的中轴球即为边界球
#     tt=np.argmin(distance_to_sphere( surface_points[boundary_point_indices],centers, radii),axis=1)
#     print(tt)
#     boundary_spheres = np.argmin(distance_to_sphere( surface_points[boundary_point_indices],centers, radii), axis=1)

#     return boundary_point_indices, boundary_spheres
def find_boundary_points(centers, center_radii, pts, k):
    # 计算每个表面点最近的中轴球索引
    distances = np.linalg.norm(pts[:, np.newaxis] - centers, axis=2) - center_radii.T
    nearest_sphere_indices = np.argmin(distances, axis=1)
    
    # 使用KNN算法为每个表面点找到邻域内的点
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    
    # 检查邻域内的点的中轴球标记，确定边界点
    boundary_points = []
    for i in range(len(pts)):
        neighbor_sphere_indices = nearest_sphere_indices[indices[i]]
        if len(set(neighbor_sphere_indices)) > 1:
            boundary_points.append(i)  # 存储边界点的索引
    
    # 根据边界点确定拥有这些边界点的中轴球，即边界球
    boundary_sphere_indices = nearest_sphere_indices[boundary_points]
    unique_boundary_sphere_indices = np.unique(boundary_sphere_indices)
    boundary_spheres = centers[unique_boundary_sphere_indices]
    boundary_radii = center_radii[unique_boundary_sphere_indices]
    
    return boundary_points, boundary_spheres,boundary_radii
    

# def cluster2(pole,pole_radii,n,pts):
#     K=int(n/4)
#     center_radii=pole_radii.reshape(-1, 1)
#     distances = np.linalg.norm(pts - pole[:, np.newaxis], axis=2) - center_radii
#     min_distances = np.min(distances, axis=0)
#     max_k_indices = np.argsort(min_distances)[-K:]
#     array=np.array(pts)[max_k_indices]
#     distances_new = np.linalg.norm(array - np.array(pole)[:, np.newaxis], axis=2) - center_radii
#     min_distances = np.min(distances_new, axis=1)
#     closest_indices = np.argmin(distances_new, axis=0)
#     newcenter=np.array(pole)[closest_indices]
#     newcenter=[]
#     for i in range (K) :
#         newcenter.append(pole[i])
#     newcenter=np.array(newcenter).ravel()
#     newcenter = newcenter.astype(np.float128)
#     lb, ub = np.full(K*3, -0.5), np.full(K*3, 0.5)
#     # variable_bounds = Bounds(lb, ub)
#     # res12=minimize(evalfunc,newcenter,jac=evalgrad,method='L-BFGS-B',options={'disp': True},bounds=variable_bounds)
#     optimized_centers = res12.x.reshape(K, 3)
#     distances12 = distance.cdist(pole , optimized_centers)
#     closest_pole_index12 = np.argmin(distances12,axis=0)
#     newcenter=np.array(pole)[closest_pole_index12]
#     radii=np.array(pole_radii)[closest_pole_index12]

#     # 循环直到K=N
#     p=-1
#     while K < n:
#         if K + (K // 2)<n:
#             p=K//2
#         else:
#             p=n-K
        
#         prev_centers = newcenter
     

#     #找到每个聚类中每个点到聚类中心的最小距离的索引
        
#         K=K+p
       

        
#         new_centers_indices = choose_sphere(prev_centers,radii,pts,pole,pole_radii,p)
#         new_centers = np.array(pole)[new_centers_indices]
#         newcenter=np.vstack([prev_centers, new_centers])
#         newcenter=newcenter.ravel()
#         newcenter = newcenter.astype(np.float64)
#         lb, ub = np.full(K*3, -0.5), np.full(K*3, 0.5)
#         variable_bounds = Bounds(lb, ub)
#         res12=minimize(evalfunc,newcenter,jac=evalgrad,method='L-BFGS-B',options={'disp': True},bounds=variable_bounds)
#         optimized_centers = res12.x.reshape(K, 3)
#         distances12 = distance.cdist(pole , optimized_centers)
#         closest_pole_index12 = np.argmin(distances12,axis=0)
#         newcenter=np.array(pole)[closest_pole_index12]
#         radii=np.array(pole_radii)[closest_pole_index12]
       
       
        
    
#     distances = distance.cdist(pole, newcenter, metric='euclidean')

#     #找到每个聚类中每个点到聚类中心的最小距离的索引
#     closest_indices = np.argmin(distances, axis=0)
#     # labels1 = assign_labels(pole,pole_radii ,pole[closest_indices],pole_radii[closest_indices])
#     # array = np.random.rand(100, 3)
#     # # ps.set_program_name("important app")
#     # # ps.set_verbosity(0)
#     # # ps.set_use_prefs_file(False)

#     # # # initialize
#     # # ps.init()
#     # for i in range(len(pole[closest_indices])):
#     #     label_tt=np.where(np.array(labels1) == i)[0]
#     #     center_temp=np.array(pole)[label_tt]
    
#         # # points = np.random.rand(100, 3)
    
#     # ps_cloud = ps.register_point_cloud("my points1",pole)
#     # # ps_cloud = ps.register_point_cloud("my points",pole[closest_indices])
#     # pole_label1,inner_poles1,outer_poles1=label_pole(pole[closest_indices],kd_tree,normals,point_cloud)
#     # ps_cloud = ps.register_point_cloud("my points2",np.array(pole[closest_indices])[inner_poles1])
#         # # # # visualize!
#         # if len(center_temp)<=1:
#         #     ps_cloud = ps.register_point_cloud("my points-%s"%i, center_temp.reshape(-1,3))
            


#         #     vals = np.tile(array[i], (len(center_temp), 1))
#         #     ps_cloud.add_color_quantity("rand colors", vals)
#         #     ps_cloud.set_radius(0.007)
#         # ps_cloudq = ps.register_point_cloud("my points", pole)  
#     # ps.show()   

#     return  np.array(pole)[closest_indices],np.array(pole_radii)[closest_indices]


def calculate_barycentric_coordinates(A, B, C, P):
    # 计算三角形的面积
    S = np.linalg.norm(np.cross(B - A, C - A)) / 2.0
    
    # 计算三个子三角形的面积
    S1 = np.linalg.norm(np.cross(B - P, C - P)) / 2.0
    S2 = np.linalg.norm(np.cross(C - P, A - P)) / 2.0
    S3 = np.linalg.norm(np.cross(A - P, B - P)) / 2.0
    
    # 计算重心坐标
    alpha = S1 / S
    beta = S2 / S
    gamma = S3 / S
    
    return alpha, beta, gamma
def calculate_projection_point(P, A, B, C):
    normal_vector = np.cross(B - A, C - A)  # assuming A, B, C are not collinear
    projection_point_A = A + np.dot(P - A, normal_vector) / np.dot(normal_vector, normal_vector) * normal_vector
    projection_point_B = B + np.dot(P - B, normal_vector) / np.dot(normal_vector, normal_vector) * normal_vector
    projection_point_C = C + np.dot(P - C, normal_vector) / np.dot(normal_vector, normal_vector) * normal_vector
    return np.array([projection_point_A, projection_point_B, projection_point_C])
def project_point_onto_triangle(A, B, C, P,A_radii,B_radii,C_radii):
    # 计算三角形的法向量
    normal_vector = np.cross(B - A, C - A)

    # 计算点到三角形的垂直投影点
    projection_point =calculate_projection_point(P, A, B, C)

    # 判断投影点是否在三角形内部
    alpha, beta, gamma = calculate_barycentric_coordinates(A, B, C, projection_point)
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and alpha+beta+gamma==1:
        return projection_point ,alpha*A_radii+beta*B_radii+gamma*C_radii
    else:
        # 如果不在三角形内部，计算点到三条边的距离，选择最小距离对应的投影点
        distances = [
            np.linalg.norm(np.cross(B - A, P - A)) / np.linalg.norm(B - A),
            np.linalg.norm(np.cross(C - B, P - B)) / np.linalg.norm(C - B),
            np.linalg.norm(np.cross(A - C, P - C)) / np.linalg.norm(A - C)
        ]
        min_index = np.argmin(distances)
        
        if min_index == 0:
            return project_point_onto_line(A, B, P,A_radii,B_radii)
        elif min_index == 1:
            return project_point_onto_line(B, C, P,B_radii,C_radii)
        else:
            return project_point_onto_line(C, A, P,C_radii,A_radii)

def project_point_onto_triangle_para(A, B, C, P):
    # 计算三角形的法向量
    normal_vector = np.cross(B - A, C - A)

    # 计算点到三角形的垂直投影点
    projection_point = calculate_projection_point(P, A, B, C)

    # 判断投影点是否在三角形内部
    alpha, beta, gamma = calculate_barycentric_coordinates(A, B, C, projection_point)
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and alpha+beta+gamma<=1:
        return alpha,beta,gamma
    else:
        # 如果不在三角形内部，计算点到三条边的距离，选择最小距离对应的投影点
        distances = [
            np.linalg.norm(np.cross(B - A, P - A)) / np.linalg.norm(B - A),
            np.linalg.norm(np.cross(C - B, P - B)) / np.linalg.norm(C - B),
            np.linalg.norm(np.cross(A - C, P - C)) / np.linalg.norm(A - C)
        ]
        min_index = np.argmin(distances)
        
        if min_index == 0:
            return project_point_onto_line_para(A, B, P),888,888
        elif min_index == 1:
            return 888,project_point_onto_line_para(B, C, P),888
        else:
            return 888,888,project_point_onto_line_para(C, A, P)
        
def project_point_onto_line(A, B, P,A_radii,B_radii):
    # 计算线段AB的方向向量
    AB = B - A
    
    # 计算点P到线段AB的投影比例
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    radii=0
    # 如果投影点在线段内，计算投影点的坐标
    if 0 <= t <= 1:
        projection_point = A + t * AB
        radii=A_radii+t*(B_radii-A_radii)
    else:
        # 如果投影点不在线段内，将其映射到线段的端点
        if t < 0:
            projection_point = A
            radii=A_radii
        else:
            projection_point = B
            radii=B_radii
    
    return projection_point,radii

def project_point_onto_line_para(A, B, P):
    # 计算线段AB的方向向量
    AB = B - A
    
    # 计算点P到线段AB的投影比例
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    
    # 如果投影点在线段内，计算投影点的坐标
    # if 0 <= t <= 1:
    #     projection_point = A + t * AB
    # else:
    #     # 如果投影点不在线段内，将其映射到线段的端点
    #     if t < 0:
    #         projection_point = A
    #     else:
    #         projection_point = B
    
    return t

def save_skel_mesh(v, f, e, path_f, path_e):
    f_file = open(path_f, "w")
    e_file = open(path_e, "w")
    v_num = len(v)
    f_num = len(f)
    e_num = len(e)

    for j in range(v_num):
        f_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(v[j][2]) + "\n")
    for j in range(f_num):
        f_file.write("f " + str(int(f[j][0]) + 1) + " " + str(int(f[j][1]) + 1) + " " + str(int(f[j][2]) + 1) + "\n")

    for j in range(v_num):
        e_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(v[j][2]) + "\n")
    for j in range(e_num):
        e_file.write("l " + str(int(e[j][0]) + 1) + " " + str(int(e[j][1]) + 1) + "\n")
# # 使用示例

   



# def farthest_sphere_sampling(centers, radii, point_cloud, num_samples):
#     sampled_indices =  [np.random.randint(len(centers))]
#     newcenters=np.delete(centers,sampled_indices,axis=0)
#     newradii=np.delete(radii,sampled_indices)
#     for _ in range(1, num_samples):
   
#         distances2 = np.min(distance.cdist(point_cloud, centers[sampled_indices]) - radii[sampled_indices], axis=1)
        

#         new_pts_index = np.argmax(distances2)
        
#         distances = distance.cdist(point_cloud[new_pts_index][np.newaxis, :], newcenters) - newradii
#         new_sphere_index = np.argmin(distances)
#         a_number = np.where((np.array(newcenters[new_sphere_index])== centers).all(axis=1))[0]
#         newcenters=np.delete(newcenters,new_sphere_index,axis=0)
#         newradii=np.delete(newradii,new_sphere_index)
      
       
      
        
#         sampled_indices.append(int(a_number))
    
#     selected_centers = [centers[i] for i in sampled_indices]
#     selected_radii = [radii[i] for i in sampled_indices]
    
#     return selected_centers, selected_radii
import numpy as np
from scipy.spatial import distance

def farthest_sphere_sampling(centers, radii, point_cloud, num_samples):
    sampled_indices =  [np.random.randint(len(centers))]
    newcenters=np.delete(centers,sampled_indices,axis=0)
    newradii=np.delete(radii,sampled_indices)
    for _ in range(1, num_samples):
   
        distances2 = np.min(distance.cdist(point_cloud, centers[sampled_indices]) - radii[sampled_indices], axis=1)
        

        new_pts_index = np.argmax(distances2)
        
        distances = distance.cdist(point_cloud[new_pts_index][np.newaxis, :], newcenters) - newradii
        new_sphere_index = np.argmin(distances)
        a_number = np.where((np.array(newcenters[new_sphere_index])== centers).all(axis=1))[0]
        newcenters=np.delete(newcenters,new_sphere_index,axis=0)
        newradii=np.delete(newradii,new_sphere_index)
      
       
      
        
        sampled_indices.append(int(a_number))
    
    selected_centers = [centers[i] for i in sampled_indices]
    selected_radii = [radii[i] for i in sampled_indices]
    
    return selected_centers, selected_radii



def farthest_sphere_sampling_torch(centers, radii, point_cloud, num_samples, device='cuda'):
    """
    最远球体采样（修复索引越界问题）
    """
    centers = torch.tensor(centers, dtype=torch.float32, device=device)  # (N, D)
    radii = torch.tensor(radii, dtype=torch.float32, device=device)      # (N,)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32, device=device)  # (M, D)

    # 初始化：选择损失最大的球体
    distances_to_points = torch.abs(torch.cdist(point_cloud, centers) - radii.unsqueeze(0))  # (M, N)
    min_distances, _ = torch.min(distances_to_points, dim=0)  # 每个球到点云的最小距离
    first_sphere_index = torch.argmax(min_distances).item()  # 找到最远的球体索引

    sampled_indices = [first_sphere_index]
    sampled_centers = centers[first_sphere_index].unsqueeze(0)
    sampled_radii = radii[first_sphere_index].unsqueeze(0)

    remaining_indices = torch.arange(len(centers), device=device)
    remaining_indices = remaining_indices[remaining_indices != first_sphere_index]

    distances_to_sampled = torch.full((point_cloud.size(0),), float('inf'), device=device)

    for _ in range(1, num_samples):
        # 计算点云到已采样球体的最近距离
        distances = torch.abs(torch.cdist(point_cloud, sampled_centers) - sampled_radii.unsqueeze(0))
        min_distances, _ = torch.min(distances, dim=1)
        distances_to_sampled = torch.min(distances_to_sampled, min_distances)

        # 找到距离最远的点
        farthest_point_index = torch.argmax(distances_to_sampled).item()
        farthest_point = point_cloud[farthest_point_index]

        # 找到包含该点的最近球体
        remaining_centers = centers[remaining_indices]
        remaining_radii = radii[remaining_indices]
        distances_to_remaining = torch.abs(torch.norm(remaining_centers - farthest_point, dim=1) - remaining_radii)
        closest_sphere_index = torch.argmin(distances_to_remaining).item()

        # 更新采样列表
        new_index = remaining_indices[closest_sphere_index].item()
        sampled_indices.append(new_index)
        sampled_centers = torch.cat([sampled_centers, centers[new_index].unsqueeze(0)], dim=0)
        sampled_radii = torch.cat([sampled_radii, radii[new_index].unsqueeze(0)], dim=0)

        # 更新剩余球体的索引
        remaining_indices = remaining_indices[remaining_indices != new_index]

    sampled_centers = sampled_centers.cpu().numpy()
    sampled_radii = sampled_radii.cpu().numpy()
    return sampled_centers, sampled_radii





# def farthest_sphere_sampling_torch(centers, radii, point_cloud, num_samples, device='cuda'):
#     """
#     基于 PyTorch 的最远球体采样
#     :param centers: 球体中心 (N, D) 的张量
#     :param radii: 球体半径 (N,) 的张量
#     :param point_cloud: 点云数据 (M, D) 的张量
#     :param num_samples: 要采样的球体数量
#     :param device: 使用的设备 ('cuda' 或 'cpu')
#     :return: 采样后的球体中心和半径
#     """
#     # 将数据移动到指定设备
#     centers = torch.tensor(centers, dtype=torch.float32, device=device)
#     radii = torch.tensor(radii, dtype=torch.float32, device=device)
#     point_cloud = torch.tensor(point_cloud, dtype=torch.float32, device=device)

#     # 初始化采样
#     sampled_indices = [torch.randint(0, len(centers), (1,)).item()]
#     sampled_centers = centers[sampled_indices]
#     sampled_radii = radii[sampled_indices]

#     # 初始化点云到已采样球体的最短距离
#     distances_to_sampled = torch.full((point_cloud.size(0),), float('inf'), device=device)

#     # 采样
#     for _ in range(1, num_samples):
#         # 计算点云中每个点到已采样球体的最近距离
#         distances = torch.cdist(point_cloud, sampled_centers) - sampled_radii
#         distances, _ = torch.min(distances, dim=1)
#         distances_to_sampled = torch.min(distances_to_sampled, distances)

#         # 找到距离最远的点
#         farthest_point_index = torch.argmax(distances_to_sampled)
#         farthest_point = point_cloud[farthest_point_index]

#         # 找到包含该点的最近球体
#         distances_to_remaining = torch.norm(centers - farthest_point, dim=1) - radii
#         closest_sphere_index = torch.argmin(distances_to_remaining)

#         # 更新采样列表
#         sampled_centers = torch.cat([sampled_centers, centers[closest_sphere_index].unsqueeze(0)], dim=0)
#         sampled_radii = torch.cat([sampled_radii, radii[closest_sphere_index].unsqueeze(0)], dim=0)

#         # 移除已采样的球体
#         centers = torch.cat([centers[:closest_sphere_index], centers[closest_sphere_index + 1:]], dim=0)
#         radii = torch.cat([radii[:closest_sphere_index], radii[closest_sphere_index + 1:]], dim=0)

#     return sampled_centers.cpu().numpy(), sampled_radii.cpu().numpy()












