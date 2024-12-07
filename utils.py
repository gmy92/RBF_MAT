import trimesh
import numpy as np
import torch
from torch import linalg as LA
from scipy.spatial import Delaunay

def read_VD(path):
    points = []
    radius = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = line.split(' ')
            points.append([float(line[1]), float(line[2]), float(line[3])])
            radius.append([float(line[4])])
    return points, radius

def read_point(path):
    points = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = line.split(' ')
            points.append([float(line[1]), float(line[2]), float(line[3])])
    return points


def save_obj(path, verts, faces=None):
    

    with open(path, 'w') as f:
        
        for v in verts:
            f.write('v %f %f %f\n' %(v[0], v[1], v[2]))
        if faces is not None:
            faces = faces.tolist()
            for ff in faces:
                f.write('f %d %d %d\n' % (ff[0] + 1, ff[1] + 1, ff[2] + 1))
        
mesh_scale=0.5       
def normlize_point(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    # normalize mesh

    bbmin = points.min(0)
    bbmax = points.max(0)
    center = (bbmin + bbmax) * 0.5

    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    normalized_points = (points - center) * scale
    return normalized_points

def save_ma(path, verts, radii , lines=None,faces=None):
    a=0
    b=0
    if lines is not None:
        a= len(lines)
    if faces is not None:
        b= len(faces)

    with open(path, 'w') as f:
       

        f.write('%d %d %d\n' %(len(verts),a,b))

        for v in range(len(verts)):
            f.write('v %f %f %f %f\n' %(verts[v][0], verts[v][1], verts[v][2],radii[v]))
        if lines is not None:
            # lines = lines.tolist()
            for ll in lines:
                f.write('e %d %d\n' % (int(ll[0]) , int(ll[1])))
        if faces is not None:
            # faces = faces.tolist()
            for ff in faces:
                f.write('f %d %d %d\n' % (int(ff[0]) , int(ff[1]) , int(ff[2]) ))


# Codes borrowed from https://gist.github.com/dendenxu/ee5008acb5607195582e7983a384e644#file-moller_trumbore-py-L27
def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)
def generate_sphere(center, radius, num_points_theta=30, num_points_phi=30):
    theta = np.linspace(0, 2 * np.pi, num_points_theta)
    phi = np.linspace(0, np.pi, num_points_phi)

    theta, phi = np.meshgrid(theta, phi)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
 

    return vertices
def combine_spheres(center,radii):
    all_vertices = []
    all_faces = []
    offset = 0  # Offset to keep track of vertex indices
    
    for i in range(len(center)):
        vertices = generate_sphere(center[i], radii[i])
       
        num_points_theta = int(len(vertices) ** 0.5)
        num_points_phi = int(len(vertices) / num_points_theta)
        # Create faces
        faces = []
        for i in range(num_points_phi - 1):
            for j in range(num_points_theta - 1):
                p1 = offset + i * num_points_theta + j
                p2 = offset + p1 + 1
                p3 = offset + (i + 1) * num_points_theta + j
                p4 = offset + p3 + 1

                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])

        all_vertices.append(vertices)
        all_faces.append(faces)
        offset += len(vertices)

    return np.vstack(all_vertices),  np.vstack(all_faces)
# def save_spheres(path,centers, radius):
    all_vertices=[]
    all_triangles=[]
    centers=np.array(centers)
    radius=np.array(radius.cpu())
    all_vertices,all_triangles=combine_spheres(centers,radius)
    with open(path, 'w') as file:
        for vertices  in all_vertices:
            
            file.write(f'v {vertices[0]} {vertices[1]} {vertices[2]}\n')

           
            
        for triangles in all_triangles:
            
            

           
            file.write(f'f {triangles[0]} {triangles[1]} {triangles[2]}\n')

def save_spheres1(path,center, radius):
    mesh= trimesh.load('sphere16.off')
    sp_v=mesh.vertices
    sp_f=mesh.faces
    radius=np.array(radius)
    with open(path, "w") as file:
        
           
            v, r =  center, radius
            
            v_ =sp_v* r + v
            for m in range(len(v_)):
                file.write('v ' + str(v_[m][0]) + ' ' + str(v_[m][1]) + ' ' + str(v_[m][2]) + '\n')

            m=0
            base = m * sp_v.shape[0] + 1
            
            for j in range(len(sp_f)):
                file.write(
                    'f ' + str(sp_f[j][0] + base) + ' ' + str(sp_f[j][1] + base) + ' ' + str(sp_f[j][2] + base) + '\n')    
    
def save_spheres(path,center, radius):
    mesh= trimesh.load('sphere16.off')
    sp_v=mesh.vertices
    sp_f=mesh.faces
   
    radius=np.array(radius)
    with open(path, "w") as file:
         for i in range(len(center)):
            
           
            v, r =  center[i], radius[i]
            
            v_ =sp_v* r + v
            for m in range(len(v_)):
                file.write('v ' + str(v_[m][0]) + ' ' + str(v_[m][1]) + ' ' + str(v_[m][2]) + '\n')

         for m in range(len(center)):
            base = m * sp_v.shape[0] + 1
            
            for j in range(len(sp_f)):
                file.write(
                    'f ' + str(sp_f[j][0] + base) + ' ' + str(sp_f[j][1] + base) + ' ' + str(sp_f[j][2] + base) + '\n')
    
def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the index's last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))

def Delaunay_triangulation(points):
    points = np.array(points)
    tri = Delaunay(points)
    return tri.simplices

def load_ma(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount=0
    pts = np.zeros((1, 3), np.float64)
    faces = np.zeros((1, 3), np.int)
    edges=np.zeros((1, 2), np.int)
    radius=np.zeros((1, 1), np.float64)
    p_num = 0
    f_num = 0
    e_num=0

    for line in lines:
        linecount = linecount + 1
        word = line.split()

        if linecount == 1:
            p_num = int(word[0])
            e_num = int(word[1])
            f_num = int(word[2])

        if linecount == 2:

            pts = np.zeros((p_num, 3), np.float)
            faces = np.zeros((f_num, 3), np.int)
            edges = np.zeros((e_num, 2), np.int)
            radius = np.zeros((p_num, 1), np.float64)
        if linecount >= 2 and linecount < 2 + p_num:
            pts[linecount - 2, :] = np.float64(word[1:4])
            radius[linecount - 2, 0] = np.float64(word[4])
        if linecount >= 2 + p_num and linecount < 2 + p_num+e_num:
            edges[linecount - 2 - p_num] = np.int32(word[1:3])
        if linecount >= 2 + p_num+e_num :
            faces[linecount - 2 - p_num-e_num] = np.int32(word[1:4])

    fopen.close()
    return pts, edges,faces,radius

def load_ply_points_normal(pc_filepath, expected_point=6624):
    fopen = open(pc_filepath, 'r', encoding='utf-8')
    lines = fopen.readlines()
    pts = np.zeros((expected_point, 6), np.float64)

    total_point = 0
    feed_point_count = 0

    start_point_data = False
    for line in lines:
        word = line.split()
        if word[0] == 'element' and word[1] == 'vertex':
            total_point = int(word[2])
            # if expected_point > total_point:
            #     pts = np.zeros((total_point, 3), np.float64)
            # continue

        if start_point_data == True:
            pts[feed_point_count, :] = np.float64(word[0:6])
            feed_point_count += 1

        if word[0] == 'end_header':
            start_point_data = True

        if feed_point_count >= expected_point:
            break

    fopen.close()
    return pts
def load_ply_points_normal_sample(pc_filepath, expected_point=48000):
    fopen = open(pc_filepath, 'r', encoding='utf-8')
    lines = fopen.readlines()
    pts = np.zeros((expected_point, 3), np.float64)

    total_point = 0
    feed_point_count = 0

    start_point_data = False
    for line in lines:
        word = line.split()
        if word[0] == 'element' and word[1] == 'vertex':
            total_point = int(word[2])
            # if expected_point > total_point:
            #     pts = np.zeros((total_point, 3), np.float64)
            # continue

        if start_point_data == True:
            pts[feed_point_count, :] = np.float64(word[0:3])
            feed_point_count += 1

        if word[0] == 'end_header':
            start_point_data = True

        if feed_point_count >= expected_point:
            break

    fopen.close()
    return pts
def winding_number(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Parallel implementation of the Generalized Winding Number of points on the mesh
    O(n_points * n_faces) memory usage, parallelized execution
    1. Project tris onto the unit sphere around every points
    2. Compute the signed solid angle of the each triangle for each point
    3. Sum the solid angle of each triangle
    Parameters
    ----------
    pts    : torch.Tensor, (n_points, 3)
    verts  : torch.Tensor, (n_verts, 3)
    faces  : torch.Tensor, (n_faces, 3)
    This implementation is also able to take a/multiple batch dimension
    """
    # projection onto unit sphere: verts implementation gives a little bit more performance
    uv = verts[..., None, :, :] - pts[..., :, None, :]  # n_points, n_verts, 3
    uv = uv / uv.norm(dim=-1, keepdim=True)  # n_points, n_verts, 3

    # gather from the computed vertices (will result in a copy for sure)
    expanded_faces = faces[..., None, :, :].expand(*faces.shape[:-2], pts.shape[-2], *faces.shape[-2:])  # n_points, n_faces, 3

    u0 = multi_gather(uv, expanded_faces[..., 0])  # n, f, 3
    u1 = multi_gather(uv, expanded_faces[..., 1])  # n, f, 3
    u2 = multi_gather(uv, expanded_faces[..., 2])  # n, f, 3

    e0 = u1 - u0  # n, f, 3
    e1 = u2 - u1  # n, f, 3
    del u1

    # compute solid angle signs
    sign = (torch.cross(e0, e1) * u2).sum(dim=-1).sign()

    e2 = u0 - u2
    del u0, u2

    l0 = e0.norm(dim=-1)
    del e0

    l1 = e1.norm(dim=-1)
    del e1

    l2 = e2.norm(dim=-1)
    del e2

    # compute edge lengths: pure triangle
    l = torch.stack([l0, l1, l2], dim=-1)  # n_points, n_faces, 3

    # compute spherical edge lengths
    l = 2 * (l/2).arcsin()  # n_points, n_faces, 3

    # compute solid angle: preparing: n_points, n_faces
    s = l.sum(dim=-1) / 2
    s0 = s - l[..., 0]
    s1 = s - l[..., 1]
    s2 = s - l[..., 2]

    # compute solid angle: and generalized winding number: n_points, n_faces
    eps = 1e-10  # NOTE: will cause nan if not bigger than 1e-10
    solid = 4 * (((s/2).tan() * (s0/2).tan() * (s1/2).tan() * (s2/2).tan()).abs() + eps).sqrt().arctan()
    signed_solid = solid * sign  # n_points, n_faces

    winding = signed_solid.sum(dim=-1) / (4 * torch.pi)  # n_points

    return winding

def cal_degree(a,b):
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    print(a)
    print(length_a)
# 计算两个向量的点积
    dot_product = np.dot(a, b)

# 计算夹角的余弦值
    cosine_angle = dot_product / (length_a * length_b)

# 计算夹角的弧度值
    angle_rad = np.arccos(cosine_angle)

# 将弧度值转换为角度值
    angle_deg = np.degrees(angle_rad)
    return cosine_angle

def cal_angle(center1,r1,center2,r2):
    distance = np.linalg.norm(center2 - center1)
    cosin1=(r1*r1+distance*distance-r2*r2)/(2*distance*r1)
    cosin2=(r2*r2+distance*distance-r1*r1)/(2*distance*r2)
    angle_rad1 = np.arcsin(cosin1)
    angle_rad2 = np.arcsin(cosin2)
    
    angle_rad1=np.degrees(angle_rad1)
    angle_rad2=np.degrees(angle_rad2)
   
    return np.cos(180-angle_rad2-angle_rad1)

def gaussian(x,center,radius):
    
    distance=(LA.vector_norm(x-center))/radius
    
    if distance < 1 :
        value=(1-distance)*(1-distance)*(1-distance)*(1-distance)*(1+4*distance)
    else:
        value=0
    return value*radius
def closest_distance_np(p1, p2, is_sum=False):
    '''
    :param p1: size[N, D], numpy array
    :param p2: size[M, D], numpy array
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    '''



   
    assert p1.size(1) == p2.size(1)
    p1 = p1.unsqueeze(0)
    p2 = p2.unsqueeze(0)

    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)
    min_dist, min_indice = torch.min(dist, dim=1)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice
def find_points_within_radius(point_cloud, target_point, radius):
    # 计算点云中每个点与目标点之间的距离
    
    distances = torch.norm(point_cloud - target_point, dim=1)

    # 找到距离在半径范围内的点的索引
    indices = torch.nonzero(distances <= radius, as_tuple=True)

    # 返回在半径范围内的点
    return point_cloud[indices[0]]
   
  
def find_intersec_point(point_cloud1,point_cloud2):
    if len(point_cloud1)==0  or  len(point_cloud2)==0:
        return 0
    else:
        dist,min_indice = closest_distance_np(point_cloud1,point_cloud2)
        
        
        return point_cloud1[min_indice]

def loss_func(value, center ,radius, weight,sample_point):
    loss_temp=[]
    for i in range(len(center)):
        for j in range (len(sample_point)):
            loss_temp += weight[i] * gaussian(sample_point[j], center[i], radius[i])
        loss+=value[i]-loss_temp[i]

def ball_vis(position, r):
    for i in range(r.shape[0]):
        mesh = trimesh.load('./assets/sphere_I.obj')
        mesh.vertices = mesh.vertices * r[i]
        mesh.vertices = mesh.vertices + position[i]
        mesh.export('./vis_ball/%04d.obj'%i)

def find_overlapping_points(point_cloud1, point_cloud2):
    # 计算每个点云1中的点与点云2中所有点的距离
    distances = torch.cdist(point_cloud1.unsqueeze(0), point_cloud2.unsqueeze(0)).squeeze()

    # 找到距离为0的点的索引
    indices = torch.nonzero(distances == 0, as_tuple=True)
   
    # 返回重叠的点
    return point_cloud1[indices[0]]


