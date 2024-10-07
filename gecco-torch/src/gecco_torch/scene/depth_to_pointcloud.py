import numpy as np

def sample_depth(image,depth,num_points):
    mask = image[:,:,3]
    # print("sample depth")
    # print(image)
    mask = mask == 255
    # print(sum(mask))
    flat_original_array = mask.flatten()
    # print(flat_original_array)

    # Randomly select 1000 indices where the values are True
    replace = False
    try:
        true_indices = np.random.choice(np.where(flat_original_array)[0], size=num_points, replace=replace)
    except:
        replace = True
        true_indices = np.random.choice(np.where(flat_original_array)[0], size=num_points, replace=replace)

    # Create a new 1D array with all False values
    new_1d_array = np.zeros_like(flat_original_array, dtype=bool)

    # Set the selected indices to True in the new 1D array
    new_1d_array[true_indices] = True

    # Reshape the new 1D array back to a 400x400 array
    mask = new_1d_array.reshape(400, 400)
    depth = depth[mask]
    x_coords = np.where(mask)[0]
    y_coords = np.where(mask)[1]
    return x_coords,y_coords,depth

def get_world_coords(image,depth,num,c2w,**kwargs):
    x_coords,y_coords,depth = sample_depth(image,depth,num)
    cx = 200
    cy = 200
    fx = 437
    fy = 437

    # fx = 50
    # fy = 50

    def blender_depth_to_actual_depth(depth,**kwargs):
        if kwargs['depth_information']['min'] != 0:
            raise Exception("The minimum value of the depth map is 0 but the conversion formula assumes its 0")
        # if depth == 255:
        #     return 2.5
        # elif depth < 1:
        #     return 1
        # else:
        #     # inverse the linear function 255/1.5 (x-1)
        #     return ((depth + 255/1.5)  * 1.5/255)
        def conversion(x):
            max_val = -kwargs['depth_information']['offset']+1/kwargs['depth_information']['size']
            if x == 255:
                return max_val
            elif x < -kwargs['depth_information']['offset']:
                return 0
            else:
                # inverse the linear function 255/1.5 (x-1)
                return x/(255*kwargs['depth_information']['size']) - (kwargs['depth_information']['offset'])
        return conversion(depth)
        
    threedpoints = np.zeros((num, 3))
    colors = np.zeros((num, 3))
    for i, (x,y) in enumerate(zip(x_coords, y_coords)):
        # offset of -1 in rendering, and 255 equals 2.5
        dep = blender_depth_to_actual_depth(depth[i],**kwargs)
        # print(depth[i],dep)
        # dep = depth[i]
        myx = (x - cx) * dep / fx
        myy = (y - cy) * dep / fy
        z = dep
        threedpoints[i] = [myy,-myx,-z]
        colors[i] = image[x,y,0:3]
    points = threedpoints

    camera_points = [np.array([x, y, z, 1]) for x, y, z in threedpoints]
    # Transform points to world space

    # world_points = np.array([np.dot(np.eye(4),point) for point in camera_points])
    world_points = np.array([np.dot(c2w,point) for point in camera_points])
    # Extract the 3D coordinates from the 4D homogeneous coordinates
    world_coordinates = np.array([(x / w, y / w, z / w) for x, y, z, w in world_points])
    points = world_coordinates
    return points,colors

def create_pointcloud(list_camera_infos,num_total_points,**kwargs):
    num_per_image = int(num_total_points/len(list_camera_infos))

    points = np.zeros((num_per_image*len(list_camera_infos),3))
    colors = np.zeros((num_per_image*len(list_camera_infos),3))
    for i in range(len(list_camera_infos)):
        p,c = get_world_coords(list_camera_infos[i].image_orig, list_camera_infos[i].depth_map,num_per_image,list_camera_infos[i].c2w,**kwargs)
        points[i*num_per_image:(i+1)*num_per_image] = p
        colors[i*num_per_image:(i+1)*num_per_image] = c

    return points,colors