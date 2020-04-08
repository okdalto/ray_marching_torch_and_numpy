import numpy as np
import cv2
import math

def sdf(position):
    return np.linalg.norm(position, axis=2) - 0.5

def get_normal(position):
    epsilon = np.full((len(position), len(position[0]), 1), 0.001)
    w = len(position)
    h = len(position[0])
    x = position[:,:,0].reshape(w, h, -1)
    y = position[:,:,1].reshape(w, h, -1)
    z = position[:,:,2].reshape(w, h, -1)

    x_neg = (x - epsilon)
    y_neg = (y - epsilon)
    z_neg = (z - epsilon)

    x_pos = (x + epsilon)
    y_pos = (y + epsilon)
    z_pos = (z + epsilon)

    x_neg_sdf = sdf(np.concatenate((x_neg, y, z), axis=2)).reshape(w, h, -1)
    y_neg_sdf = sdf(np.concatenate((x, y_neg, z), axis=2)).reshape(w, h, -1)
    z_neg_sdf = sdf(np.concatenate((x, y, z_neg), axis=2)).reshape(w, h, -1)

    x_pos_sdf = sdf(np.concatenate((x_pos, y, z), axis=2)).reshape(w, h, -1)
    y_pos_sdf = sdf(np.concatenate((x, y_pos, z), axis=2)).reshape(w, h, -1)
    z_pos_sdf = sdf(np.concatenate((x, y, z_pos), axis=2)).reshape(w, h, -1)

    x_result = x_pos_sdf - x_neg_sdf
    y_result = y_pos_sdf - y_neg_sdf
    z_result = z_pos_sdf - z_neg_sdf

    result = np.concatenate((x_result,y_result,z_result), axis=2)
    result = result/np.linalg.norm(result, axis=2).reshape(len(position), len(position[0]), -1)
    return (-result)

max_step = 100
width = 256
height = 256
cam_position = np.array([0.,0.,-5.])
ray_position = np.zeros((width, height, 3))
ray_direction = np.zeros((width, height, 3))
color = np.zeros((width, height, 1))
width_half = int(width*0.5)
height_half = int(height*0.5)


uv_u = np.linspace(-1,1,width)
uv_v = np.linspace(-1,1,height)
u,v = np.meshgrid(uv_u, uv_v)
u = u.reshape(width,height,-1)
v = v.reshape(width,height,-1)
uv_z = np.zeros((width, height, 1))
uv = np.concatenate((u,v,uv_z), axis=2)

frame_count = 0
while True:
    speed = 0.1
    light_direction = np.array([math.cos(frame_count * speed),2.,math.sin(frame_count * speed)])
    light_direction = light_direction / np.linalg.norm(light_direction)

    ray_position = np.full((width, height, 3), cam_position)

    ray_direction = uv - ray_position
    ray_direction = ray_direction/np.linalg.norm(ray_direction, axis=2).reshape(width,height,-1)


    dist = np.zeros((width, height, 1))

    for step in range(max_step):
        dist = sdf(ray_position).reshape(width,height,-1)
        ray_position += ray_direction * dist
    normal = get_normal(ray_position)
    diff = np.zeros((width, height, 3))
    diff = normal * light_direction
    diff = np.sum(diff, axis=2).reshape((width, height, -1))

    col = np.where(dist < 0.001, diff, 0)

    cv2.imshow("numpy_version", col)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()