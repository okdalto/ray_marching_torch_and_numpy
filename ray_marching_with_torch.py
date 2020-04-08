import numpy as np
import cv2
import torch
import math

def sdf(position):
    return torch.norm(position, dim=2) - 0.5

def get_normal(position, device, epsilon):
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

    x_neg_sdf = sdf(torch.cat((x_neg, y, z), axis=2)).reshape(w, h, -1)
    y_neg_sdf = sdf(torch.cat((x, y_neg, z), axis=2)).reshape(w, h, -1)
    z_neg_sdf = sdf(torch.cat((x, y, z_neg), axis=2)).reshape(w, h, -1)

    x_pos_sdf = sdf(torch.cat((x_pos, y, z), axis=2)).reshape(w, h, -1)
    y_pos_sdf = sdf(torch.cat((x, y_pos, z), axis=2)).reshape(w, h, -1)
    z_pos_sdf = sdf(torch.cat((x, y, z_pos), axis=2)).reshape(w, h, -1)

    x_result = x_pos_sdf - x_neg_sdf
    y_result = y_pos_sdf - y_neg_sdf
    z_result = z_pos_sdf - z_neg_sdf

    result = torch.cat((x_result,y_result,z_result), axis=2)
    result = result/torch.norm(result, dim=2).reshape(len(position), len(position[0]), -1)
    return (-result)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)


def render(max_step = 100, width = 512, height = 512):

    epsilon = torch.full((width, height, 1), 0.001).to(device)

    color = torch.zeros((width, height, 1))

    uv_u = np.linspace(-1,1,width)
    uv_v = np.linspace(-1,1,height)
    u,v = np.meshgrid(uv_u, uv_v)
    u = u.reshape(width,height,-1)
    v = v.reshape(width,height,-1)
    uv_z = np.zeros((width, height, 1))
    uv = np.concatenate((u,v,uv_z), axis=2)
    uv = torch.from_numpy(uv).to(device)

    cam_position = torch.tensor([0,0,-5]).to(device)
    ray_direction = torch.zeros((width, height, 3)).to(device)
    light_direction = torch.tensor([0., 0., 0.]).to(device)

    diff = torch.zeros((width, height, 3)).to(device)
    zero = torch.tensor(0.0).to(device)

    frame_count = torch.tensor(0).to(device)
    while True:
        speed = 0.1

        #set light direction
        light_direction[0] = torch.cos(frame_count * speed)
        light_direction[1] = 2
        light_direction[2] = torch.sin(frame_count * speed)
        #normalize direction
        light_direction = light_direction / torch.norm(light_direction)

        #set ray position to camera position
        ray_position_x = torch.full((width, height, 1), cam_position[0]).to(device)
        ray_position_y = torch.full((width, height, 1), cam_position[1]).to(device)
        ray_position_z = torch.full((width, height, 1), cam_position[2]).to(device)
        ray_position = torch.cat((ray_position_x, ray_position_y, ray_position_z), dim=2)

        #set ray direction
        ray_direction = uv - ray_position
        #normalize direction
        ray_direction = ray_direction/torch.norm(ray_direction, dim=2).reshape(width, height, -1)

        for step in range(max_step):
            dist = sdf(ray_position).reshape(width,height,-1)
            ray_position += ray_direction * dist
        normal = get_normal(ray_position, device, epsilon)
        diff = normal * light_direction
        diff = torch.sum(diff, axis=2).reshape((width, height, -1))

        color = torch.where(dist < epsilon, diff, zero)

        cv2.imshow("torch_version", color.cpu().data.numpy())
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


render()