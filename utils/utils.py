import os
import json
import numpy as np
import cv2
from PIL import Image
from rembg import remove
import trimesh

def predict_foreground_bbox(image):
    image_no_bg = remove(image.convert('RGBA'), alpha_matting=True)
    alpha = np.asarray(image_no_bg)[:, :, -1]
    non_zero_columns = np.any(alpha, axis=0)
    non_zero_rows = np.any(alpha, axis=1)
    x_min, x_max = np.where(non_zero_columns)[0][[0, -1]]
    y_min, y_max = np.where(non_zero_rows)[0][[0, -1]]
    return x_min, y_min, x_max, y_max

def create_image_grid(images, rows, cols):
    assert len(images) == rows * cols
    width, height = images[0].size
    grid = Image.new('RGB', size=(cols * width, rows * height))
    grid_width, grid_height = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * width, i // cols * height))
    return grid

def convert_mesh_format(exp_dir, output_format=".obj"):
    ply_path = os.path.join(exp_dir, "mesh.ply")
    mesh_path = os.path.join(exp_dir, f"mesh{output_format}")
    mesh = trimesh.load_mesh(ply_path)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    mesh.apply_transform(rotation_matrix)
    mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    mesh.faces = np.fliplr(mesh.faces)
    if output_format == ".obj":
        mesh.export(mesh_path, file_type='obj', include_color=True)
    else:
        mesh.export(mesh_path, file_type='glb')
    return mesh_path

def preprocess_image(input_image, lower_contrast=True, rescale=True):
    image_array = np.array(input_image)
    in_width, in_height = image_array.shape[:2]

    if lower_contrast:
        alpha = 0.8
        beta = 0
        image_array = cv2.convertScaleAbs(image_array, alpha=alpha, beta=beta)
        image_array[image_array[:, :, -1] > 200, -1] = 255

    _, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_width
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image_array[y:y + h, x:x + w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)

    rgba_array = np.array(rgba) / 255.0
    rgb = rgba_array[..., :3] * rgba_array[:, :, -1:] + (1 - rgba_array[:, :, -1:])
    return Image.fromarray(rgb * 255).astype(np.uint8)

def calculate_poses(phis, thetas, size, radius=1.2, device='cuda'):
    import torch

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)

    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))

    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device)[:3].unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses

def generate_poses(shape_dir, pose_estimate):
    img_ids, input_poses = get_poses(pose_estimate)

    out_dict = {}
    focal = 560 / 2
    h = w = 256
    out_dict['intrinsics'] = [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
    out_dict['near_far'] = [1.2-0.7, 1.2+0.6]
    out_dict['c2ws'] = {}
    for view_id, img_id in enumerate(img_ids):
        pose = input_poses[view_id]
        pose = pose.tolist()
        pose = [pose[0], pose[1], pose[2], [0, 0, 0, 1]]
        out_dict['c2ws'][img_id] = pose
    json_path = os.path.join(shape_dir, 'pose.json')
    with open(json_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
