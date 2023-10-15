import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import predict_bounding_box, preprocess_image_nosave, generate_poses, convert_mesh_format
from heightEstimate.estimate_wild_imgs import estimate_elevation

def preprocess_image(predictor, raw_image, lower_contrast=False):
    raw_image.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_image.convert("RGB"), predict_bounding_box(raw_image))
    input_256 = preprocess_image_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def run_stage1(model, device, experiment_dir, input_image, scale, ddim_steps):
    stage1_directory = os.path.join(experiment_dir, "stage1_8")
    os.makedirs(stage1_directory, exist_ok=True)

    output_images = predict_stage1_gradio(model, input_image, save_path=stage1_directory, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)

    stage2_steps = 50
    zero123_infer(model, experiment_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)

    try:
        elevation = estimate_elevation(experiment_dir)
    except:
        print("Failed to estimate elevation angle")
        elevation = 90
    print("Estimated elevation angle:", elevation)
    generate_poses(experiment_dir, elevation)

    if elevation <= 75:
        output_images_2 = predict_stage1_gradio(model, input_image, save_path=stage1_directory, adjust_set=list(range(4, 8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_images_2 = predict_stage1_gradio(model, input_image, save_path=stage1_directory, adjust_set=list(range(8, 12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90 - elevation, output_images + output_images_2

def run_stage2(model, device, experiment_dir, elevation, scale, stage2_steps=50):
    if 90 - elevation <= 75:
        zero123_infer(model, experiment_dir, indices=list(range(1, 8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, experiment_dir, indices=list(range(1, 4)) + list(range(8, 12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct_3d_mesh(experiment_dir, output_format=".ply", device_index=0, resolution=256):
    experiment_dir = os.path.abspath(experiment_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_index} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {experiment_dir} \
                    --mode export_mesh \
                    --conf confs/Memora_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(experiment_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(experiment_dir, output_format=output_format)

def generate_multiview(shape_directory, args):
    device = f"cuda:{args.gpu_index}"

    models = init_model(device, 'zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    predictor = sam_init(args.gpu_index)
    input_raw = Image.open(args.image_path)

    input_256 = preprocess_image(predictor, input_raw)

    elev, stage1_images = run_stage1(model_zero123, device, shape_directory, input_256, scale=3, ddim_steps=75)
    run_stage2(model_zero123, device, shape_directory, elev, scale=3, stage2_steps=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, default="./demo/demo_examples/01_wild_hydrant.png", help='Path to the input image')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    shape_id = args.image_path.split('/')[-1].split('.')[0]
    shape_directory = f"./exp/{shape_id}"
    os.makedirs(shape_directory, exist_ok=True)

    generate_multiview(shape_directory, args)

    mesh_path = reconstruct_3d_mesh(shape_directory, output_format=args.output_format, device_index=args.gpu_index, resolution=args.mesh_resolution)
    runstage2()
