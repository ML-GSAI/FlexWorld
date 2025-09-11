from functools import partial
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from ops.cam_utils import Mcam
from ops.gs.base import GaussianMgr
import os
from ops.PcdMgr import PcdMgr

import numpy as np
from datetime import datetime

PREDEFINED_DIRS = ["./cache", "./results"]

SCANNING_TOP_DIRS = ["./"]

def list_ply_files(directory):
    """列出指定目录中的所有 PNG 文件"""
    try:
        files = [f for f in os.listdir(directory) if f.endswith(".ply")]
        return sorted(files)
    except Exception as e:
        return [f"Error: {e}"]
    

def _sort_folders_by_timestamp(folder_names):
    def extract_timestamp(folder_name):
        try:
            timestamp_part = "_".join(folder_name.split("_")[-2:])
            return datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
        except:
            return None

    sorted_folders = sorted(folder_names, key=lambda x: extract_timestamp(x) or datetime.min, reverse=True)
    return sorted_folders

def list_dirs_mayhave_ply():
    dirs = []
    for root in SCANNING_TOP_DIRS:
        dirs.extend([os.path.join(root, d) for d in _sort_folders_by_timestamp(os.listdir(root))])
    dirs = PREDEFINED_DIRS + dirs
    dirs = [d for d in dirs if os.path.isdir(d) and len(list_ply_files(d)) > 0]
    return dirs

    
def update_dir_list():
    """更新目录列表"""
    return gr.update(choices=list_dirs_mayhave_ply())

def update_file_list(selected_dir):
    """更新文件列表"""
    return gr.update(choices=list_ply_files(selected_dir))

def refresh_all(selected_dir):
    return update_dir_list(), update_file_list(selected_dir)


def load_new_ply(dir, file, flip):
    filepath = os.path.join(dir, file)
    if not GaussianMgr.is_gaussian_ply(filepath):
        gr.Warning("Not a Gaussian ply file, auto init with fixed scale")
        pcd = PcdMgr(ply_file_path=filepath)
        gsmgr.init_from_pts(pcd.pts, mode="fixed", scale=0.0003, opacity=0.95)
    else:
        gsmgr.load_ply(filepath, flip=flip)

    return render()

def create_image(text):
    width, height = 400, 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    return image

def trans_cam(cam, vec):
    direc = np.dot(cam.R, vec)
    direc[1] = 0
    cam.T = cam.T + direc * speed
    return cam

def rot_cam(cam:Mcam, vec):
    ele, azi, _ = cam.get_orbit()
    cam = cam.set_orbit_inplace(ele + vec[0], azi + vec[1])
    return cam

def update_image_stream(key):
    for _ in range(5):
        yield update_image(key)

def update_image(key):
    global cam
    print(key, end=" ")
    trans_key_mapping = {
        "w": np.array([0, 0, -0.01]),
        "a": np.array([-0.01, 0, 0]),
        "d": np.array([0.01, 0, 0]),
        "s": np.array([0, 0, 0.01]),
    }
    transabs_key_mapping = {
        "q": np.array([0, -0.01, 0]),
        "e": np.array([0, 0.01, 0]),
    }
    rot_key_mapping = {
        "i": np.array([-1, 0]),
        "k": np.array([1, 0]),
        "j": np.array([0, 2]),
        "l": np.array([0, -2]),
    }
    if key in trans_key_mapping:
        cam = trans_cam(cam, trans_key_mapping[key])
    elif key in rot_key_mapping:
        cam = rot_cam(cam, rot_key_mapping[key])
    elif key in transabs_key_mapping:
        cam.T = cam.T + transabs_key_mapping[key] * speed
    elif key == "r":
        cam = Mcam()
    return render()
    
def render():
    global cam
    img = gsmgr.render(cam)[0]
    img = img.cpu().numpy()
    img = Image.fromarray( (img * 255).astype("uint8") )
    ratio = 2000 // img.size[0]
    img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio))
    )
    return img


def set_speed(x):
    global speed
    speed = x

def set_fov(x):
    global cam
    cam.setfov(x, axis='x')
    return render()


js = """
<script>
document.addEventListener('keydown', (event) => {
    let key = event.key.toLowerCase();
    if (!['w', 'a', 's', 'd', 'q', 'e', 'i','j','k','l','r'].includes(key)) {
        return;
    }
    const textbox = document.getElementById(key); 
    textbox.click();
});
</script>
"""
demo = gr.Blocks(head=js)
gsmgr = GaussianMgr()


cam  = Mcam()
speed = 1.0

with demo:
    gr.Markdown("## WASD Move Q Down E Up, IJKL Rotate, R reset")
    
    image = gr.Image(type="pil", value=create_image("Press W, A, S, D"), label="3DGS", streaming=True)
    with gr.Row():
        dir_dropdown = gr.Dropdown(
            choices=list_dirs_mayhave_ply(), label="Choose dir", value=PREDEFINED_DIRS[0], interactive=True
        )
        file_dropdown = gr.Dropdown(
            choices=list_ply_files(dir_dropdown.value), label="Choose Ply")
            # 更新文件列表
        with gr.Column():
            check_flip = gr.Checkbox(label="Flip YZ", value=True, visible=False)
            reloadbtn = gr.Button("Reload", elem_id="reload")
            reloadbtn.click(load_new_ply, inputs=[dir_dropdown, file_dropdown, check_flip], outputs=image)
            refreshbtn = gr.Button("Refresh", elem_id="refresh")
            refreshbtn.click(refresh_all, inputs=dir_dropdown ,outputs=[dir_dropdown, file_dropdown])

        dir_dropdown.change(update_file_list, inputs=dir_dropdown, outputs=file_dropdown)
        dir_dropdown.select(update_dir_list, outputs=dir_dropdown)
        
        # 显示图片并压缩
        file_dropdown.change(
            load_new_ply, inputs=[dir_dropdown, file_dropdown, check_flip], outputs=image
        )

        

    speedslide = gr.Slider(minimum=0.1, maximum=100, step=0.1, value=10, label="Move Speed")
    speedslide.change(fn=set_speed, inputs=speedslide)

    fovslide = gr.Slider(minimum=20, maximum=120, step=1, value=60, label="x axis Field of View")
    fovslide.change(set_fov, inputs=fovslide, outputs=image)

    upbtn = gr.UploadButton("Upload a ply", file_count="single", type="filepath")
    upbtn.upload(lambda x: gsmgr.load_ply(x), inputs=upbtn)
    
    for id in ["w", "a", "s", "d", "q", "e", "i", "j", "k", "l","r"]:
        key = gr.Button(visible=False, elem_id=id)
        key.click(fn=partial(update_image_stream, id), outputs=image, show_progress="hidden")



demo.launch(server_port=8000)
