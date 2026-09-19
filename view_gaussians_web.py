"""Browser-based real-time viewer for a trained SpacetimeGaussians model.

Gives proper 3D navigation (three.js orbit/pan/zoom, WASD fly, adjustable up-axis)
instead of the hand-rolled OpenCV mouse handling. Frames are rendered server-side
with the CUDA rasterizer and streamed to the browser, so quality and speed match
the offline renderer.

  python view_gaussians_web.py --ply log/<run>/point_cloud/iteration_<N>/point_cloud.ply

Then open the printed http://localhost:<port> URL.
"""
import sys, os, time, argparse, threading
import numpy as np, torch, viser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.scene.cameras import Camera

ap = argparse.ArgumentParser()
ap.add_argument('--ply', required=True)
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--centre', type=float, nargs=3, default=[0.5, 1.0, 0.42])
ap.add_argument('--radius', type=float, default=9.0)
ap.add_argument('--duration', type=int, default=10)
ap.add_argument('--model', default='ours_lite')
ap.add_argument('--rgbfunction', default='None')
ap.add_argument('--up', default='+y', choices=['+y', '-y', '+z', '-z'])
a = ap.parse_args()

torch.set_grad_enabled(False)
g = getmodel(a.model)(3, a.rgbfunction)
g.load_ply(a.ply)
render, GRs, GRz = getrenderpip('test_ours_lite' if a.model == 'ours_lite' else 'test_ours_full_fused')
bg = torch.zeros(9, dtype=torch.float32, device='cuda')

XYZ_ALL = g.get_xyz.detach().clone()
FULL = {n: getattr(g, n).detach().clone() for n in
        ['_xyz','_features_dc','_opacity','_scaling','_rotation',
         '_trbf_center','_trbf_scale','_motion','_omega'] if getattr(g, n, None) is not None}
CENTRE = torch.tensor(a.centre, dtype=torch.float32, device='cuda')
lock = threading.Lock()

def apply_cull(radius):
    keep = (torch.ones(len(XYZ_ALL), dtype=torch.bool, device='cuda') if radius <= 0
            else torch.linalg.norm(XYZ_ALL - CENTRE, dim=1) < radius)
    for n, v in FULL.items():
        setattr(g, n, torch.nn.Parameter(v[keep], requires_grad=False))
    g.computedtrbfscale = torch.exp(g._trbf_scale)
    g.computedopacity = g.opacity_activation(g._opacity)
    g.computedscales = torch.exp(g._scaling)
    return int(keep.sum())

nvis = apply_cull(0.0)

def quat_to_R(wxyz):
    w, x, y, z = wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

server = viser.ViserServer(port=a.port)
UPVEC = {'+y': (0.,1.,0.), '-y': (0.,-1.,0.), '+z': (0.,0.,1.), '-z': (0.,0.,-1.)}

with server.gui.add_folder('Render'):
    gui_res = server.gui.add_slider('height (px)', 256, 1440, 32, 720)
    gui_fps = server.gui.add_text('fps', initial_value='-', disabled=True)
    gui_n = server.gui.add_text('gaussians', initial_value=f'{nvis}', disabled=True)
with server.gui.add_folder('Time'):
    gui_t = server.gui.add_slider('t', 0.0, 1.0, 1.0/max(a.duration,1)/4, 0.0)
    gui_play = server.gui.add_checkbox('play', False)
with server.gui.add_folder('Scene'):
    gui_cull = server.gui.add_dropdown('floater cull', ('off','25 m','15 m','10 m'), initial_value='off')
    gui_up = server.gui.add_dropdown('up axis', ('+y','-y','+z','-z'), initial_value=a.up)
    gui_reset = server.gui.add_button('reset view')

@gui_cull.on_update
def _(_):
    global nvis
    r = {'off':0.0, '25 m':25.0, '15 m':15.0, '10 m':10.0}[gui_cull.value]
    with lock:
        nvis = apply_cull(r)
    gui_n.value = f'{nvis}'

def place(client):
    client.camera.up_direction = UPVEC[gui_up.value]
    c = np.array(a.centre)
    client.camera.position = tuple(c + np.array([a.radius*0.7, a.radius*0.45, a.radius*0.7]))
    client.camera.look_at = tuple(c)

@gui_up.on_update
def _(_):
    for cl in server.get_clients().values(): cl.camera.up_direction = UPVEC[gui_up.value]

@gui_reset.on_click
def _(_):
    for cl in server.get_clients().values(): place(cl)

@server.on_client_connect
def _(client: viser.ClientHandle):
    print(f'client {client.client_id} connected')
    place(client)

print(f'\n  open  http://localhost:{a.port}   (orbit = drag, pan = right-drag/two-finger, zoom = scroll, WASD to fly)\n')
fps = 0.0
while True:
    clients = server.get_clients()
    if not clients:
        time.sleep(0.05); continue
    if gui_play.value:
        gui_t.value = (gui_t.value + 1.0/(a.duration*8)) % 1.0
    t0 = time.time()
    for client in clients.values():
        cam = client.camera
        H = int(gui_res.value); W = max(64, int(H * cam.aspect))
        fovy = float(cam.fov); fovx = 2*np.arctan(np.tan(fovy/2)*cam.aspect)
        R_c2w = quat_to_R(np.asarray(cam.wxyz, float))
        pos = np.asarray(cam.position, float)
        T = -R_c2w.T @ pos
        with lock:
            if g.ts is None or g.ts.shape[-2:] != (H, W):
                g.ts = torch.ones(1, 1, H, W).cuda()
            c = Camera(0, R_c2w, T, fovx, fovy, (W, H), None, 'web', 0,
                       timestamp=float(gui_t.value))
            img = torch.clamp(render(c, g, None, bg, scaling_modifier=1.0,
                                     basicfunction=trbfunction, GRsetting=GRs, GRzer=GRz)['render'], 0, 1)
        rgb = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        client.scene.set_background_image(rgb, format='jpeg', jpeg_quality=85)
    dt = time.time() - t0
    fps = 0.85*fps + 0.15/max(dt, 1e-6)
    gui_fps.value = f'{fps:.1f}'
