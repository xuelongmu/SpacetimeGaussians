"""Render a free-camera orbit around a trained SpacetimeGaussians model.

The repo only renders the dataset's own viewpoints; this walks an arbitrary
camera path so the solve can be inspected as a 3D scene.
"""
import sys, os, argparse, numpy as np, torch, torchvision
sys.path.insert(0, '/home/zerospace/SpacetimeGaussians')
from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.scene.cameras import Camera
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov

ap = argparse.ArgumentParser()
ap.add_argument('--ply', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=960)
ap.add_argument('--focal', type=float, default=1180.99)   # match the rig's fx
ap.add_argument('--radius', type=float, default=9.0)
ap.add_argument('--centre', type=float, nargs=3, default=[0.5,1.0,0.42])
ap.add_argument('--cull', type=float, default=0.0, help='drop gaussians beyond this radius (display filter)')
ap.add_argument('--height_off', type=float, default=2.5, help='camera height above scene centre (m)')
ap.add_argument('--elev', type=float, default=None)
ap.add_argument('--duration', type=int, default=10)
ap.add_argument('--freeze_time', action='store_true', help='hold t=0 instead of animating')
a = ap.parse_args()

torch.set_grad_enabled(False)
GaussianModel = getmodel('ours_lite')
g = GaussianModel(3, 'None')
g.load_ply(a.ply)
render, GRsetting, GRzer = getrenderpip('test_ours_lite')
bg = torch.zeros(9, dtype=torch.float32, device='cuda')
g.ts = torch.ones(1, 1, a.height, a.width).cuda()

# Anchor on the camera rig, not the gaussians: densification spawns a large halo
# of distant floaters that would otherwise dominate any extent estimate.
centre = np.array(a.centre, float)
radius = a.radius
if a.cull:
    d = torch.linalg.norm(g.get_xyz - torch.tensor(centre, dtype=torch.float32, device='cuda'), dim=1)
    keep = d < a.cull
    print(f'culling floaters: keeping {int(keep.sum())}/{len(d)} gaussians within {a.cull} m')
    for n in ['_xyz','_features_dc','_opacity','_scaling','_rotation','_trbf_center','_trbf_scale','_motion','_omega']:
        if hasattr(g, n) and getattr(g, n) is not None:
            setattr(g, n, torch.nn.Parameter(getattr(g, n).detach()[keep]))
    g.computedtrbfscale = torch.exp(g._trbf_scale)
    g.computedopacity = g.opacity_activation(g._opacity)
    g.computedscales = torch.exp(g._scaling)
print(f'{len(g.get_xyz)} gaussians | orbit centre {centre} radius {radius} m height {a.height_off} m')

fovx = focal2fov(a.focal, a.width); fovy = focal2fov(a.focal, a.height)
os.makedirs(a.out, exist_ok=True)

# Captury world is Y-up; orbit in the XZ plane and look at the centre.
UP = np.array([0.0, 1.0, 0.0])
for i in range(a.frames):
    th = 2 * np.pi * i / a.frames
    eye = centre + np.array([radius*np.cos(th), a.height_off, radius*np.sin(th)])
    f = centre - eye; f /= np.linalg.norm(f)          # forward (+Z of camera)
    r = np.cross(f, UP);  r /= np.linalg.norm(r)      # right  (+X)
    u = np.cross(f, r)                                # down   (+Y), OpenCV convention
    Rw2c = np.stack([r, u, f], 0)
    t = -Rw2c @ eye
    ts = 0.0 if a.freeze_time else (i % a.duration) / a.duration
    cam = Camera(colmap_id=0, R=Rw2c.T, T=t, FoVx=fovx, FoVy=fovy,
                 image=(a.width, a.height), gt_alpha_mask=None,
                 image_name=f'orbit{i:04d}', uid=i, timestamp=ts)
    img = torch.clamp(render(cam, g, None, bg, scaling_modifier=1.0,
                             basicfunction=trbfunction, GRsetting=GRsetting,
                             GRzer=GRzer)['render'], 0.0, 1.0)
    torchvision.utils.save_image(img, os.path.join(a.out, f'{i:04d}.png'))
    if i % 20 == 0: print(f'  {i}/{a.frames}', flush=True)
print('done ->', a.out)
