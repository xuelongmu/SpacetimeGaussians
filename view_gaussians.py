"""Interactive real-time viewer for a trained SpacetimeGaussians model.

Uses the CUDA rasterizer that is already built, plus OpenCV's Qt window, so it
needs nothing beyond this venv -- no SIBR, no system packages. The rasterizer
runs at a few hundred FPS on an A6000, so the UI loop is the only cost.

  left-drag   orbit          scroll / +,-   zoom
  right-drag  pan            [ , ]          step time
  space       play/pause     f              cycle floater cull
  r           reset view     s              save screenshot
  q / ESC     quit
"""
import sys, os, time, argparse, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper_train import getmodel, getrenderpip, trbfunction
from thirdparty.gaussian_splatting.scene.cameras import Camera
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov

ap = argparse.ArgumentParser()
ap.add_argument('--ply', required=True)
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=960)
ap.add_argument('--focal', type=float, default=1180.99)
ap.add_argument('--centre', type=float, nargs=3, default=[0.5, 1.0, 0.42])
ap.add_argument('--radius', type=float, default=9.0)
ap.add_argument('--duration', type=int, default=10)
ap.add_argument('--model', default='ours_lite')
ap.add_argument('--rgbfunction', default='None')
a = ap.parse_args()

torch.set_grad_enabled(False)
g = getmodel(a.model)(3, a.rgbfunction)
g.load_ply(a.ply)
render, GRs, GRz = getrenderpip('test_ours_lite' if a.model == 'ours_lite' else 'test_ours_full_fused')
bg = torch.zeros(9, dtype=torch.float32, device='cuda')
g.ts = torch.ones(1, 1, a.height, a.width).cuda()

# Keep the full model; cull is applied per-frame as a view filter so it stays toggleable.
XYZ_ALL = g.get_xyz.detach().clone()
FULL = {n: getattr(g, n).detach().clone() for n in
        ['_xyz','_features_dc','_opacity','_scaling','_rotation',
         '_trbf_center','_trbf_scale','_motion','_omega'] if getattr(g, n, None) is not None}
CULLS = [0.0, 25.0, 15.0, 10.0]

def apply_cull(radius):
    if radius <= 0:
        keep = torch.ones(len(XYZ_ALL), dtype=torch.bool, device='cuda')
    else:
        c = torch.tensor(a.centre, dtype=torch.float32, device='cuda')
        keep = torch.linalg.norm(XYZ_ALL - c, dim=1) < radius
    for n, v in FULL.items():
        setattr(g, n, torch.nn.Parameter(v[keep], requires_grad=False))
    g.computedtrbfscale = torch.exp(g._trbf_scale)
    g.computedopacity = g.opacity_activation(g._opacity)
    g.computedscales = torch.exp(g._scaling)
    return int(keep.sum())

st = dict(azim=0.0, elev=-18.0, rad=a.radius, centre=np.array(a.centre, float),
          t=0.0, play=False, cull=0, drag=None, last=(0, 0), n=len(XYZ_ALL))
st['n'] = apply_cull(CULLS[st['cull']])
fovx, fovy = focal2fov(a.focal, a.width), focal2fov(a.focal, a.height)

def make_cam():
    el, az = np.radians(st['elev']), np.radians(st['azim'])
    eye = st['centre'] + st['rad'] * np.array(
        [np.cos(el)*np.cos(az), np.sin(-el), np.cos(el)*np.sin(az)])
    f = st['centre'] - eye; f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up); n = np.linalg.norm(r)
    r = np.array([1.0, 0, 0]) if n < 1e-6 else r/n
    u = np.cross(f, r)
    Rw2c = np.stack([r, u, f], 0)
    return Camera(0, Rw2c.T, -Rw2c @ eye, fovx, fovy, (a.width, a.height), None,
                  'view', 0, timestamp=float(st['t']))

def on_mouse(ev, x, y, flags, _):
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        st['drag'] = 'orbit' if ev == cv2.EVENT_LBUTTONDOWN else 'pan'; st['last'] = (x, y)
    elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        st['drag'] = None
    elif ev == cv2.EVENT_MOUSEMOVE and st['drag']:
        dx, dy = x - st['last'][0], y - st['last'][1]; st['last'] = (x, y)
        if st['drag'] == 'orbit':
            st['azim'] += dx * 0.4
            st['elev'] = float(np.clip(st['elev'] + dy * 0.3, -89, 89))
        else:
            az = np.radians(st['azim'])
            right = np.array([-np.sin(az), 0, np.cos(az)])
            st['centre'] = st['centre'] + (-dx*right + dy*np.array([0,1.0,0])) * st['rad']*0.002
    elif ev == cv2.EVENT_MOUSEWHEEL:
        st['rad'] = float(np.clip(st['rad'] * (0.9 if flags > 0 else 1.1), 0.2, 400))

WIN = 'SpacetimeGaussians viewer'
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)
print(__doc__)
fps, shot = 0.0, 0
while True:
    t0 = time.time()
    img = torch.clamp(render(make_cam(), g, None, bg, scaling_modifier=1.0,
                             basicfunction=trbfunction, GRsetting=GRs, GRzer=GRz)['render'], 0, 1)
    frame = (img.permute(1, 2, 0).flip(-1).detach().cpu().numpy() * 255).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    cull = CULLS[st['cull']]
    hud = (f"{fps:5.1f} FPS | {st['n']} gauss | t={st['t']:.2f} "
           f"| cull={'off' if cull==0 else f'{cull:g}m'} | r={st['rad']:.1f}m")
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    cv2.imshow(WIN, frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27): break
    elif k == ord(' '): st['play'] = not st['play']
    elif k == ord('['): st['t'] = max(0.0, st['t'] - 1.0/a.duration)
    elif k == ord(']'): st['t'] = min(1.0, st['t'] + 1.0/a.duration)
    elif k == ord('f'):
        st['cull'] = (st['cull'] + 1) % len(CULLS); st['n'] = apply_cull(CULLS[st['cull']])
    elif k == ord('r'):
        st.update(azim=0.0, elev=-18.0, rad=a.radius, centre=np.array(a.centre, float))
    elif k in (ord('+'), ord('=')): st['rad'] *= 0.9
    elif k == ord('-'): st['rad'] *= 1.1
    elif k == ord('s'):
        p = f'viewer_shot_{shot:03d}.png'; cv2.imwrite(p, frame); shot += 1; print('saved', p)
    if st['play']:
        st['t'] += 1.0/(a.duration*4)
        if st['t'] > 1.0: st['t'] = 0.0
    fps = 0.9*fps + 0.1/max(time.time()-t0, 1e-6)
cv2.destroyAllWindows()
