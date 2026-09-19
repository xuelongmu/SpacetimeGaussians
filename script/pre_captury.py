"""Build a SpacetimeGaussians dataset (technicolor layout) from a Captury Live take
plus a multical/Captury camera calibration.

Captury records one AVI per camera plus a `stream<NN>.txt` sidecar listing
`timestamp_us;frame_id` per slot, where `-1;-1` marks a frame the camera never
delivered. Those gaps are common and differ per camera, so the slots where *every*
camera has a frame must be found before choosing timesteps -- see --list-slots.

Camera order: stream<NN>.avi <-> camera.calib camera NN <-> meta.txt CameraUids[NN].
Captury reports serials as (multical serial + 2**25).

Example:
  python script/pre_captury.py --take /path/09190138_JonXuelong \
      --calib live-sessions/<session>/evaluation.json \
      --out ~/data/asteria/JonXuelong --start 0 --frames 10
"""
import sys, os, json, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thirdparty.colmap.pre_colmap import COLMAPDatabase


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx-Ryy-Rzz, 0, 0, 0],
        [Ryx+Rxy, Ryy-Rxx-Rzz, 0, 0],
        [Rzx+Rxz, Rzy+Ryz, Rzz-Rxx-Ryy, 0],
        [Ryz-Rzy, Rzx-Rxz, Rxy-Ryx, Rxx+Ryy+Rzz]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def slot_availability(take, ncam):
    """Per-camera boolean mask of slots that actually hold a frame."""
    valid = {}
    for i in range(ncam):
        rows = [l.split(';') for l in open(f'{take}/stream{i:02d}.txt') if l.strip()]
        valid[i] = np.array([int(r[1]) >= 0 for r in rows])
    return valid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--take', required=True, help='Captury take directory')
    ap.add_argument('--calib', required=True, help='multical evaluation.json')
    ap.add_argument('--out', required=True, help='output dataset root')
    ap.add_argument('--start', type=int, default=0, help='first slot to use')
    ap.add_argument('--frames', type=int, default=10, help='number of timesteps')
    ap.add_argument('--ncam', type=int, default=25)
    ap.add_argument('--list-slots', action='store_true',
                    help='report slots where every camera has a frame, then exit')
    a = ap.parse_args()

    valid = slot_availability(a.take, a.ncam)
    allv = np.ones(len(valid[0]), bool)
    for i in range(a.ncam):
        allv &= valid[i]
    slots = np.where(allv)[0]
    print(f'slots with all {a.ncam} cameras present: {len(slots)} of {len(allv)}')
    if a.list_slots:
        print(slots.tolist())
        return
    if len(slots) == 0:
        sys.exit('No slot has every camera. Pick a camera subset or use --list-slots.')
    want = [s for s in slots if s >= a.start][:a.frames]
    if len(want) < a.frames:
        sys.exit(f'Only {len(want)} usable slots at/after {a.start}; need {a.frames}.')
    print('using slots:', want)

    r = json.load(open(a.calib))['result']
    assert r['units'] == 'metres' and r['transform_convention'] == 'world_to_camera', \
        'unexpected calibration convention'
    idx2ser = {v['index']: s for s, v in r['provenance']['cameras'].items()}
    cams = {}
    for i in range(a.ncam):
        s = idx2ser[i]
        M = np.array(r['camera_poses'][s], float)
        cams[i] = dict(K=np.array(r['cameras'][s]['K'], float),
                       dist=np.array(r['cameras'][s]['dist'], float),
                       W=r['cameras'][s]['image_size'][0],
                       H=r['cameras'][s]['image_size'][1],
                       R=M[:3, :3], t=M[:3, 3])

    out = a.out
    for t in range(len(want)):
        os.makedirs(f'{out}/colmap_{t}/input', exist_ok=True)
        os.makedirs(f'{out}/colmap_{t}/manual', exist_ok=True)

    # Undistort to PINHOLE up front: the technicolor loader requires it, and it keeps
    # COLMAP from having to model distortion it did not estimate.
    for i in range(a.ncam):
        c = cv2.VideoCapture(f'{a.take}/stream{i:02d}.avi')
        K, dist, W, H = cams[i]['K'], cams[i]['dist'], cams[i]['W'], cams[i]['H']
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)
        grab, slot = {s: n for n, s in enumerate(want)}, 0
        while slot <= want[-1]:
            ok, f = c.read()
            if slot in grab:
                if not ok:
                    sys.exit(f'cam{i:02d} slot {slot} failed to decode')
                cv2.imwrite(f'{out}/colmap_{grab[slot]}/input/cam{i:02d}.png',
                            cv2.remap(f, mx, my, cv2.INTER_LINEAR))
            slot += 1
        c.release()
        print(f'  cam{i:02d} done', flush=True)

    for t in range(len(want)):
        proj = f'{out}/colmap_{t}'
        if os.path.exists(f'{proj}/input.db'):
            os.remove(f'{proj}/input.db')
        db = COLMAPDatabase.connect(f'{proj}/input.db')
        db.create_tables()
        imtxt, camtxt = [], []
        for i in range(a.ncam):
            c, cid = cams[i], i + 1
            fx, fy, cx, cy = c['K'][0, 0], c['K'][1, 1], c['K'][0, 2], c['K'][1, 2]
            q, T = rotmat2qvec(c['R']), c['t']
            fn = f'cam{i:02d}.png'
            imtxt.append(f"{cid} " + " ".join(map(str, q)) + " " +
                         " ".join(map(str, T)) + f" {cid} {fn}\n\n")
            camtxt.append(f"{cid} PINHOLE {c['W']} {c['H']} {fx} {fy} {cx} {cy}\n")
            db.add_camera(1, c['W'], c['H'], np.array([fx, fy, cx, cy]))
            db.add_image(fn, cid, prior_q=q, prior_t=T, image_id=cid)
        db.commit(); db.close()
        open(f'{proj}/manual/images.txt', 'w').write("".join(imtxt))
        open(f'{proj}/manual/cameras.txt', 'w').write("".join(camtxt))
        open(f'{proj}/manual/points3D.txt', 'w').write("")
    print(f'wrote {len(want)} colmap_* projects with {a.ncam} cameras each -> {out}')
    print('next: run getcolmapsinglen3d() per timestep, or the colmap CLI chain '
          '(feature_extractor / exhaustive_matcher / point_triangulator / image_undistorter)')


if __name__ == '__main__':
    main()
