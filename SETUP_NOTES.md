# Local setup (uv, torch 2.11 + cu128, RTX A6000)

The upstream `script/setup.sh` targets conda / Python 3.7 / torch 1.12 / CUDA 11.6.
This box runs a uv venv on Python 3.11 with torch 2.11.0+cu128 against nvcc 12.8.

## Use it

    cd ~/SpacetimeGaussians
    source .venv/bin/activate      # or just call .venv/bin/python

## Rebuilding the CUDA extensions

`CUDA_HOME` is not set globally, and uv's default build isolation hides torch from
`setup.py`, so both must be supplied per-command:

    export CUDA_HOME=/usr/local/cuda-12.8
    export TORCH_CUDA_ARCH_LIST="8.6"     # A6000 = sm_86; skips the other archs
    uv pip install --no-build-isolation thirdparty/gaussian_splatting/submodules/<name>

Extensions installed: gaussian_rasterization_ch9, gaussian_rasterization_ch3,
forward_full, forward_lite, simple-knn.

## Local patches required for the newer toolchain

- `helper_model.py` — imports `knn` from `torch_knn` instead of `mmcv.ops`. The pinned
  mmcv submodule does not build against torch 2.x and OpenMMLab's prebuilt wheels stop
  at cu121/torch 2.4. `torch_knn.knn` is a chunked `torch.cdist` with identical
  signature and semantics (returns `(B, k, npoint)` int64, `idx[:, 0]` is self).
  mmcv is not installed; `thirdparty/mmcv` stays uninitialised.
- `simple-knn/simple_knn.cu` — added `#include <cfloat>`. CUDA 12.8 no longer pulls in
  `float.h` transitively, so `FLT_MAX` was undefined.
- `thirdparty/colmap/pre_colmap.py`, `thirdparty/gaussian_splatting/utils/pre_colmap.py`
  — `np.NaN` -> `np.nan`, removed in NumPy 2.0.

## COLMAP

COLMAP 3.13.0 (CUDA 12.9 build) lives in its own mamba env, `~/miniforge3/envs/colmap`,
installed with:

    mamba create -n colmap -c conda-forge 'colmap=3.13.0=cuda_129*'

It is only ever invoked as a CLI (`os.system("colmap ...")`), so it does not need to
share a Python env with anything. `.venv/bin/colmap` is a symlink to it, which means
activating the SpacetimeGaussians venv puts `colmap` on PATH. Upstream's separate
`colmapenv` Python env is not needed: `script/pre_*.py` run fine under `.venv`, which
already has cv2/tqdm/natsort/Pillow.

Verified working: GPU SIFT extraction, exhaustive matching, and `point_triangulator`
with `--Mapper.ba_global_function_tolerance` (the flag most at risk of renaming).

### 3.13-specific patches

- `thirdparty/gaussian_splatting/convert.py` — `--SiftExtraction.use_gpu` ->
  `--FeatureExtraction.use_gpu`, `--SiftMatching.use_gpu` -> `--FeatureMatching.use_gpu`.
  3.13 moved `use_gpu`/`gpu_index` into the `Feature*` namespaces; the rest of the
  `Sift*` options (thresholds, octaves) are unchanged. This file is used by the
  `pre_no_prior.py` path. `helper3dg.py` never passes these and relies on the
  default (=1), so it needed no change.
- `pre_colmap.py` (both copies) — `array.tostring()` -> `tobytes()` and
  `np.fromstring` -> `np.frombuffer`, removed in NumPy 2.0.

### Database schema

3.13 added `rigs`/`frames`/`frame_data`/`pose_priors` and moved the `prior_*` columns
out of `images`. The repo's `COLMAPDatabase.create_tables()` still writes the old
schema, but this is **not** a problem: COLMAP 3.13 migrates such a database in place on
first use. Checked end to end — a repo-written `input.db` extracts, matches, and keeps
the shared camera intrinsics the repo intends.

## Verified end to end

Neural 3D `cook_spinach` (1.2 GB, from the Neural_3D_Video v1.0 GitHub release,
unpacked at ~/data/Neural3D), first 4 frames:

    python script/pre_n3d.py --videopath ~/data/Neural3D/cook_spinach --startframe 0 --endframe 4
    python train.py --quiet --eval --configpath configs/n3d_lite/cook_spinach.json \
        --model_path log/smoketest --source_path ~/data/Neural3D/cook_spinach/colmap_0 \
        --duration 4 --iterations 500
    python test.py --quiet --eval --skip_train --valloader colmapvalid \
        --configpath configs/n3d_lite/cook_spinach.json --model_path log/smoketest \
        --source_path ~/data/Neural3D/cook_spinach/colmap_0 --duration 4 --test_iteration 500

CLI args beat config values (the config only fills in options left at their default),
so `--duration` and `--iterations` shorten a run without editing the json.

Result: 21/21 images registered per frame with ~7.6k triangulated points; training
converged 0.229 -> 0.063 in 500 iters at ~18 it/s; 13,270 gaussians; rendering gave
26.1 dB PSNR / 0.84 SSIM at 1352x1014. Low numbers are expected at 500 of 25,000 iters.

## Further patches found by running the pipeline

- `script/pre_n3d.py`, `script/pre_no_prior.py` — the input images are **hard-linked**
  into `colmap_<n>/input/` rather than symlinked. COLMAP 3.12+ resolves a symlink and
  registers the image under its *target* path, so every image was added twice (42 rows
  for 21 images); the duplicates had no rig/frame entry and `point_triangulator` died in
  `DatabaseCache::Load` with `unordered_map::at`. Hard links cost no extra disk and
  COLMAP sees a real file at the expected name. `shutil.copy` is the fallback if the
  dataset is on another filesystem.
- `thirdparty/gaussian_splatting/helper3dg.py` — 19 sites did
  `exit_code = os.system(...)`, then `exit(exit_code)`. `os.system` returns a *wait
  status*, not an exit code: a COLMAP SIGABRT gives 34816, and `exit(34816)` truncates
  mod 256 to **0**, so crashes were reported as success. Now `sys.exit(1)`.
  Worth knowing: any preprocessing run from before this fix could have "succeeded"
  with a partial reconstruction.
- `test.py` — `sk_ssim(..., multichannel=True)` -> `channel_axis=2, data_range=1.0`.
  scikit-image removed `multichannel`, and without it the 3-channel axis was treated as
  spatial, so SSIM failed with "win_size exceeds image extent".

- `script/pre_n3d.py:168` — `extractframes(v, downscale=downscale)` ignored the
  `--startframe`/`--endframe` arguments, so it always decoded all **300** frames per
  camera even when only a few were wanted. That is ~28 GB of PNGs for a 4-frame test
  (21 cams x 300 frames). The range is now passed through. Note the arguments still
  only bound the COLMAP loop in upstream code; this makes them bound extraction too.

Left alone: `kornia.create_meshgrid` prints a DeprecationWarning per camera. Noisy but
harmless.

## Viewing results

There is **no Linux viewer**. Both viewer archives on the project's HuggingFace repo
(`viewer.zip`, `viewercuda118.zip`, ~109 MB each) are Windows builds -- they contain
MSVC DLLs such as `assimp-vc140-mt.dll`. The README's viewer command is a
`.exe`, and `script/setup.sh`'s build notes are Visual Studio / Windows only.

What works here today: render frames with `test.py` and encode them.

    ffmpeg -y -framerate 30 -pattern_type glob -i 'renders/*.png' \
        -c:v libx264 -pix_fmt yuv420p out.mp4

With `--duration N` the held-out test views are one camera across N timesteps, so the
rendered PNGs are already a time sequence. Verified that consecutive renders differ and
track the ground-truth motion, i.e. the `trbf`/`motion`/`omega` temporal terms are live,
not a frozen scene.

`view_gaussians.py` (added here) is a real-time interactive viewer -- left-drag to
orbit, right-drag to pan, scroll to zoom, `[`/`]` to step time, space to play, `f` to
cycle the floater cull, `s` to screenshot:

    python view_gaussians.py --ply log/<run>/point_cloud/iteration_<N>/point_cloud.ply \
        --centre 0.5 1.0 0.42 --radius 9 --duration 10

It needs nothing outside this venv: it reuses the CUDA rasterizer that is already built
and OpenCV's Qt5 window (`cv2.getBuildInformation()` reports `GUI: QT5` here). Measured
**41.8 FPS at 1280x960** for the whole loop including the GPU->CPU copy and the Qt blit;
the rasterizer alone runs at 290 FPS (3.4 ms), so the display path, not the rendering,
is the limit. This is why the SIBR route was abandoned -- see below.

`orbit_render.py` (added here) walks a free camera around a trained model, which
the repo itself cannot do -- `test.py` only renders the dataset's own viewpoints:

    python orbit_render.py --ply log/<run>/point_cloud/iteration_<N>/point_cloud.ply \
        --out /tmp/orbit --frames 120 --centre 0.5 1.0 0.42 --radius 9 --cull 15

`--centre/--radius/--height_off` define the orbit (world units, Captury world is Y-up);
`--freeze_time` holds t=0 instead of animating over `--duration`. `--cull R` drops
gaussians further than R metres from the centre: a *display* filter, not a model edit.
It matters because densification spawns a large halo of distant floaters -- on the
asteria run only 31% of 410k gaussians sat within 15 m of the rig, with p90 at 204 m,
so any extent estimated from the gaussians is meaningless. Anchor the orbit on the
camera centres instead.

To inspect the *camera* solve rather than the gaussians, COLMAP's own GUI is
interactive and already installed (needs a display; DISPLAY=:1 here):

    colmap gui --import_path <scene>/colmap_0/sparse/0 \
        --database_path <scene>/colmap_0/input.db --image_path <scene>/colmap_0/images

Why not SIBR: there is no prebuilt Linux SIBR. INRIA ships source only, and building it
needs ~308 Ubuntu packages (glew, assimp, boost, gtk3, opencv, embree, ffmpeg dev, ...)
plus merging `realtimedemolite/`'s patched `CudaRasterizer` and `projects` into an SIBR
checkout, against instructions the authors only tested on Windows. Since the rasterizer
already renders far faster than any display can use, `view_gaussians.py` delivers the
same interactivity for none of that. A clone sits at `thirdparty/SIBR_viewers` if the
route is ever revisited; note plain `sibr_core@develop` does NOT contain the
`gaussianviewer` project, so the 3DGS fork or its branch is the right base.

Other options, in rough order of effort:

- **SIBR viewer built from source on Linux.** SIBR itself supports Ubuntu, and
  `thirdparty/gaussian_splatting/realtimedemolite/` holds the patched `CudaRasterizer`
  and `projects` that must be merged into an SIBR checkout. Plausible but the authors
  only tested Windows, so treat it as unverified work, not a recipe.
- **The Windows binary**, on a Windows box with CUDA >= 11.0, against a model directory
  copied over.
- **Generic 3DGS web viewers (SuperSplat, antimatter15/splat, Polycam) will not work
  properly.** The PLY carries 15 non-standard fields -- `trbf_center`, `trbf_scale`,
  `motion_0..8`, `omega_0..3` -- and only `f_dc` colour with no `f_rest` SH. Such a
  viewer can read the position/scale/rotation/opacity columns, but it ignores the
  temporal opacity gating and per-gaussian motion, so every timestep is drawn at once.
  Expect a smeared static blob rather than the scene.

## Sharing the GPU

Training loads all frames into VRAM (24 GB for Neural 3D, 48 GB for Technicolor) on a
49 GB card. Nothing else can be using the GPU meaningfully at the same time.
