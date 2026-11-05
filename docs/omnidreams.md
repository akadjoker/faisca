# OmniDreams: notes for running it and for feeding it from Lucent

Status on 2026-11-04. Exploration, not a plan step yet.

## What it is

NVIDIA OmniDreams (arXiv 2606.03159, code github.com/NVIDIA/flashdreams,
weights huggingface.co/nvidia/omni-dreams-models, gated, NVIDIA Open Model
License). An action conditioned video world model for closed loop driving
simulation. Inputs: the first real camera frame, a text prompt (weather, time of
day), and per frame a drawing of the world (lane lines, road edges, crosswalks,
signs, actors as boxes). Output: photorealistic camera video, 1280x704, 30 fps,
generated in chunks of 8 frames (5 for the first chunk).

## Model (read from flashdreams 34ce7b7 and the checkpoint)

- DiT from Cosmos-Predict2.5 2B: 28 blocks, 2048 channels, 16 heads of 128,
  GELU MLP 8192, LayerNorm without affine modulated by AdaLN-LoRA (rank 256),
  RMS norm on Q and K, 3D RoPE (head dim split 2:2:2 over t, h, w, NTK
  extrapolation 3.0 on h and w), cross attention to 512 text tokens.
  Checkpoint `single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt`:
  4118900683 bytes, sha256 c7cb6f0c...dd62f, 570 bf16 tensors, 2.059 B
  parameters, loads with `torch.load(weights_only=True)`.
- Input tokens: 16 latent channels + condition mask + padding mask, patch 2x2
  (x_embedder 72 -> 2048). The HD map video goes through the VAE encoder to 16
  channels and has its own patch embedding (64 -> 2048), added.
- Text: nvidia/Cosmos-Reason1-7B (Qwen2.5-VL-7B text part, not gated, 16.6 GB
  bf16), fixed system prompt, padded to 512 tokens, all 28 hidden states
  normalized and concatenated (512 x 100352), projected to 1024 by the DiT.
  Computed once per prompt.
- VAEs from lightx2v/Autoencoders: lightvaew2_1 (encoder, 32 MB) and
  lighttaew2_1 (TAEHV decoder, 45 MB). 8x spatial, 4x temporal.
- Sampling: flow matching, 2 steps (timesteps 1000 and 500, shift 5), then one
  more forward at noise 128 to write the KV cache: 3 DiT passes per chunk.
  Chunk = 2 latent frames = 7040 tokens; the self attention window keeps 6
  latent frames (21120 keys).
- Cost estimate (not measured): ~50 TMAC for the first chunk, ~87 TMAC with a
  full window, attention larger than the matmuls once the window is full.
  NVIDIA: 68 fps on one GB300.

## Kaggle setup (works)

- CLI: `~/venvs/kaggle/bin/kaggle`, token in `~/.kaggle/access_token`.
- Machine: 2x T4 (sm_75, 15 GB), 4 vCPU, 31 GB RAM, CUDA 13.0, Python 3.13.
- Kaggle secrets do not reach kernels pushed through the API. The gated
  checkpoint therefore lives in the output of notebook
  `akadjoker/omnidreams-weights` (run once from the web editor) and other
  kernels attach it with `"kernel_sources": ["akadjoker/omnidreams-weights"]`.
- Do not push to a notebook while its web editor is open (draft conflict).
- `akadjoker/omnidreams-step2` builds the reference: uv env with Python 3.12
  (`apt-get install yasm` first, flashdreams-webm needs it), torch 2.12.1
  cu130, attention cuDNN/Flash replaced by `F.scaled_dot_product_attention`
  (T4 has neither), CUDA graphs off, text encoder on the CPU, local checkpoint
  path. The ClipGT folder must be staged with `prepare_clipgt_dir`. ludus needs
  `LIBRARY_PATH` with a `libcuda.so` link to `/usr/local/cuda-13.0/compat`.
- Scripts in tools/omnidreams, in the order they were run: `recon/` (machine
  probe and ludus build test; rebuilt from the session, the Kaggle copy was
  overwritten), `secret_test/` (shows secrets fail for API pushes; same
  notebook id as recon), `weights/` (the download notebook), `step1/`
  (checkpoint check, tensor list, scene download, ludus example render) and
  `step2/`
  (`driver.py` is the reference; `step2_run.py` embeds it for Kaggle, push
  with `kaggle kernels push -p tools/omnidreams/step2 --accelerator
  NvidiaTeslaT4`). On a local NVIDIA machine run `driver.py` directly inside
  the flashdreams uv env with STEP_OUT, STEP_CHECKPOINT, STEP_SCENE set (with
  sm_80 or newer the attention patch is not needed). The driving loop is
  `initialize_cache_from_embeddings`, then per chunk `generate(i, cache,
  hdmap)` and `finalize(i, cache)`, hdmap pixels in [-1, 1].

## First result (Kaggle, step2 v6, 2026-11-04)

Public scene, prompt1 (clear midday), 3 chunks = 21 frames, files in
docs/omnidreams/ (`generated.mp4`, `generated_with_hdmap.mp4` with the hdmap
below, `overlay_0000.png` hdmap over the real first frame).

- T4, bf16, plain SDPA in blocks of 1024 queries, hdmap VAE encoder on the
  second T4 (on one T4 it runs out of memory: DiT 3.9 GiB + KV cache 4.8 GiB
  + the encoder's conv3d 4.4 GiB). Peak 12.5 GiB on GPU 0.
- Per chunk generate / finalize: 39 / 17 s, 53 / 24 s, 67 / 32 s (grows with
  the window). Text encode on the CPU 177 s, pipeline setup 4 min (downloads).
- The output looks right: the street, the trees and the parked cars follow
  the first frame. The black car on the left of the first frame is smeared away
  in the first chunk; its box in the hdmap does not sit on it (see below).
- Alignment of the hdmap over the real frame is close but not exact: the
  drawn yellow line is about 100 px left of the painted one near the hood, and
  the left car's box is low and left.
- Cause (found 2026-11-05, not yet rerun): the driver multiplied
  `scene.ego_poses` by `scene.camera_extrinsics`; the world_scenario loader
  converts both to OpenCV RDF (clipgt_loader.py 312-337 and 560-579), ludus
  reads the product as FLU. Read back from hdmap_frames.npz the drawing camera
  sat 1.45 m right of the ego and 3.6 m up, and rose to 13 m over the 21
  frames instead of driving 3.6 -> 13 m forward at 1.45 m height. So the
  conditioning of that first video was wrong, which fits the smeared car.
- Fix in step2/driver.py: rig to world from the ludus ego track times ludus'
  `sensor_to_rig` (nominalSensor2Rig_FLU, Euler "xyz", times
  correction_sensor_R_FLU), the way apps/interactive_drive rasterizer.py
  `_render_one_camera` does; frames at the photo's timestamp plus k / 30 s
  instead of the next 30 Hz ego sample (that was 23 ms late).
- Calibration of the front camera in this scene: nominal rpy (0.29, 0.46,
  -0.19) deg, t (1.690, 0.006, 1.453) m; correction rpy (-0.16, 0.12, 0.50)
  deg (yaw 0.5 deg is about 5 px at 1280 wide); correction_rig_T
  (-0.057, -0.003, 0.009) m, which no ludus loader applies.

## The conditioning drawing (ludus), what Lucent has to reproduce

Lucent (github.com/akadjoker/lucent, branch `osi-playback`) already has the
inputs (OSI trace playback, OpenDRIVE reader) and the output (car camera, TCP
frame stream). What it needs is one more render mode.

Spec read from flashdreams 34ce7b7, `integrations_v2/omnidreams/impl/`:
`ludus-renderer/ludus_renderer/_cpp/render/ludus_cuda.cu` (CU),
`ludus-renderer/ludus_renderer/clipgt.py` (CG), `conditioning/renderer.py`,
`conditioning/world_scenario/ftheta.py`. Line numbers in brackets.

Image and pass
- 1280x704 (width and height multiples of 8, or the NDC mapping stretches),
  uint8 RGB, rows top-down, clear black. No lighting, shadows, TAA, tonemapping
  or exposure.
- Z-buffer on, smaller depth wins; no backface culling in the rasterizer;
  polygons, cubes and lines share one depth buffer with no bias, except the
  cube wireframe (-0.001 in NDC z). Equal depth: the triangle whose packed
  vertex color 0xFFBBGGRR is larger wins.
- Color and z interpolated linearly in screen space (no perspective
  correction).
- Fog: every color is multiplied by `clamp(1 - z / 200, 0, 1)` and rounded
  (+0.5) [CU 2479-2488]. So the pixels are NOT the plain palette: things fade to
  black at 200 m. z is the camera depth when the camera's max ray angle is at
  most 90 degrees, the ray length otherwise.
- No supersampling at this size (skipped past 2048 px a side), but both
  callers (conditioning/renderer.py LudusRenderer, apps/interactive_drive
  rasterizer.py) call `set_msaa_samples(4)` and `set_depth_scaling(True)`; what
  that MSAA does in the CUDA rasterizer is not read yet.
- The two callers differ: LudusRenderer uses cube tessellation 0 and the
  default wireframe width, interactive_drive cube tessellation 3 (box faces
  curve under f-theta) and `set_line_widths(..., wireframe=4.0)`.
- Spatial cull at 300 m around the camera; beyond 200 m things are drawn black
  with zero width.

Camera
- Pose in: camera-to-world, FLU (x forward, y left, z up). Inverted, then FLU
  to RDF: x_cam = -y, y_cam = -z, z_cam = x (+z forward, y down).
- F-theta [CU 55-140]: p = view * X; theta = acos(z / |p|); r = sum fw_poly[i]
  theta^i (6 coefficients, fitted from the bw poly with constant 0), and for
  theta > max_ray_angle r = poly(max) + (theta - max) * poly'(max);
  u = cx + r x / |xy|, v = cy + r y / |xy| (r / |xy| = 0 when |xy| <= 1e-6).
  Pixel centres at integer + 0.5. max_ray_angle = largest bw angle over the 4
  image corners. No linear distortion term.

Lines (screen space ribbons)
- Width in pixels: 7 (lines, edges, stop lines), 5 (poles), times the same
  depth factor `clamp(1 - z / 200, 0, 1)` per vertex.
- Offset per vertex by the perpendicular in pixels; miter at the original
  interior vertices (1 / cos(half angle), capped at 2); round caps (6 segment
  half disc) at both ends of every polyline, so every dash has round caps.
- Tessellation for the fisheye: a segment is split into 2 / 4 / 8 / 16 equal
  world parts when the pixel distance between the projected midpoint and the
  midpoint of the projected ends exceeds 1 / 4 / 16 / 64. Only the first 256
  vertices of a polyline are used.

Map elements from ClipGT (lane_boundary, intersection and road_island are
excluded by default)
- road_boundary (253,1,232); wait_line (108,179,59); pole (183,69,177), a
  single point becomes a 3 m vertical line.
- Lane lines: white (255,255,255) or yellow (255,255,0) by colors[0], style by
  styles[0]. Dashes in world metres from the start of the line, after
  resampling every 0.2 m: long 3 on 9 off, short 1/1, white dotted 0.5/1.0.
  Double lines (solid+dashed, two solids) offset +-0.075 m. Yellow dots: world
  points drawn as screen hexagons of radius 3.5 px times the depth factor,
  every 1.5 m (0.5 m for other DOT styles).
- Polygons, fan triangulated (0, j, j+1) from the raw outline: crosswalk
  (139,93,255), road_marking (20,254,185); subdivided into 4 / 16 / 64
  triangles at errors 1 / 2 / 4 px.

Boxes
- Unit cube times the full size, pose = quaternion + centre, no subdivision.
- Front is local +x. Corner color = back + (x + 0.5) (front - back): the +x
  face is the front color, -x the back color, the four other faces a gradient.
- Per class front / back: car (0,46,136) / (126,206,255), truck (204,55,0) /
  (255,192,64), pedestrian (148,0,62) / (255,124,171), cyclist (0,80,66) /
  (102,208,198), other (53,26,20) / (166,136,125). Traffic light (100,100,100)
  and sign (8,2,255) are static boxes (default sizes 0.3x0.3x0.5 and
  0.01x0.5x0.8), no light state colors.
- A face is drawn when it faces the camera (or the camera is inside). All 12
  edges get a wireframe (200,200,200), 2 px times the depth factor, no caps.
- Actor poses between samples: position lerped and extrapolated up to 0.5 s,
  rotation slerped and held at the ends.

Still open: whether the front camera's max ray angle
is above 90 degrees, the rasterizer fill rule beyond the 1/16 px snap.

Lucent changes:

1. A separate pass that writes straight to the output image, none of the
   realistic chain, with the depth fog above and no MSAA.
2. From OpenDRIVE, typed polylines instead of road marking meshes: white or
   yellow, solid or dashed (roadMark), outer road edges, stop lines, crosswalk
   polygons, poles and signs (objects, signals). The reader handles `line` and
   `arc` only; real maps also need `spiral`, `poly3`, `paramPoly3`, elevation.
3. OSI actors as unlit cuboids with front and back colors.
4. F-theta projection in the mesh shader, with subdivision of long edges.
5. Axis conversion: Lucent is Y up with Z = -y of OpenDRIVE; ludus is FLU.

Validation: draw the public scene (nvidia/omni-dreams-scenes, ClipGT
clipgt-0d404ff7) in both and compare pixel by pixel. Needs a ClipGT reader in
Lucent, or the other way round: an OpenDRIVE/OSI scenario converted to ClipGT
and drawn by ludus as the reference.

## Next, on the owner's NVIDIA machine

1. Run the reference locally (the step2 driver) and check: the hdmap overlay on
   the first real frame, the generated video, timings.
2. ludus reference frames of the public scene for one camera.
3. Lucent conditioning mode, compared with those frames.
4. Later, faisca: `.pt` to GGUF, LightTAE, DiT layer by layer against the
   step2 dumps, LightVAE, the text encoder through the Qwen2.5 path.
