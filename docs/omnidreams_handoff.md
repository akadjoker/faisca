# OmniDreams: handoff notes for the next Claude session

Written 2026-11-05 at the end of the first session, for the session on the
owner's other machine. Read docs/omnidreams.md first; this file holds what is
not in it: context, state, and how to work with the owner on this topic.

## Context

- The goal: see how the owner's own renderer can feed NVIDIA OmniDreams (a
  video world model for closed loop driving simulation). The owner explained
  the wider context in the first session; ask again rather than assume, and
  keep it out of this public repo.
- That renderer is Lucent (github.com/akadjoker/lucent, Vulkan 1.3, branch
  `osi-playback` has the OSI trace reader, an OpenDRIVE reader, a car camera
  and a TCP frame stream). Its role: draw the conditioning video (lines and
  boxes on black, the ludus look, spec in docs/omnidreams.md) from OSI and
  OpenDRIVE scenarios, so scenarios can be fed to the model. Possibly also
  the first frame from its realistic mode (untested idea).
- The Lucent repo is public; the owner was told once to check what its README
  names, leave it to them.
- Running OmniDreams inside faisca on the CPU is a later, optional goal (not a
  plan step in CLAUDE.md; the owner has not decided).

## How to talk to the owner about this

- European Portuguese, short, plain words. The owner got lost when the
  explanations were long and technical ("não estou a perceber nada"). What
  worked: a simple analogy (the model is a painter: one photo, one sentence,
  one simple drawing per moment, and it paints the film), a small table or
  diagram, then stop.
- The owner wants progress over investigation for its own sake ("não quero
  perder muito tempo"); keep documents short and practical.
- Key points the owner now understands: the model only makes video, it does
  not decide behaviour (the simulator moves the boxes, the driving software
  under test decides); the real photo is needed once, the drawing every frame.

## State at the end of the session

- Branch `omnidreams` (pushed): docs/omnidreams.md, docs/omnidreams/ (first
  frame, 21 hdmap frames in hdmap_frames.npz, overlay, generated.mp4,
  generated_with_hdmap.mp4), tools/omnidreams/ (Kaggle scripts).
- First video generated on Kaggle T4 (21 frames). Works; one defect: the
  black car on the left of the first frame is smeared away, and its box in the
  hdmap is misplaced. The hdmap over the real frame is close but off (yellow
  line ~100 px left near the hood). Cause unknown. My pose is ego pose times
  `camera_extrinsics` (camera_to_vehicle) passed as FLU; a subagent found the
  world_scenario loader works in RDF (clipgt_loader.py:312-337, 559-571), so
  the extrinsic convention is the first suspect. Check against ludus'
  examples/render_hdmap_scene.py (its own `compute_camera_poses`) at the same
  timestamp; note that example's mp4 path is broken (camera_poses shape) but
  `--frame N` works and saves under `_images/`.
- Kaggle (CLI `~/venvs/kaggle/bin/kaggle` on the first machine; on a new
  machine: `pip install kaggle six`, token from the owner's Kaggle settings into
  `~/.kaggle/access_token`, never paste it in chat):
  - `akadjoker/omnidreams-weights`: its committed output holds the gated
    checkpoint (secrets do not reach API pushed kernels, HTTP 400, so it was
    run once from the web editor). Attach with `kernel_sources`.
  - `akadjoker/omnidreams-step2`: the reference run, version 6 works.
  - Never push to a notebook while the owner has its web editor open.
- Local memory notes on the first machine (not available here): the facts
  above and in docs/omnidreams.md cover them.

## The other machine

- Owner said: NVIDIA GPU with 8 GB VRAM, 64 GB RAM. The GPU model is not
  known yet; ask (it decides bf16 / FP8 support and the attention backend).
- 8 GB is not enough for the stock setup (DiT 4.1 GB + KV cache 4.8 GB + VAE
  encoder peaks 4.4 GB). Plan agreed for now: Lucent runs locally, the model
  keeps running on Kaggle with Lucent's frames uploaded (a Kaggle dataset via
  the API). Fitting 8 GB (weights streamed from RAM per block, shorter window,
  1168x640, 8 bit) only if the owner wants it, each has a cost.
- The text embeddings of the public scene's prompt1 are in the step2 Kaggle
  output (`text_embeddings.pt`, 98 MB, not committed), so the 7B text encoder
  need not run locally.

## Next steps (agreed)

1. Public scene on the machine: `hf download nvidia/omni-dreams-scenes
   scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz --repo-type
   dataset`, unzip; ludus reference frames for camera_front_wide_120fov.
2. Solve the hdmap alignment (see above) before anything is compared.
3. Lucent conditioning mode (spec in docs/omnidreams.md), compared pixel by
   pixel with the ludus frames of the same scene and timestamps.
4. Lucent frames into the model on Kaggle, look at the video.
5. Open questions the owner wanted to find out: what fleet data exists and in
   what format (labels, cameras), what goes wrong in the videos others made
   (conditioning or model), which camera models are used.
