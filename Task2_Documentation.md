# Task 2: Video Object Segmentation with Memory (SAM 2)

This document explains the Task 2 cell of `AVP_Project.ipynb`: what it adds to the Task 1 pipeline, how the code works step by step, and what it produces.

---

## 1. Where Task 2 fits in the pipeline

Task 1 pipeline:

```
video ──► background model (median, BG0) ──► detection (background subtraction + morphology + connected components)
      ──► Euclidean centroid tracker ──► synopsis (rectangular crops pasted on the reference frame)
```

Task 2 **extends** the pipeline. It does not replace it. It improves the **synopsis generation** stage:

```
Task 1 tracks ──► box prompt at first appearance ──► SAM 2 (memory-based propagation)
             ──► pixel-accurate mask of the object in every frame of a short clip
             ──► synopsis made of segmented, moving cut-outs on a colour reference frame
```

| Stage                | Task 1                                           | Task 2 (with SAM 2)                                              |
|----------------------|--------------------------------------------------|------------------------------------------------------------------|
| What is pasted       | rectangular crop (object + floor around it)      | only the object's pixels (mask cut-out)                          |
| Object shape         | blob from thresholding, holes and shadows        | learned segmentation mask, follows the real object outline       |
| Colour               | grey-scale                                       | colour frames and colour reference frame                         |
| Time                 | one still crop per object                        | the object is followed for `clip_len` frames and moves in its cell |
| Synopsis output      | a few still canvases                             | an animated synopsis video, objects added one by one             |

Task 1 still supplies the **background model, the detections and the tracks**. SAM 2 only needs one box per object, and it takes that box from the Task 1 tracker. So Task 1 tells SAM 2 *what* to segment, and SAM 2 tells the synopsis *exactly which pixels* belong to the object.

---

## 2. The model: SAM 2

- **Paper:** N. Ravi et al., *"SAM 2: Segment Anything in Images and Videos"*, Meta AI, 2024. arXiv:2408.00714
- **Code:** <https://github.com/facebookresearch/sam2> (Apache 2.0 license)
- **Checkpoint used:** `sam2.1_hiera_small.pt` (SAM 2.1, Hiera-Small image encoder), downloaded from Meta's official link
- **Config used:** `configs/sam2.1/sam2.1_hiera_s.yaml` (shipped with the `sam2` package)

We use the pretrained model as it is, with no training or fine-tuning.

### Why SAM 2 fits the topic "Video Object Segmentation with Memory"

- **Semi-supervised VOS:** the object is annotated only on the first frame (here with a box prompt), and the model segments it in every following frame by itself.
- **Prompt-based segmentation:** the annotation is a *prompt* (box, points or mask) instead of a full ground-truth mask.
- **Memory bank:** for every new frame, SAM 2 runs *memory attention*. The features of the current frame attend to stored memories of earlier frames: the prompted frame plus the most recent frames, together with *object pointers* (one token per frame that sums up the object). The mask is decoded from these memory-conditioned features, and then the memory encoder turns the new frame and its predicted mask into a memory that is stored for the next frames.
- **Bounded memory:** the memory the model reads from works like a FIFO queue with a fixed size. The SAM 2 configs use `num_maskmem = 7`, which means the prompted frame + the 6 most recent frames (`sam2/modeling/sam2_base.py`). Object pointers are taken from at most 16 past frames (`max_obj_ptrs_in_encoder = 16`). The prompted frame is always kept because it holds the only user-given information, and older non-prompted frames drop out of the window. So the attention cost per frame stays constant however long the video is. If the memory grew without limit, GPU memory and attention cost would grow linearly with the video length, and old, outdated appearances (different pose, lighting, scale) would make the matching noisier, so errors would build up.

---

## 3. How to run

1. Install the extra packages (once):
   ```bash
   pip install torch torchvision
   pip install "git+https://github.com/facebookresearch/sam2.git"
   ```
2. Run the **Task 1 cell** first. Task 2 reuses its variables and functions.
3. Run the **Task 2 cell**. The SAM 2 checkpoint (~180 MB) is downloaded automatically the first time.

A CUDA GPU is used when available (bfloat16 autocast). On CPU it still works, just much slower. The notebook needs several GB of RAM (Task 1 alone keeps all 1330 grey frames in memory, about 2.7 GB).

### What Task 2 uses from Task 1

| Name                          | Used for                                                      |
|-------------------------------|---------------------------------------------------------------|
| `frames1`, `BG0_1`            | re-running detection to get each track's first bounding box    |
| `detect_objects()`            | the Task 1 detector (box + cleaned blob mask)                 |
| `valid_tracks`                | objects to segment (tracks with at least 10 frames)           |
| `get_BG0()`                   | median background, here applied to colour frames              |
| `writeVideo()`                | turning the saved `.png` frames into the `.mp4` video         |
| `fps`, `grid_cols`, `grid_rows`, `cell_w`, `cell_h` | timestamps and the synopsis grid layout   |

---

## 4. Code walkthrough

The cell starts with the required header (`#Task 2`, `#Task requirement`, `#Task overview`), then runs these steps.

### Settings

| Variable       | Value | Meaning                                                              |
|----------------|-------|----------------------------------------------------------------------|
| `clip_len`     | 60    | number of frames SAM 2 follows each object (60 frames = 2 s at 30 fps) |
| `clear_ratio`  | 0.5   | a frame is "clear" if the object's mask is at least 50 % of its largest mask in the clip |
| `min_object_area` | 10000 | objects whose largest mask is smaller than this (pixels) are skipped as border fragments / noise (set to 0 to keep every object) |
| `add_every`    | 15    | a new object appears on the synopsis canvas every 15 frames          |
| `clip_dir`     | `Results/Task2/clip_frames` | temporary JPEG frames given to SAM 2 (deleted at the end) |
| `synopsis_dir` | `Results/Task2/Synopsis`    | output synopsis frames                                    |
| `compare_dir`  | `Results/Task2/Comparison`  | Task 1 vs SAM 2 comparison images                         |

### Step 1: Load SAM 2

The checkpoint is downloaded with `urllib.request.urlretrieve` if it is not on disk yet. Then `build_sam2_video_predictor(config, checkpoint, device)` builds the video predictor.

### Helper functions

- **`read_color_frames(video_path, start, count)`**: opens the video with OpenCV, jumps to frame `start` and reads `count` colour (BGR) frames. Task 1 keeps only grey frames, but SAM 2 needs colour input.
- **`first_box_of_track(track)`**: a Task 1 track only stores centres, not boxes. This function runs the Task 1 detector again on the frame where the track started and returns the detection whose centre is closest to the track's first centre. That is the box that created the track.
- **`segment_track(track, box)`**: the core SAM 2 part:
  1. reads the clip that starts at `track.first_frame` (at most `clip_len` frames, never past the track's last frame),
  2. writes it as `00000.jpg, 00001.jpg, ...`, the video format the SAM 2 predictor reads,
  3. `predictor.init_state(...)` encodes the clip (`offload_video_to_cpu=True` keeps the frames in RAM so the 8 GB GPU is enough),
  4. `predictor.add_new_points_or_box(..., frame_idx=0, box=...)` gives the Task 1 box as the prompt on the first frame,
  5. `predictor.propagate_in_video(state)` goes through the clip frame by frame. Each frame is segmented by reading from the memory bank, and the mask logits are thresholded at 0 to get a binary mask,
  6. `predictor.reset_state(state)` clears the memory before the next object.
- **`mask_iou(a, b)`**: intersection over union of two binary masks.

### Steps 2 to 5: Segment every Task 1 track

One loop goes over `valid_tracks`. For each track it gets the first box (`first_box_of_track`), runs `segment_track` (steps 2 and 3), compares the result with Task 1 (step 4) and prepares the cut-outs (step 5).

#### Step 4: Compare with the Task 1 blob

On the first frame of every track, the Task 1 blob (the cleaned background-subtraction mask inside the track's box) is compared with the SAM 2 mask:

- the **IoU** is printed per track, and the mean IoU over all tracks is printed at the end,
- an image `Results/Task2/Comparison/track_XXX.png` is saved with three panels:
  **Task 1 rectangular crop | Task 1 blob mask | SAM 2 cut-out**.

This is done for every track, also the ones skipped in step 5, so the mean IoU covers all 19 tracks.

#### Step 5: Keep the clear frames and skip fragments

The mask size (`areas`, number of object pixels per frame) shows how visible the object is:

1. **Skip fragments:** if the largest mask in the clip is smaller than `min_object_area`, the "object" is only a sliver at the image border (for example the near player's head entering at the bottom edge) or a small noise blob, so it is not pasted.
2. **Keep the clear frames:** many Task 1 tracks start while the object is still entering the view. SAM 2 keeps following it as it walks in (the memory bank carries the object from the tiny first box to the full body), so the mask grows over the clip. Only frames whose mask is at least `clear_ratio` × the largest mask are kept, the same idea as the "first clear appearance" in Task 1.
3. **Union box:** the box around all kept masks is computed, and every kept frame and mask is cropped to it. All cut-outs of one object then have the same size, so the object moves naturally inside its window and does not jitter in the synopsis. The crops are `.copy()`-ed because a NumPy slice is only a view. Without the copy, every small cut-out would keep its whole 1920×1080 frame in memory (about 500 MB per object), and the notebook ran out of RAM during our tests.

### Step 6: Build the synopsis with cut-outs

- **Reference frame:** every 10th colour frame is read and passed to the Task 1 function `get_BG0`, which takes the per-pixel median. This is the same background idea as Task 1, but in colour. It is saved as `Results/Task2/reference_frame.png`.
- **Canvases:** objects are put into canvases of `grid_cols × grid_rows` (4 × 3 = 12) cells in order of first appearance, the same layout as Task 1. When a canvas is full, a new one starts from a clean reference frame.
- **Animation:** a canvas lasts `(objects − 1) × add_every + longest` frames, where `longest` is the number of kept frames of the longest object on that canvas. Object number `slot` appears at frame `slot × add_every` (objects are added one by one) and then loops through its segmented frames.
- **Pasting:** each cut-out is scaled to fit 80 % of its cell, keeping its aspect ratio, and centred in the cell. Only the pixels inside the SAM 2 mask are copied (`region[msk] = cut[msk]`), so the reference frame stays visible around the object. The mask is resized with nearest-neighbour interpolation so it stays binary.
- **Timestamp:** the time of first appearance (`first_frame / fps`, in seconds) is written above each object in green with a black outline so it is readable on any background.
- Each canvas frame is saved as `Results/Task2/Synopsis/synopsis_XXXX.png`.

### Step 7: Output video

The Task 1 function `writeVideo` turns the saved `.png` frames into `tabletennis_synopsis_sam2.mp4` at the original video's frame rate.

---

## 5. Outputs

| File / folder                               | Content                                                    |
|---------------------------------------------|------------------------------------------------------------|
| `tabletennis_synopsis_sam2.mp4`             | **Task 2 output video**: animated synopsis with SAM 2 cut-outs |
| `Results/Task2/Synopsis/synopsis_XXXX.png`  | every frame of the synopsis video                          |
| `Results/Task2/Comparison/track_XXX.png`    | Task 1 crop / Task 1 blob / SAM 2 cut-out for each object  |
| `Results/Task2/reference_frame.png`         | colour median reference frame                              |
| `sam2.1_hiera_small.pt`                     | downloaded SAM 2 checkpoint (not part of the results)      |

---

## 6. Results

Run on `tabletennis_short.mp4` (1920×1080, 30 fps, 1330 frames) with an NVIDIA RTX 5060 Laptop GPU (8 GB).

**Task 1 input:** the centroid tracker created 42 tracks, and 19 of them are valid (at least 10 frames).

**Agreement between Task 1 blobs and SAM 2 masks (first frame of each track):**

| Track | Frames  | IoU  | In synopsis? |
|------:|---------|-----:|--------------|
| 1     | 0–59    | 0.77 | yes |
| 2     | 41–100  | 0.71 | yes |
| 3     | 41–100  | 0.80 | yes |
| 12    | 166–225 | 0.88 | yes |
| 14    | 187–214 | 0.69 | no (largest mask 3812 px) |
| 15    | 238–297 | 0.69 | no (2256 px) |
| 16    | 304–342 | 0.56 | no (2742 px) |
| 17    | 374–409 | 0.63 | no (3855 px) |
| 18    | 407–421 | 0.63 | no (2849 px) |
| 22    | 547–571 | 0.67 | yes |
| 24    | 622–681 | 0.67 | no (2585 px) |
| 25    | 637–654 | 0.70 | no (2772 px) |
| 26    | 658–717 | 0.68 | yes |
| 27    | 769–828 | 0.82 | no (3982 px) |
| 28    | 803–839 | 0.50 | no (1458 px) |
| 32    | 866–925 | 0.47 | no (5161 px) |
| 33    | 990–1021| 0.50 | no (2088 px) |
| 34    | 1043–1075 | 0.55 | no (1669 px) |
| 36    | 1146–1161 | 0.66 | no (1806 px) |

- **Mean IoU = 0.66.** The two methods agree on roughly where the object is, but their shapes are clearly different.
- The comparison images show why. The Task 1 blob has holes inside the body (where the shirt colour is close to the floor), broken legs and arms, and blobs grown by the morphology. SAM 2 gives one solid object with sharp edges that follow the real outline (see `Results/Task2/Comparison/track_003.png`).
- 12 of the 19 Task 1 tracks start as a thin box (about 26 px high) along the bottom edge of the frame. That is the near player's head or arm entering the view. SAM 2 segments these correctly, but they stay tiny, so they are skipped by `min_object_area`.
- **Track 12** is a good example of the memory. Its Task 1 box is only 62 × 40 px when the player starts entering the view, yet SAM 2 keeps following him through the clip until his full body is visible (the mask grows from a small patch to 31,400 px). Only the clear frames are pasted.
- **Synopsis:** 6 objects on one reference frame, 135 frames (4.5 s) in `tabletennis_synopsis_sam2.mp4`.

---

## 7. Limitations and notes for grading

- SAM 2 only refines objects that Task 1 found. If the Task 1 tracker splits one player into several tracks, or loses and restarts a track, the same player can appear in several cells, each with its own SAM 2 cut-out. Merging broken tracks is a different topic (Re-Identification), so it is not done here.
- The box prompt comes from background subtraction. If the box covers two objects, SAM 2 segments what is inside it. Track 1 is like this: Task 1 merged both players into one box at the first frame, so its cut-out shows both players together.
- A few masks have small holes or specks (for example a dark patch on the red shirt in track 1). The optional SAM 2 CUDA extension that fills small holes was not compiled. Without it SAM 2 prints a warning ("cannot import name '_C'") that can be ignored, as the SAM 2 install guide says.
- Skipping objects with `min_object_area` is a choice: the synopsis shows 6 clear objects instead of 19 objects where most are slivers. With `min_object_area = 0`, every Task 1 track is pasted.
- Each object is followed for at most `clip_len` frames to keep memory and run time small. SAM 2 itself would also work on the full video, because its memory bank is bounded.
- The SAM 2 masks are not compared with ground truth (there are no annotations for this video). The IoU in step 4 measures how much the two methods agree, not how accurate either one is. The comparison images show the difference visually.

## 8. References

1. N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, C. Feichtenhofer. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024. Code: <https://github.com/facebookresearch/sam2>
2. S. W. Oh, J.-Y. Lee, N. Xu, S. J. Kim. *Video Object Segmentation using Space-Time Memory Networks (STM).* ICCV 2019.
3. H. K. Cheng, A. G. Schwing. *XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model.* ECCV 2022.
