# From a vessel mask to a vessel graph (Milestone 3)

**Goal.** Turn a vessel mask (white = vessel) into the graph the solver needs: nodes (junctions and vessel ends) and segments, each with a length and a radius.

## Steps (`vessel_graph/extract.py`)
1. **Clean.** Keep only the field of view; drop specks smaller than 30 px; fill holes up to 10 px.
2. **Skeleton.** Thin every vessel to a 1-pixel centreline. Build a pixel graph with 8-neighbour links, dropping a diagonal link whenever a two-step straight path exists, so a centreline pixel always has exactly two neighbours.
3. **Nodes.** Pixels with one neighbour are vessel tips; pixels with three or more are junctions (touching junction pixels form one node).
4. **Segments.** Walk along the centreline from each node until the next node.
5. **Clean-up.** Prune "spurs" (tips shorter than 2 × junction radius + 3 px, an artefact of skeletonization), then merge nodes left with only two segments.
6. **Crossings.** Where an artery crosses a vein the skeleton invents a junction. A 4-arm junction, or two 3-arm junctions joined by a short bridge, is split into two separate vessels when its arms pair into two nearly straight paths (≥ 155°, arm direction measured beyond the junction). Otherwise it is kept and flagged as a possible crossing.
7. **Width (sub-pixel).** Every vessel pixel is assigned to its nearest centreline pixel. Segment width = assigned area ÷ centreline length, using only the segment's core (away from junctions, tips and crossings). With a probability map the "area" is summed probability. Segments with no core fall back to the distance transform and are marked unreliable.
8. **Flags.** Tips within a few pixels of the photo or field-of-view edge are marked *truncated*: the photo cut the vessel; it did not end there.

Lengths are measured on a lightly smoothed centreline (5-point moving average), which removes the ~8 % over-length of pixel staircases.

## Why the area method instead of the distance transform
The distance transform gives the distance from a centreline pixel to the nearest background pixel, so it moves in whole-pixel steps. For a vessel 2–4 px wide, one step is 25–50 % of the radius. The area method averages over the whole segment, so the steps cancel out.

| radius (px) | distance transform | area, binary mask | area, soft mask |
|---|---|---|---|
| 1 | 50 % | 3.0 % | 0.3 % |
| 2 | 19 % | 1.9 % | 0.2 % |
| 4 | 11 % | 0.6 % | 0.3 % |
| 8 | 5.5 % | 0.3 % | 0.3 % |

(median error, synthetic straight tubes at 18 angles × 3 sub-pixel offsets; `scripts/m3_width_accuracy.py`)

## Validation
Synthetic masks of known geometry (`synthetic_masks.py`): straight, curved, Y-junction, X-crossing, 4-arm stars, rings, and fundus-like two-tree images with crossings and specks. They are compared without relying on topology (`validation.py`):
- *recall* = share of the true centreline found within 2 px;
- *precision* = share of the found centreline that is real;
- radius error on segments matched by majority vote.

Results are in `docs/PLAN.md` §6.

## Known limitations
- **Artery/vein crossings at branch points** are flagged, not resolved. Until arteries and veins are segmented separately (M5), real graphs can join the two trees, and the physics must not be run on them.
- **A thin vessel crossing a thick one** looks the same in a binary mask as two opposite side branches.
- **Fragments under 30 px are dropped.** Isolated vessel pieces that small cannot be modelled anyway.
- **Pixel size is an assumption** unless calibrated (optic disc, M4/M7). `to_network(pixel_size_um)` makes it explicit.
