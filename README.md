# Compliant mechanism synthesis

Objective: Synthesize compliant mechanisms that achieve:
- Target rigidities between input and output ports.
- Desired motion paths
- Specified force transmission characteristics

## First approach plan

Initial setup (the simplest):

- 2D (0,0) to (GRID_SIZE,GRID_SIZE) grid of pixels
- Bottom plate is fixed, top plate is free to move
- Target is: top plate has to have a specified k_x, k_y and k_theta

The media:
- 2D grid of pixels, each pixel is either solid or void
- Solid pixels are modeled with FEA. (Steel's mechanical properties, for now)

The model:
- 4x4 patches projected to an embedding space of d_model.
- Diffusion transformer encoder predicts the deltas to create a new design that
more better matches the target k_x, k_y and k_theta.

Training:
- Typical diffusion style training, with a noise scheduler and a loss that
compares the predicted design's k_x, k_y and k_theta to the target.
- Additional loss for minimizing solid's surface area, to encourage simpler
designs. Simply summing up all the interface edges between solid and void
pixels should work.
- Also, conectivity, if the resulting design is not connected, we add a penalty
to the loss.


