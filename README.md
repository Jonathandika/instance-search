# Instance Search

In this project, we designed and implemented an Instance Search system to search images in the gallery that contain a specific object based on a query image and a bounding box to indicate the object inside the query image. The gallery consists of 5000 images, and there are 20 query instances.

## Approach

In this project, we discuss two implementations of Instance Search using CNN as the feature extractor. During the experimentation process, we implemented many models and approaches, but in this README, we will focus on the two best approaches.

### Model

The 3 models being used in this project are:

1. CLIP (Contrastive Language-Image Pretraining)
2. VGG19 (Visual Geometry Group 19)
3. Combination of CLIP and VGG19

Both models have been chosen for their effectiveness in extracting features from images.
