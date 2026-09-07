# Geometry_aware
Geometry-Aware Enhanced Coarse-to-Fine Registration Method for Indoor Point Clouds

Train

The proposed method can be trained on the 3DMatch and 3DLoMatch indoor point cloud registration datasets by running:

sh scripts/train_3dmatch.sh

The training process jointly optimizes the superpoint matching loss, overlap prediction loss, and point matching loss.

Test

Point correspondences are first extracted by running:

sh scripts/test_3dmatch.sh

The extracted correspondences and predicted relative poses are stored in the corresponding experiment directory.
To evaluate the model on 3DLoMatch, please change the benchmark keyword in the corresponding test configuration file from 3DMatch to 3DLoMatch, and then run the test script again.
The extracted point correspondences are evaluated using the RANSAC algorithm to estimate the relative transformation between the source and target point clouds. The final registration results are obtained from the corresponding evaluation directory.
