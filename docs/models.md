## Model Types

#### 🔹 Single Instance

- The model predicts the pose of a single animal per frame.
- Useful for datasets where there is exactly one animal present in each frame.
- Simple and fast—no need for instance detection or tracking.

#### 🔹 Top-Down

- **Stage 1: Centroid detection** – The model first predicts the centroid (center point) of each animal in the frame, providing candidate locations for each instance.
- **Stage 2: Centered instance pose estimation** – For each detected centroid, a second model predicts the pose of the animal within a cropped region centered on that centroid.
- This approach enables accurate pose estimation in crowded scenes by focusing on one animal at a time.
- Particularly effective for datasets with moderate to high animal density, where animals are not heavily overlapping.

#### 🔹 Bottom-Up

- Predicts all body part locations (keypoints) and their pairwise associations (Part Affinity Fields, PAFs) for all animals in the frame simultaneously.
- Assembles detected keypoints into individual animal instances by solving a global assignment problem based on the predicted PAFs.
- Effective for challenging scenarios with frequent occlusions, close physical contact, or overlapping animals.

#### 🔹 Top-Down Multi-Class

- **Stage 1: Centroid detection** – The model predicts the centroid of each animal instance (same as standard top-down, without classification).
- **Stage 2: Centered instance pose estimation with classification** – For each detected centroid, a second model predicts the pose and classifies the instance into predefined classes (e.g., different species, individuals, or behavioral states).
- Enables accurate pose estimation and classification in scenarios with multiple animal classes.
- **Training Requirement**: Multi-class models require ground truth track IDs during training and are used to assign persistent IDs to animals across frames.

#### 🔹 Bottom-Up Multi-Class

- Predicts all body part locations (keypoints) and their class labels for all animals simultaneously.
- Directly classifies keypoints and groups them into instances with class assignments.
- Assembles detected keypoints into individual animal instances by solving a global assignment problem, while maintaining class-specific groupings.
- **Training Requirement**: Multi-class models require ground truth track IDs during training and are used to assign persistent IDs to animals across frames.