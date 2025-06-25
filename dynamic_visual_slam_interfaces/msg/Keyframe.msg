# Keyframe data for bundle adjustment
std_msgs/Header header
uint64 frame_id
geometry_msgs/Transform pose  # Camera pose in world frame (T_world_camera)

# 3D landmarks observed in this frame
Landmark[] landmarks

# 2D observations of landmarks in this frame  
Observation[] observations

# Descriptors
uint8[] descriptors  # Flattened descriptor matrix
int32 descriptor_rows
int32 descriptor_cols