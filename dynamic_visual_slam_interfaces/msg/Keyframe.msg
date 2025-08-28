## Keyframe data for bundle adjustment optimization
## Publisher: frontend node, Subscriber: backend node
## Coordinates: pose in world frame (optical), landmarks in world frame (meters)

std_msgs/Header header

## Unique keyframe identifier (monotonically increasing)
uint64 frame_id

## Camera pose in world frame (T_world_camera)
geometry_msgs/Transform pose  

## 3D landmarks observed in this frame
Landmark[] landmarks

## 2D pixel observations corresponding to landmarks (same array indexing)
Observation[] observations