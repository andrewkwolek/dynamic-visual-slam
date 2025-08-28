## 2D pixel observation of 3D landmark
## Coordinates: image frame (u,v), origin at top-left

## ID of associated 3D landmark
uint64 landmark_id

## 2D pixel coordinates in image frame
float64 pixel_x
float64 pixel_y

## ORB descriptor for feature matching (32 bytes)
uint8[] descriptor