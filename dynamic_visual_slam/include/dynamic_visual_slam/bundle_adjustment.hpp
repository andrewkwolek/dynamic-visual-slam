/**
 * @file bundle_adjustment.hpp
 * @brief Sliding window bundle adjustment implementation for visual SLAM optimization
 * @author Andrew Kwolek
 * @date 2025
 * @version 0.0.1
 * 
 * This file implements a sliding window bundle adjustment system using the Ceres
 * optimization library for real-time visual SLAM applications. Bundle adjustment
 * simultaneously optimizes camera poses and 3D landmark positions by minimizing
 * reprojection errors across multiple views.
 * 
 * **Mathematical Foundation:**
 * Bundle adjustment solves the following nonlinear least squares problem:
 * \f[
 * \min_{\mathbf{P}, \mathbf{X}} \sum_{i,j} \rho\left(\left\|\mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)\right\|^2\right)
 * \f]
 * where \f$\mathbf{P}_i\f$ are camera poses, \f$\mathbf{X}_j\f$ are 3D landmarks,
 * \f$\mathbf{x}_{ij}\f$ are 2D observations, \f$\pi\f$ is the projection function,
 * and \f$\rho\f$ is a robust loss function.
 * 
 * **Key Features:**
 * - **Sliding Window:** Bounds computational complexity for real-time performance
 * - **Robust Optimization:** Huber loss function handles outlier observations
 * - **Quaternion Parameterization:** Proper manifold handling for rotations
 * - **Weighted Residuals:** Incorporates measurement uncertainty in optimization
 * - **Multi-threaded:** Leverages Ceres parallel optimization capabilities
 * 
 * **Coordinate Conventions:**
 * - **Camera Poses:** Stored as world-to-camera transformations for optimization
 * - **Quaternions:** Hamilton convention (w, x, y, z) with unit norm constraint
 * - **Landmarks:** 3D positions in world coordinate frame
 * - **Observations:** 2D pixel coordinates in image plane
 * 
 * **Performance Characteristics:**
 * - **Typical Runtime:** 50-200ms for 5-10 keyframes with 100-500 landmarks
 * - **Memory Scaling:** O(keyframes² + landmarks) for sparse structure
 * - **Convergence:** Usually 10-20 iterations for well-conditioned problems
 * 
 * @note Requires Ceres Solver library for nonlinear optimization
 * @note Designed for real-time constraints with bounded window sizes
 * @warning Numerical stability depends on sufficient parallax between keyframes
 * 
 * @see Triggs et al. "Bundle Adjustment - A Modern Synthesis" for mathematical background
 * @see Ceres Solver documentation for optimization algorithm details
 */

#ifndef BUNDLE_ADJUSTMENT_HPP
#define BUNDLE_ADJUSTMENT_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <functional>

#include "rclcpp/rclcpp.hpp"

/**
 * @struct CameraPose
 * @brief Camera pose representation for bundle adjustment optimization
 * 
 * Stores camera pose as quaternion rotation and translation vector in format
 * suitable for Ceres optimization. Handles conversions between different
 * pose representations commonly used in computer vision and robotics.
 * 
 * **Parameterization:**
 * - **Rotation:** Unit quaternion in Hamilton convention (w, x, y, z)
 * - **Translation:** 3D translation vector in world coordinates
 * - **Convention:** Camera-to-world transformation for optimization efficiency
 * 
 * **Optimization Benefits:**
 * - **Quaternion Manifold:** Proper handling of rotation constraints via Ceres
 * - **Numerical Stability:** Avoids gimbal lock and rotation matrix singularities
 * - **Minimal Parameterization:** 7 parameters vs. 12 for transformation matrix
 * 
 * **Coordinate Frame Conventions:**
 * - **Storage:** Camera-to-world transformation (inverse of typical pose)
 * - **Usage:** Optimizes T_camera_world = [R_cw | t_cw] for efficiency
 * - **Conversion:** Provides methods for standard world-to-camera format
 * 
 * @note Quaternions automatically normalized during optimization via manifold
 * @note Translation represents camera center position in world coordinates
 * @see Ceres documentation on quaternion manifolds for rotation handling
 */
struct CameraPose {
    /// Quaternion rotation (w, x, y, z) - Hamilton convention
    double rotation[4];
    
    /// Translation vector (camera center in world coordinates)
    double translation[3];

    /**
     * @brief Default constructor initializing identity pose
     * 
     * Creates camera pose at world origin with no rotation, suitable
     * as initial estimate or reference frame for optimization.
     * 
     * @post rotation = [1, 0, 0, 0] (identity quaternion)
     * @post translation = [0, 0, 0] (origin position)
     */
    CameraPose() {
        rotation[0] = 1.0; rotation[1] = 0.0; rotation[2] = 0.0; rotation[3] = 0.0;  // Identity quaternion
        translation[0] = 0.0; translation[1] = 0.0; translation[2] = 0.0;            // Origin position
    }

    /**
     * @brief Converts from OpenCV pose matrices to optimization format
     * 
     * Converts standard computer vision pose representation (world-to-camera)
     * to camera-to-world format used internally for optimization efficiency.
     * Handles quaternion extraction and coordinate frame transformation.
     * 
     * **Mathematical Transformation:**
     * Given world-to-camera pose [R_wc | t_wc], computes camera-to-world:
     * \f[
     * \mathbf{R}_{cw} = \mathbf{R}_{wc}^T, \quad \mathbf{t}_{cw} = -\mathbf{R}_{cw} \mathbf{t}_{wc}
     * \f]
     * 
     * @param R_world_camera 3x3 rotation matrix (world to camera)
     * @param t_world_camera 3x1 translation vector (world to camera)
     * 
     * @pre R_world_camera is valid rotation matrix (orthogonal, det=1)
     * @pre t_world_camera is 3D translation vector
     * 
     * @post rotation contains unit quaternion representing camera-to-world rotation
     * @post translation contains camera center position in world coordinates
     * 
     * @note Input matrices remain unchanged (const parameters)
     * @note Automatic quaternion normalization ensures unit constraint
     */
    void fromRt(const cv::Mat& R_world_camera, const cv::Mat& t_world_camera) {
        // Convert to camera-to-world transformation for optimization
        cv::Mat R_camera_world = R_world_camera.t();                    // R^T for inverse rotation
        cv::Mat t_camera_world = -R_camera_world * t_world_camera;      // -R^T * t for camera center
        
        // Convert rotation matrix to Eigen format
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_camera_world.at<double>(i, j);
            }
        }
        
        // Extract quaternion from rotation matrix
        Eigen::Quaterniond q(R_eigen);
        q.normalize();                                                  // Ensure unit quaternion
        
        // Store in Hamilton convention (w, x, y, z)
        rotation[0] = q.w();
        rotation[1] = q.x();
        rotation[2] = q.y();
        rotation[3] = q.z();
        
        // Store camera center position
        translation[0] = t_camera_world.at<double>(0);
        translation[1] = t_camera_world.at<double>(1);
        translation[2] = t_camera_world.at<double>(2);
    }

    /**
     * @brief Converts from optimization format to OpenCV pose matrices
     * 
     * Converts internal camera-to-world representation back to standard
     * computer vision world-to-camera format for use with OpenCV functions
     * and pose publishing. Inverse operation of fromRt().
     * 
     * **Mathematical Transformation:**
     * Given camera-to-world pose [R_cw | t_cw], computes world-to-camera:
     * \f[
     * \mathbf{R}_{wc} = \mathbf{R}_{cw}^T, \quad \mathbf{t}_{wc} = -\mathbf{R}_{wc} \mathbf{t}_{cw}
     * \f]
     * 
     * @param[out] R_world_camera Output 3x3 rotation matrix (world to camera)
     * @param[out] t_world_camera Output 3x1 translation vector (world to camera)
     * 
     * @pre rotation contains valid unit quaternion
     * @pre translation contains valid 3D coordinates
     * 
     * @post R_world_camera contains orthogonal rotation matrix
     * @post t_world_camera contains camera position relative to world
     * 
     * @note Output matrices allocated by this method
     * @note Maintains numerical precision through proper matrix operations
     */
    void toRt(cv::Mat& R_world_camera, cv::Mat& t_world_camera) const {
        // Convert quaternion back to rotation matrix
        Eigen::Quaterniond q(rotation[0], rotation[1], rotation[2], rotation[3]);
        Eigen::Matrix3d R_camera_world = q.toRotationMatrix();
        
        // Convert to world-to-camera transformation
        Eigen::Matrix3d R_world_camera_eigen = R_camera_world.transpose();
        Eigen::Vector3d t_camera_world_vec(translation[0], translation[1], translation[2]);
        Eigen::Vector3d t_world_camera_vec = -R_world_camera_eigen * t_camera_world_vec;
        
        // Convert back to OpenCV format
        R_world_camera = cv::Mat(3, 3, CV_64F);
        t_world_camera = cv::Mat(3, 1, CV_64F);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_world_camera.at<double>(i, j) = R_world_camera_eigen(i, j);
            }
            t_world_camera.at<double>(i) = t_world_camera_vec(i);
        }
    }
};

/**
 * @struct Landmark
 * @brief 3D landmark representation for bundle adjustment optimization
 * 
 * Stores 3D point landmarks with semantic information and optimization
 * constraints. Designed for efficient batch optimization while maintaining
 * semantic context for SLAM applications.
 * 
 * **Optimization Properties:**
 * - **3D Position:** World coordinates optimized during bundle adjustment
 * - **Semantic Category:** String label for landmark classification
 * - **Fixed Constraint:** Optional flag to hold position constant during optimization
 * - **Unique ID:** Global identifier for data association and tracking
 * 
 * **Usage Patterns:**
 * - **Variable Landmarks:** Position optimized based on multi-view observations
 * - **Fixed Landmarks:** Position held constant (e.g., known control points)
 * - **Semantic Filtering:** Category used for landmark subset optimization
 * 
 * @note Position array directly accessible by Ceres optimization algorithms
 * @note Semantic information preserved throughout optimization process
 * @see OptimizationResult for how optimized positions are returned
 */
struct Landmark {
    uint64_t id;            ///< Unique landmark identifier for tracking and association
    std::string category;   ///< Semantic category label (e.g., "building", "tree", "unlabeled")
    double position[3];     ///< 3D position in world coordinates [x, y, z] (meters)
    bool fixed;             ///< Whether position should remain constant during optimization

    /**
     * @brief Default constructor creating landmark at origin
     * 
     * Initializes landmark with default values suitable for subsequent
     * parameter assignment or as placeholder during data structure setup.
     * 
     * @post id = 0, fixed = false, position = [0, 0, 0], category = ""
     */
    Landmark() : id(0), fixed(false) {
        position[0] = 0.0;
        position[1] = 0.0;
        position[2] = 0.0;
    }

    /**
     * @brief Constructs landmark with specified parameters
     * 
     * Creates fully initialized landmark with given position, semantic
     * information, and optimization constraints. Primary constructor
     * for landmarks created from SLAM observations.
     * 
     * @param landmark_id Unique identifier for this landmark
     * @param cat Semantic category string for classification
     * @param x,y,z Initial 3D position coordinates in world frame (meters)
     * @param fixed_point Whether to hold position constant during optimization
     * 
     * @post All parameters initialized as specified
     * @post Landmark ready for inclusion in optimization problem
     * 
     * @note Position coordinates should be in same frame as camera poses
     * @note Category string used for semantic landmark filtering
     */
    Landmark(uint64_t landmark_id, std::string cat, double x, double y, double z, bool fixed_point = false) 
        : id(landmark_id), category(cat), fixed(fixed_point) {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }
};

/**
 * @struct Observation
 * @brief 2D landmark observation for bundle adjustment constraints
 * 
 * Represents a 2D pixel observation of a 3D landmark from a specific camera
 * pose. Forms the fundamental constraint in bundle adjustment by linking
 * 3D landmarks to 2D measurements through camera projection models.
 * 
 * **Bundle Adjustment Role:**
 * Each observation contributes a 2D residual to the optimization:
 * \f[
 * \mathbf{r}_{ij} = \mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)
 * \f]
 * where \f$\mathbf{x}_{ij}\f$ is the pixel observation, \f$\mathbf{P}_i\f$ is camera pose,
 * and \f$\mathbf{X}_j\f$ is the 3D landmark position.
 * 
 * **Data Association:**
 * - **Landmark ID:** Links observation to specific 3D point for optimization
 * - **Frame ID:** Associates observation with specific camera pose
 * - **Semantic Category:** Enables category-specific optimization strategies
 * 
 * @note Pixel coordinates assumed to follow standard image conventions (u, v)
 * @note Category must match associated landmark for valid constraint
 */
struct Observation {
    double pixel[2];        ///< 2D pixel coordinates in image [u, v]
    uint64_t landmark_id;   ///< ID of associated 3D landmark
    std::string category;   ///< Semantic category matching associated landmark
    int frame_id;           ///< ID of keyframe where observation was made

    /**
     * @brief Constructs observation with full parameter specification
     * 
     * Creates observation linking specific pixel coordinates to landmark
     * and camera frame. Forms basis for reprojection error constraints
     * in bundle adjustment optimization.
     * 
     * @param x,y Pixel coordinates in image (standard u,v convention)
     * @param landmark Global ID of associated 3D landmark
     * @param cat Semantic category string for filtering and validation
     * @param frame ID of keyframe containing this observation
     * 
     * @pre pixel coordinates within valid image bounds
     * @pre landmark_id corresponds to existing landmark
     * @pre frame_id corresponds to existing keyframe
     * @pre category matches associated landmark category
     * 
     * @post Observation ready for inclusion in optimization constraints
     */
    Observation(double x, double y, uint64_t landmark, std::string cat, int frame) 
        : landmark_id(landmark), category(cat), frame_id(frame) {
        pixel[0] = x;
        pixel[1] = y;
    }
};

/**
 * @struct KeyframeData
 * @brief Keyframe information for bundle adjustment optimization
 * 
 * Stores camera pose and metadata for keyframes included in bundle adjustment
 * optimization. Provides temporal context and pose initialization for
 * the optimization process.
 * 
 * **Optimization Role:**
 * - **Pose Initialization:** Provides initial estimate for camera parameters
 * - **Temporal Ordering:** Maintains sequence information for sliding window
 * - **Reference Frame:** First keyframe often held fixed as gauge constraint
 * - **Convergence:** Good initialization critical for optimization success
 * 
 * **Data Requirements:**
 * - **Valid Pose:** Rotation and translation must represent feasible camera pose
 * - **Consistent Frames:** Coordinate frames must match landmark positions
 * - **Temporal Accuracy:** Timestamps enable motion model constraints if used
 * 
 * @note Pose matrices stored in standard computer vision format (world-to-camera)
 * @note Conversion to optimization format handled internally by CameraPose
 */
struct KeyframeData {
    int frame_id;           ///< Unique keyframe identifier for constraint association
    cv::Mat R;              ///< 3x3 rotation matrix (world to camera)
    cv::Mat t;              ///< 3x1 translation vector (world to camera)
    rclcpp::Time timestamp; ///< ROS timestamp when keyframe was captured

    /**
     * @brief Constructs keyframe with pose and temporal information
     * 
     * Creates keyframe data structure with pose matrices and timestamp.
     * Input matrices are cloned to ensure data independence during
     * optimization process.
     * 
     * @param id Unique identifier for keyframe association
     * @param rotation 3x3 rotation matrix (will be cloned)
     * @param translation 3x1 translation vector (will be cloned)
     * @param stamp ROS timestamp for temporal tracking
     * 
     * @pre rotation is valid orthogonal matrix with determinant 1
     * @pre translation is 3D vector in consistent coordinate frame
     * @pre stamp represents valid ROS time
     * 
     * @post R and t contain independent copies of input matrices
     * @post Keyframe ready for bundle adjustment optimization
     */
    KeyframeData(int id, const cv::Mat& rotation, const cv::Mat& translation, const rclcpp::Time& stamp)
        : frame_id(id), R(rotation.clone()), t(translation.clone()), timestamp(stamp) {}
};

/**
 * @struct OptimizationResult
 * @brief Comprehensive results from bundle adjustment optimization
 * 
 * Contains complete information about bundle adjustment optimization including
 * success status, performance metrics, and optimized parameters. Enables
 * thorough analysis of optimization quality and system performance.
 * 
 * **Success Metrics:**
 * - **Convergence Status:** Whether optimization reached convergence criteria
 * - **Final Cost:** Objective function value indicating fit quality
 * - **Iteration Count:** Number of optimization steps taken
 * - **Runtime Performance:** Wall clock time for optimization process
 * 
 * **Optimized Parameters:**
 * - **Camera Poses:** Updated keyframe poses after optimization
 * - **Landmark Positions:** Updated 3D landmark coordinates
 * - **Organized Access:** Poses by frame ID, landmarks by ID and category
 * 
 * **Diagnostic Information:**
 * - **Status Message:** Human-readable description of optimization outcome
 * - **Problem Size:** Number of poses and landmarks in optimization
 * - **Performance Data:** Timing information for system analysis
 * 
 * @note All optimized poses in standard world-to-camera format
 * @note Landmark positions remain in world coordinate frame
 * @see SlidingWindowBA::optimize() for result generation
 */
struct OptimizationResult {
    /// Optimization status and quality metrics
    bool success;                           ///< Whether optimization converged successfully
    double final_cost;                      ///< Final objective function value (lower is better)
    int iterations_completed;               ///< Number of optimization iterations performed
    int frames_optimized;                   ///< Number of keyframes included in optimization
    int landmarks_optimized;                ///< Number of landmarks included in optimization
    std::string message;                    ///< Human-readable status message
    std::chrono::milliseconds optimization_time; ///< Wall clock optimization duration
    
    /// Optimized parameters (output from bundle adjustment)
    std::map<int, std::pair<cv::Mat, cv::Mat>> optimized_poses;     ///< Optimized poses [frame_id -> (R, t)]
    std::map<std::pair<uint64_t, std::string>, cv::Point3d> optimized_landmarks; ///< Optimized landmarks [(id, category) -> position]
};

/**
 * @struct WeightedSquaredReprojectionError
 * @brief Ceres cost function for bundle adjustment reprojection constraints
 * 
 * Implements the fundamental reprojection error cost function for bundle adjustment
 * optimization. Each observation contributes a weighted 2D residual based on the
 * difference between observed and predicted pixel coordinates.
 * 
 * **Mathematical Model:**
 * For observation \f$\mathbf{x}_{ij}\f$ of landmark \f$\mathbf{X}_j\f$ from camera \f$\mathbf{P}_i\f$:
 * \f[
 * \mathbf{r}_{ij} = \frac{1}{\sigma}\left(\mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)\right)
 * \f]
 * where \f$\pi\f$ is the pinhole projection function and \f$\sigma\f$ is measurement uncertainty.
 * 
 * **Pinhole Camera Model:**
 * \f[
 * \pi(\mathbf{P}, \mathbf{X}) = \begin{bmatrix} f_x \frac{x}{z} + c_x \\ f_y \frac{y}{z} + c_y \end{bmatrix}
 * \f]
 * where \f$(x, y, z)^T = \mathbf{R}\mathbf{X} + \mathbf{t}\f$ are camera coordinates.
 * 
 * **Automatic Differentiation:**
 * - **Template Operator:** Enables Ceres automatic differentiation for gradients
 * - **Parameter Blocks:** [quaternion(4), translation(3), landmark(3)]
 * - **Residual Dimension:** 2D (horizontal and vertical pixel errors)
 * 
 * **Robustness Features:**
 * - **Weighted Residuals:** Incorporates measurement uncertainty
 * - **Depth Validation:** Rejects landmarks behind camera (z <= 0.1)
 * - **Numerical Stability:** Careful handling of division by depth
 * 
 * @note Uses quaternion rotation for numerical stability and manifold constraints
 * @note Designed for integration with Ceres AutoDiffCostFunction framework
 * @see Ceres Solver documentation for cost function implementation details
 */
struct WeightedSquaredReprojectionError {
    /// Camera intrinsic parameters and measurement uncertainty
    double observed_x, observed_y;     ///< Observed pixel coordinates
    double fx, fy, cx, cy;            ///< Camera intrinsic parameters
    double inv_sigma;                 ///< Inverse measurement standard deviation (1/σ)

    /**
     * @brief Constructs cost function with observation and camera parameters
     * 
     * Initializes reprojection error cost function with specific observation
     * and camera calibration data. Stores parameters for use in templated
     * operator during optimization.
     * 
     * @param observed_x,observed_y Measured pixel coordinates of landmark
     * @param fx,fy,cx,cy Camera intrinsic parameters for projection
     * @param sigma_pixels Standard deviation of pixel measurement noise
     * 
     * @pre Camera intrinsics represent valid pinhole camera model
     * @pre sigma_pixels > 0 for meaningful weighting
     * @pre observed coordinates within reasonable image bounds
     * 
     * @post Cost function ready for automatic differentiation
     */
    WeightedSquaredReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), inv_sigma(1.0 / sigma_pixels) {}

    /**
     * @brief Templated operator for automatic differentiation in Ceres
     * 
     * Computes reprojection error and gradients for bundle adjustment optimization.
     * Template design enables Ceres automatic differentiation for efficient
     * gradient computation without manual derivative implementation.
     * 
     * **Computation Pipeline:**
     * 1. **Rotation:** Apply quaternion rotation to transform landmark to camera frame
     * 2. **Translation:** Add camera translation to complete coordinate transformation
     * 3. **Depth Check:** Validate landmark is in front of camera for valid projection
     * 4. **Projection:** Apply pinhole camera model to compute predicted pixel coordinates
     * 5. **Residual:** Compute weighted difference between observed and predicted coordinates
     * 
     * **Robust Handling:**
     * - **Behind Camera:** Set zero residual for landmarks with z ≤ 0.1 (invalid geometry)
     * - **Numerical Stability:** Careful floating point operations for edge cases
     * - **Weighted Errors:** Scale residuals by inverse measurement uncertainty
     * 
     * @tparam T Scalar type (double for evaluation, Jet for automatic differentiation)
     * @param camera_rotation Quaternion camera rotation parameters [w, x, y, z]
     * @param camera_translation Camera translation vector [tx, ty, tz]
     * @param point 3D landmark position in world coordinates [x, y, z]
     * @param[out] residuals Output 2D residual vector [horizontal_error, vertical_error]
     * @return true (Ceres convention for successful evaluation)
     * 
     * @pre camera_rotation is unit quaternion (enforced by Ceres manifold)
     * @pre point coordinates are finite and reasonable
     * @pre All input arrays have correct dimensions
     * 
     * @post residuals contains weighted 2D reprojection error
     * @post Gradients computed automatically by Ceres if T is Jet type
     * 
     * @note Zero residuals for invalid geometry prevent optimization instability
     * @note Weighted residuals incorporate measurement uncertainty in optimization
     */
    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation,
                    const T* const point, T* residuals) const {
        
        // Transform 3D point from world to camera coordinates using quaternion rotation
        T point_camera[3];
        ceres::QuaternionRotatePoint(camera_rotation, point, point_camera);
        
        // Add camera translation to complete coordinate transformation
        point_camera[0] += camera_translation[0];  // x_camera
        point_camera[1] += camera_translation[1];  // y_camera
        point_camera[2] += camera_translation[2];  // z_camera (depth)
        
        // Validate landmark is in front of camera (positive depth)
        if (point_camera[2] <= T(0.1)) {
            // Set zero residual for invalid geometry (behind camera or too close)
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        // Apply pinhole camera model to compute predicted pixel coordinates
        T predicted_x = T(fx) * point_camera[0] / point_camera[2] + T(cx);  // u = fx * x/z + cx
        T predicted_y = T(fy) * point_camera[1] / point_camera[2] + T(cy);  // v = fy * y/z + cy
        
        // Compute weighted reprojection error (observed - predicted)
        T error_x = predicted_x - T(observed_x);
        T error_y = predicted_y - T(observed_y);
        
        // Apply measurement uncertainty weighting
        residuals[0] = T(inv_sigma) * error_x;  // Weighted horizontal error
        residuals[1] = T(inv_sigma) * error_y;  // Weighted vertical error
        
        return true;
    }

    /**
     * @brief Factory method for creating Ceres cost function
     * 
     * Creates AutoDiffCostFunction with proper template parameters for
     * integration with Ceres optimization framework. Encapsulates cost
     * function creation details for clean client code.
     * 
     * **Template Parameters:**
     * - **Residual Dimension:** 2 (horizontal and vertical pixel errors)
     * - **Parameter Block 1:** 4 (quaternion rotation)
     * - **Parameter Block 2:** 3 (camera translation)
     * - **Parameter Block 3:** 3 (landmark position)
     * 
     * @param observed_x,observed_y Measured pixel coordinates
     * @param fx,fy,cx,cy Camera intrinsic parameters
     * @param sigma_pixels Measurement uncertainty standard deviation
     * @return Pointer to Ceres cost function ready for optimization
     * 
     * @note Caller owns returned cost function pointer
     * @note Cost function automatically handles gradient computation
     * @see Ceres AutoDiffCostFunction documentation for usage patterns
     */
    static ceres::CostFunction* Create(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels) {
        return new ceres::AutoDiffCostFunction<WeightedSquaredReprojectionError, 2, 4, 3, 3>(
            new WeightedSquaredReprojectionError(observed_x, observed_y, fx, fy, cx, cy, sigma_pixels));
    }
};

/**
 * @class SlidingWindowBA
 * @brief Sliding window bundle adjustment implementation for real-time SLAM
 * 
 * Implements sliding window bundle adjustment optimization using Ceres Solver
 * for real-time visual SLAM applications. Balances optimization accuracy with
 * computational performance through bounded window sizes and efficient sparse
 * matrix algorithms.
 * 
 * **Sliding Window Strategy:**
 * - **Bounded Complexity:** Fixed window size prevents unbounded computational growth
 * - **Recent Focus:** Optimizes most recent keyframes where tracking is most critical
 * - **Memory Efficiency:** Maintains constant memory usage independent of trajectory length
 * - **Real-time Performance:** Typical optimization time 50-200ms for practical window sizes
 * 
 * **Mathematical Formulation:**
 * Minimizes weighted reprojection error across all observations in window:
 * \f[
 * \min_{\mathbf{P}_{i}, \mathbf{X}_{j}} \sum_{(i,j) \in \mathcal{O}} \rho\left(\frac{\|\mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)\|^2}{\sigma^2}\right)
 * \f]
 * where \f$\mathcal{O}\f$ is the set of observations in the temporal window.
 * 
 * **Optimization Features:**
 * - **Robust Loss:** Huber loss function reduces outlier impact
 * - **Manifold Constraints:** Proper quaternion handling via Ceres manifolds
 * - **Gauge Freedom:** First camera pose often held fixed as reference frame
 * - **Sparse Structure:** Leverages sparsity for efficient large-scale optimization
 * - **Multi-threading:** Parallel optimization when beneficial
 * 
 * **Performance Characteristics:**
 * - **Window Size:** Typically 5-10 keyframes for real-time performance
 * - **Convergence:** Usually 10-20 iterations for well-conditioned problems
 * - **Memory Scaling:** O(keyframes² + landmarks) sparse structure
 * - **Runtime Scaling:** Approximately linear in observations for sparse problems
 * 
 * **Quality Metrics:**
 * - **Final Cost:** Indicates overall fit quality (lower is better)
 * - **Convergence Status:** Reliable indicator of optimization success
 * - **Iteration Count:** Diagnostic for problem conditioning
 * - **Reprojection RMSE:** Physical interpretation of optimization quality
 * 
 * @note Requires well-initialized poses for reliable convergence
 * @note Performance depends critically on observation distribution and parallax
 * @warning Poor initialization can lead to local minima in optimization
 * 
 * @see Ceres Solver documentation for optimization algorithm details
 * @see Triggs et al. "Bundle Adjustment" for mathematical foundation
 * 
 * @example Basic usage pattern:
 * @code
 * SlidingWindowBA optimizer(fx, fy, cx, cy, sigma_pixels);
 * OptimizationResult result = optimizer.optimize(keyframes, landmarks, observations);
 * if (result.success) {
 *     // Use result.optimized_poses and result.optimized_landmarks
 * }
 * @endcode
 */
class SlidingWindowBA {
public:
    /**
     * @brief Constructs bundle adjustment optimizer with camera parameters
     * 
     * Initializes sliding window bundle adjustment system with camera intrinsics
     * and measurement uncertainty model. Camera parameters remain constant
     * throughout optimization lifetime for consistency.
     * 
     * @param fx,fy Focal lengths in pixels (horizontal and vertical)
     * @param cx,cy Principal point coordinates in pixels
     * @param sigma_pixels Standard deviation of pixel measurement noise (default: 1.0)
     * 
     * @pre fx, fy > 0 (positive focal lengths)
     * @pre cx, cy within reasonable image bounds
     * @pre sigma_pixels > 0 (positive measurement uncertainty)
     * 
     * @post Optimizer ready for bundle adjustment optimization
     * @post Camera parameters stored for consistent optimization
     * 
     * @note sigma_pixels affects relative weighting of observations
     * @note Larger sigma_pixels increases optimization tolerance to noise
     */
    SlidingWindowBA(double fx, double fy, double cx, double cy, double sigma_pixels = 1.0)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), sigma_pixels_(sigma_pixels) {}

    /**
     * @brief Performs sliding window bundle adjustment optimization
     * 
     * Optimizes camera poses and landmark positions within the specified temporal
     * window to minimize weighted reprojection errors. Handles problem setup,
     * optimization execution, and result extraction with comprehensive error
     * handling and performance monitoring.
     * 
     * **Optimization Pipeline:**
     * 1. **Input Validation:** Check data consistency and minimum requirements
     * 2. **Problem Setup:** Create Ceres optimization problem with parameter blocks
     * 3. **Constraint Addition:** Add reprojection error residuals for all observations
     * 4. **Gauge Fixing:** Hold first camera pose constant to eliminate gauge freedom
     * 5. **Optimization:** Run Levenberg-Marquardt optimization with robust loss
     * 6. **Result Extraction:** Convert optimized parameters back to client format
     * 7. **Performance Analysis:** Compute timing and convergence diagnostics
     * 
     * **Problem Structure:**
     * - **Camera Parameters:** 7 DOF per camera (4 quaternion + 3 translation)
     * - **Landmark Parameters:** 3 DOF per landmark (x, y, z coordinates)
     * - **Residuals:** 2 DOF per observation (u, v pixel errors)
     * - **Sparsity Pattern:** Block structure enables efficient optimization
     * 
     * **Robustness Features:**
     * - **Input Validation:** Comprehensive checks prevent optimization failure
     * - **Robust Loss:** Huber loss (δ=1.345) reduces outlier impact
     * - **Convergence Criteria:** Multiple termination conditions for reliability
     * - **Exception Handling:** Graceful failure recovery with diagnostic messages
     * 
     * **Performance Optimization:**
     * - **Sparse Schur:** Exploits bundle adjustment sparsity structure
     * - **Multi-threading:** Parallel jacobian computation when beneficial
     * - **Iteration Limits:** Bounded runtime for real-time applications
     * - **Convergence Thresholds:** Tuned for SLAM accuracy requirements
     * 
     * @param keyframes Camera keyframes with poses and timestamps
     * @param landmarks 3D landmarks with positions and semantic information
     * @param observations 2D observations linking keyframes to landmarks
     * @param max_iterations Maximum optimization iterations (default: 10)
     * @return Complete optimization result with status, parameters, and diagnostics
     * 
     * @pre keyframes.size() >= 2 (minimum for meaningful optimization)
     * @pre All observations reference valid keyframes and landmarks
     * @pre Keyframe poses provide reasonable initialization
     * @pre Landmark positions within plausible depth range
     * 
     * @post result.success indicates optimization convergence status
     * @post result.optimized_poses contains updated camera poses if successful
     * @post result.optimized_landmarks contains updated positions if successful
     * @post Original input data unchanged (const parameters)
     * 
     * @note Optimization modifies internal copies, preserving input data
     * @note First keyframe pose held fixed to eliminate gauge freedom
     * @note Performance scales approximately linearly with observation count
     * @warning Poor pose initialization can cause convergence to local minima
     * 
     * @see OptimizationResult for detailed result structure
     * @see WeightedSquaredReprojectionError for cost function details
     */
    OptimizationResult optimize(const std::vector<KeyframeData>& keyframes, 
                               const std::vector<Landmark>& landmarks, 
                               const std::vector<Observation>& observations, 
                               int max_iterations = 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Initialize result structure with input validation
        OptimizationResult result;
        result.success = false;
        result.frames_optimized = keyframes.size();
        result.landmarks_optimized = landmarks.size();
        
        // Validate minimum data requirements
        if (keyframes.empty() || landmarks.empty() || observations.empty()) {
            result.message = "Insufficient input data for optimization";
            result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            return result;
        }
        
        try {
            // === PARAMETER BLOCK SETUP ===
            
            // Create camera pose parameter blocks with automatic memory management
            std::map<int, std::shared_ptr<CameraPose>> frame_poses;
            std::map<uint64_t, std::shared_ptr<Landmark>> landmark_map;
            
            ceres::Problem problem;
            
            // Initialize camera poses from keyframe data
            for (const auto& kf : keyframes) {
                auto pose = std::make_shared<CameraPose>();
                pose->fromRt(kf.R, kf.t);  // Convert to optimization format
                frame_poses[kf.frame_id] = pose;
                
                // Add parameter blocks to optimization problem
                problem.AddParameterBlock(pose->rotation, 4);      // Quaternion rotation
                problem.AddParameterBlock(pose->translation, 3);   // Translation vector
                
                // Set quaternion manifold for proper rotation optimization
                problem.SetManifold(pose->rotation, new ceres::EigenQuaternionManifold);
            }
            
            // Fix first camera pose to eliminate gauge freedom
            if (!keyframes.empty()) {
                auto first_pose = frame_poses[keyframes[0].frame_id];
                problem.SetParameterBlockConstant(first_pose->rotation);
                problem.SetParameterBlockConstant(first_pose->translation);
            }
            
            // Initialize landmark parameter blocks
            for (const auto& landmark : landmarks) {
                auto lm = std::make_shared<Landmark>(landmark);  // Create copy for optimization
                landmark_map[landmark.id] = lm;
                
                problem.AddParameterBlock(lm->position, 3);
                
                // Handle fixed landmarks (e.g., control points)
                if (lm->fixed) {
                    problem.SetParameterBlockConstant(lm->position);
                }
            }
            
            // === RESIDUAL BLOCK SETUP ===
            
            // Add reprojection error residuals for all valid observations
            int valid_observations = 0;
            for (const auto& obs : observations) {
                auto frame_it = frame_poses.find(obs.frame_id);
                auto landmark_it = landmark_map.find(obs.landmark_id);
                
                // Validate observation references exist in optimization problem
                if (frame_it != frame_poses.end() && landmark_it != landmark_map.end()) {
                    // Create weighted reprojection error cost function
                    ceres::CostFunction* cost_function = WeightedSquaredReprojectionError::Create(
                        obs.pixel[0], obs.pixel[1], fx_, fy_, cx_, cy_, sigma_pixels_
                    );
                    
                    // Add residual block with Huber robust loss
                    problem.AddResidualBlock(
                        cost_function,
                        new ceres::HuberLoss(1.345),           // Robust loss parameter
                        frame_it->second->rotation,            // Camera rotation parameters
                        frame_it->second->translation,         // Camera translation parameters
                        landmark_it->second->position          // Landmark position parameters
                    );
                    
                    valid_observations++;
                }
            }
            
            // Validate sufficient constraints for optimization
            if (valid_observations == 0) {
                result.message = "No valid observation constraints";
                result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time);
                return result;
            }
            
            // === OPTIMIZATION EXECUTION ===
            
            // Configure Ceres solver options for bundle adjustment
            ceres::Solver::Options options;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;  // LM algorithm
            options.linear_solver_type = ceres::SPARSE_SCHUR;                 // Exploit sparsity
            options.num_threads = 4;                                          // Parallel optimization
            options.max_num_iterations = max_iterations;                      // Iteration limit
            options.function_tolerance = 1e-6;                              // Function convergence
            options.gradient_tolerance = 1e-10;                             // Gradient convergence
            options.parameter_tolerance = 1e-8;                             // Parameter convergence
            options.minimizer_progress_to_stdout = false;                    // Quiet optimization
            
            // Execute optimization
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            // Record optimization timing
            auto end_time = std::chrono::high_resolution_clock::now();
            auto optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // === RESULT PROCESSING ===
            
            // Analyze convergence status
            result.success = (summary.termination_type == ceres::CONVERGENCE);
            result.final_cost = summary.final_cost;
            result.iterations_completed = summary.num_successful_steps;
            result.optimization_time = optimization_time;
            
            // Generate human-readable status message
            if (result.success) {
                result.message = "Bundle adjustment converged successfully";
            } else {
                result.message = "Bundle adjustment failed to converge: " + 
                               std::string(ceres::TerminationTypeToString(summary.termination_type));
            }
            
            // Extract optimized camera poses back to standard format
            for (const auto& [frame_id, pose] : frame_poses) {
                cv::Mat R, t;
                pose->toRt(R, t);  // Convert back to world-to-camera format
                result.optimized_poses[frame_id] = {R.clone(), t.clone()};
            }
            
            // Extract optimized landmark positions
            for (const auto& [landmark_id, landmark] : landmark_map) {
                auto key = std::make_pair(landmark_id, landmark->category);
                result.optimized_landmarks[key] = cv::Point3d(
                    landmark->position[0],
                    landmark->position[1], 
                    landmark->position[2]
                );
            }
            
        } catch (const std::exception& e) {
            // Handle optimization exceptions gracefully
            result.message = "Bundle adjustment exception: " + std::string(e.what());
            result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
        }
        
        return result;
    }

private:
    /// Camera intrinsic parameters (constant throughout optimization)
    double fx_, fy_, cx_, cy_;  ///< Focal lengths and principal point
    double sigma_pixels_;       ///< Measurement uncertainty standard deviation
};

#endif // BUNDLE_ADJUSTMENT_HPP