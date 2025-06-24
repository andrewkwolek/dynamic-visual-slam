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
#include <unordered_set>
#include <memory>
#include <deque>
#include <iostream>

#include "rclcpp/rclcpp.hpp"

// Structure to store camera parameters for optimization
struct CameraPose {
    double rotation[4];    // Quaternion (w, x, y, z)
    double translation[3]; // Translation vector (x, y, z)

    CameraPose() {
        // Initialize with identity rotation and zero translation
        rotation[0] = 1.0; // w
        rotation[1] = 0.0; // x
        rotation[2] = 0.0; // y
        rotation[3] = 0.0; // z
        translation[0] = 0.0;
        translation[1] = 0.0;
        translation[2] = 0.0;
    }

    // Convert from OpenCV/Eigen to internal format
    void fromRt(const cv::Mat& R, const cv::Mat& t) {
        // Convert rotation matrix to quaternion
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        
        rotation[0] = q.w();
        rotation[1] = q.x();
        rotation[2] = q.y();
        rotation[3] = q.z();
        
        // Store translation
        translation[0] = t.at<double>(0);
        translation[1] = t.at<double>(1);
        translation[2] = t.at<double>(2);
    }

    // Convert to OpenCV format
    void toRt(cv::Mat& R, cv::Mat& t) const {
        // Convert quaternion to rotation matrix
        Eigen::Quaterniond q(rotation[0], rotation[1], rotation[2], rotation[3]);
        Eigen::Matrix3d R_eigen = q.toRotationMatrix();
        
        // Convert to cv::Mat
        R = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R.at<double>(i, j) = R_eigen(i, j);
            }
        }
        
        // Convert translation
        t = cv::Mat(3, 1, CV_64F);
        t.at<double>(0) = translation[0];
        t.at<double>(1) = translation[1];
        t.at<double>(2) = translation[2];
    }
};

// Structure to store a 3D landmark point
struct Landmark {
    double position[3];  // 3D position (x, y, z)
    bool fixed;          // Whether this point should be optimized or not

    Landmark() : fixed(false) {
        position[0] = 0.0;
        position[1] = 0.0;
        position[2] = 0.0;
    }

    Landmark(double x, double y, double z, bool fixed_point = false) : fixed(fixed_point) {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }
};

// Structure to store a 2D observation of a 3D landmark
struct Observation {
    double pixel[2];     // 2D pixel coordinates (x, y)
    int landmark_id;     // ID of the observed landmark
    int frame_id;        // ID of the frame where the observation was made

    Observation(double x, double y, int landmark, int frame) 
        : landmark_id(landmark), frame_id(frame) {
        pixel[0] = x;
        pixel[1] = y;
    }
};

// Weighted-squared reprojection error with isotropic Gaussian noise and fixed camera intrinsics
struct WeightedSquaredReprojectionError {
    WeightedSquaredReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), inv_sigma(1.0 / sigma_pixels) {}

    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation,
                    const T* const point, T* residuals) const {
        // Apply camera rotation to the point
        T point_camera[3];
        ceres::QuaternionRotatePoint(camera_rotation, point, point_camera);
        
        // Apply camera translation
        point_camera[0] += camera_translation[0];
        point_camera[1] += camera_translation[1];
        point_camera[2] += camera_translation[2];
        
        // Check for points behind the camera
        if (point_camera[2] <= T(0)) {
            residuals[0] = T(1000.0) * T(inv_sigma);
            residuals[1] = T(1000.0) * T(inv_sigma);
            return true;
        }
        
        // Project to image plane using FIXED camera parameters
        T predicted_x = T(fx) * point_camera[0] / point_camera[2] + T(cx);
        T predicted_y = T(fy) * point_camera[1] / point_camera[2] + T(cy);
        
        // Compute reprojection error: e_reproj = z_j - π(x_j^C, ξ)
        T error_x = predicted_x - T(observed_x);
        T error_y = predicted_y - T(observed_y);
        
        // Apply isotropic weighting: 1/σ
        // When Ceres squares these residuals: (1/σ²) * (error_x² + error_y²)
        // This implements ||z_j - π(x_j^C, ξ)||²_{σI} from the book
        residuals[0] = T(inv_sigma) * error_x;
        residuals[1] = T(inv_sigma) * error_y;
        
        return true;
    }

    // Factory to create the cost function
    static ceres::CostFunction* Create(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels) {
        return new ceres::AutoDiffCostFunction<WeightedSquaredReprojectionError, 2, 4, 3, 3>(
            new WeightedSquaredReprojectionError(observed_x, observed_y, fx, fy, cx, cy, sigma_pixels));
    }

    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;  // Fixed camera intrinsics (not optimized)
    double inv_sigma;       // 1/σ for efficiency (precomputed)
};

class SlidingWindowBA {
public:
    SlidingWindowBA(int window_size, double fx, double fy, double cx, double cy, double sigma_pixels = 1.0)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), sigma_pixels_(sigma_pixels),
          window_size_(window_size), next_landmark_id_(0), next_frame_id_(0) {}

    // Set the noise parameter
    void setNoiseModel(double sigma_pixels) {
        sigma_pixels_ = sigma_pixels;
    }

    // Add a new frame to the sliding window
    int addFrame(const cv::Mat& R, const cv::Mat& t) {
        int frame_id = next_frame_id_++;
        
        // Create camera pose
        auto pose = std::make_shared<CameraPose>();
        pose->fromRt(R, t);
        frame_poses_[frame_id] = pose;
        
        // Add to frame queue
        frame_queue_.push_back(frame_id);
        
        // Remove old frames if we exceed the window size
        if (frame_queue_.size() > static_cast<size_t>(window_size_)) {
            int old_frame_id = frame_queue_.front();
            frame_queue_.pop_front();
            
            // Remove the old frame's pose
            frame_poses_.erase(old_frame_id);
            
            // Remove observations for this frame
            auto it = observations_.begin();
            while (it != observations_.end()) {
                if (it->frame_id == old_frame_id) {
                    it = observations_.erase(it);
                } else {
                    ++it;
                }
            }
            
            pruneLandmarks();
        }
        
        return frame_id;
    }

    // Add or update a landmark with a new observation
    int addObservation(int frame_id, double x, double y, double X, double Y, double Z) {
        // Check if frame exists
        if (frame_poses_.find(frame_id) == frame_poses_.end()) {
            return -1;
        }
        
        // Create a new landmark
        int landmark_id = next_landmark_id_++;
        landmarks_[landmark_id] = std::make_shared<Landmark>(X, Y, Z);
        
        // Add the observation
        observations_.emplace_back(x, y, landmark_id, frame_id);
        
        return landmark_id;
    }

    // Add an observation of an existing landmark
    void addObservation(int frame_id, int landmark_id, double x, double y) {
        // Check if frame and landmark exist
        if (frame_poses_.find(frame_id) == frame_poses_.end() || 
            landmarks_.find(landmark_id) == landmarks_.end()) {
            return;
        }
        
        // Add the observation
        observations_.emplace_back(x, y, landmark_id, frame_id);
    }

    // Run robust bundle adjustment with weighted-squared reprojection error + Huber loss
    void optimize(int num_iterations) {
        // Create the Ceres problem
        ceres::Problem problem;
        
        // Set up camera parameter blocks (only poses, intrinsics are fixed)
        for (const auto& frame_pair : frame_poses_) {
            int frame_id = frame_pair.first;
            const auto& pose = frame_pair.second;
            
            problem.AddParameterBlock(pose->rotation, 4);
            problem.AddParameterBlock(pose->translation, 3);
            
            // Fix the first camera pose for gauge freedom
            if (frame_id == frame_queue_.front()) {
                problem.SetParameterBlockConstant(pose->rotation);
                problem.SetParameterBlockConstant(pose->translation);
            }
            
            // Add quaternion normalization constraint
            problem.SetManifold(pose->rotation, new ceres::EigenQuaternionManifold);
        }
        
        // Add landmark parameter blocks
        for (const auto& landmark_pair : landmarks_) {
            const auto& landmark = landmark_pair.second;
            
            problem.AddParameterBlock(landmark->position, 3);
            
            // Fix landmarks if specified
            if (landmark->fixed) {
                problem.SetParameterBlockConstant(landmark->position);
            }
        }
        
        // Add residual blocks with weighted-squared reprojection error + Huber loss
        for (const auto& obs : observations_) {
            auto frame_it = frame_poses_.find(obs.frame_id);
            auto landmark_it = landmarks_.find(obs.landmark_id);
            
            if (frame_it != frame_poses_.end() && landmark_it != landmarks_.end()) {
                // Create weighted-squared reprojection error with FIXED intrinsics
                ceres::CostFunction* cost_function = WeightedSquaredReprojectionError::Create(
                    obs.pixel[0], obs.pixel[1], fx_, fy_, cx_, cy_, sigma_pixels_
                );
                
                problem.AddResidualBlock(
                    cost_function,
                    new ceres::HuberLoss(1.345),      // Standard Huber threshold for outlier rejection
                    frame_it->second->rotation,       // Optimize: camera rotation
                    frame_it->second->translation,    // Optimize: camera translation  
                    landmark_it->second->position     // Optimize: 3D landmark position
                );
                // Note: Camera intrinsics (fx_, fy_, cx_, cy_) are fixed constants
            }
        }
        
        // Solver options optimized for bundle adjustment
        ceres::Solver::Options options;
        
        // Use Levenberg-Marquardt for robust convergence
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.initial_trust_region_radius = 1e4;
        options.max_trust_region_radius = 1e8;
        options.min_trust_region_radius = 1e-32;
        
        // SPARSE_SCHUR is optimal for bundle adjustment structure
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.num_threads = 4;
        options.max_num_iterations = num_iterations;
        
        // Convergence criteria
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-8;
        
        options.minimizer_progress_to_stdout = false;
        
        // Run the solver
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // Optional: Log optimization results
        if (summary.termination_type == ceres::CONVERGENCE) {
            RCLCPP_DEBUG(rclcpp::get_logger("SlidingWindowBA"), 
                        "BA converged: %d iterations, final cost: %e", 
                        summary.num_successful_steps, summary.final_cost);
        } else {
            RCLCPP_WARN(rclcpp::get_logger("SlidingWindowBA"), 
                       "BA did not converge: %s", summary.BriefReport().c_str());
        }
    }

    // Get the latest optimized camera pose
    std::pair<cv::Mat, cv::Mat> getLatestPose() const {
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
        
        if (!frame_queue_.empty()) {
            int latest_frame_id = frame_queue_.back();
            auto it = frame_poses_.find(latest_frame_id);
            
            if (it != frame_poses_.end()) {
                it->second->toRt(R, t);
            }
        }
        
        return {R, t};
    }

    // Get a specific camera pose
    std::pair<cv::Mat, cv::Mat> getPose(int frame_id) const {
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
        
        auto it = frame_poses_.find(frame_id);
        if (it != frame_poses_.end()) {
            it->second->toRt(R, t);
        }
        
        return {R, t};
    }

private:
    // Remove landmarks that are no longer observed
    void pruneLandmarks() {
        // First, collect all landmark IDs that are still observed
        std::unordered_set<int> observed_landmarks;
        for (const auto& obs : observations_) {
            observed_landmarks.insert(obs.landmark_id);
        }
        
        // Then remove any landmarks not in this set
        auto it = landmarks_.begin();
        while (it != landmarks_.end()) {
            if (observed_landmarks.find(it->first) == observed_landmarks.end()) {
                it = landmarks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Camera parameters (fixed - not optimized)
    double fx_, fy_, cx_, cy_;
    double sigma_pixels_;  // Isotropic Gaussian noise standard deviation
    
    // Sliding window parameters
    int window_size_;
    std::deque<int> frame_queue_;
    
    // Data structures
    std::map<int, std::shared_ptr<CameraPose>> frame_poses_;
    std::map<int, std::shared_ptr<Landmark>> landmarks_;
    std::vector<Observation> observations_;
    
    // ID counters
    int next_landmark_id_;
    int next_frame_id_;
};

#endif // BUNDLE_ADJUSTMENT_HPP