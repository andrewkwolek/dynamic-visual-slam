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

// Structure to store camera parameters for optimization
struct CameraPose {
    double rotation[4];
    double translation[3];

    CameraPose() {
        rotation[0] = 1.0; rotation[1] = 0.0; rotation[2] = 0.0; rotation[3] = 0.0;
        translation[0] = 0.0; translation[1] = 0.0; translation[2] = 0.0;
    }

    void fromRt(const cv::Mat& R_world_camera, const cv::Mat& t_world_camera) {
        cv::Mat R_camera_world = R_world_camera.t(); 
        cv::Mat t_camera_world = -R_camera_world * t_world_camera;
        
        Eigen::Matrix3d R_eigen;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eigen(i, j) = R_camera_world.at<double>(i, j);
            }
        }
        
        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        rotation[0] = q.w();
        rotation[1] = q.x();
        rotation[2] = q.y();
        rotation[3] = q.z();
        
        translation[0] = t_camera_world.at<double>(0);
        translation[1] = t_camera_world.at<double>(1);
        translation[2] = t_camera_world.at<double>(2);
    }

    void toRt(cv::Mat& R_world_camera, cv::Mat& t_world_camera) const {
        Eigen::Quaterniond q(rotation[0], rotation[1], rotation[2], rotation[3]);
        Eigen::Matrix3d R_camera_world = q.toRotationMatrix();
        
        Eigen::Matrix3d R_world_camera_eigen = R_camera_world.transpose();
        Eigen::Vector3d t_camera_world_vec(translation[0], translation[1], translation[2]);
        Eigen::Vector3d t_world_camera_vec = -R_world_camera_eigen * t_camera_world_vec;
        
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

// Structure to store a 3D landmark point
struct Landmark {
    uint64_t id;
    std::string category;
    double position[3];
    bool fixed;

    Landmark() : id(0), fixed(false) {
        position[0] = 0.0;
        position[1] = 0.0;
        position[2] = 0.0;
    }

    Landmark(uint64_t landmark_id, std::string cat, double x, double y, double z, bool fixed_point = false) 
        : id(landmark_id), category(cat), fixed(fixed_point) {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }
};

// Structure to store a 2D observation
struct Observation {
    double pixel[2];
    uint64_t landmark_id;
    std::string category;
    int frame_id;

    Observation(double x, double y, uint64_t landmark, std::string cat, int frame) 
        : landmark_id(landmark), category(cat), frame_id(frame) {
        pixel[0] = x;
        pixel[1] = y;
    }
};

// Structure to hold keyframe data
struct KeyframeData {
    int frame_id;
    cv::Mat R;
    cv::Mat t;
    rclcpp::Time timestamp;

    KeyframeData(int id, const cv::Mat& rotation, const cv::Mat& translation, const rclcpp::Time& stamp)
        : frame_id(id), R(rotation.clone()), t(translation.clone()), timestamp(stamp) {}
};

// Result structure for optimization
struct OptimizationResult {
    bool success;
    double final_cost;
    int iterations_completed;
    int frames_optimized;
    int landmarks_optimized;
    std::string message;
    std::chrono::milliseconds optimization_time;
    std::map<int, std::pair<cv::Mat, cv::Mat>> optimized_poses;
    std::map<std::pair<uint64_t, std::string>, cv::Point3d> optimized_landmarks;
};

struct WeightedSquaredReprojectionError {
    WeightedSquaredReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), inv_sigma(1.0 / sigma_pixels) {}

    template <typename T>
    bool operator()(const T* const camera_rotation, const T* const camera_translation,
                    const T* const point, T* residuals) const {
        
        T point_camera[3];
        ceres::QuaternionRotatePoint(camera_rotation, point, point_camera);
        
        point_camera[0] += camera_translation[0];
        point_camera[1] += camera_translation[1]; 
        point_camera[2] += camera_translation[2];
        
        if (point_camera[2] <= T(0.1)) {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        T predicted_x = T(fx) * point_camera[0] / point_camera[2] + T(cx);
        T predicted_y = T(fy) * point_camera[1] / point_camera[2] + T(cy);
        
        T error_x = predicted_x - T(observed_x);
        T error_y = predicted_y - T(observed_y);
        
        residuals[0] = T(inv_sigma) * error_x;
        residuals[1] = T(inv_sigma) * error_y;
        
        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y, double fx, double fy, double cx, double cy, double sigma_pixels) {
        return new ceres::AutoDiffCostFunction<WeightedSquaredReprojectionError, 2, 4, 3, 3>(
            new WeightedSquaredReprojectionError(observed_x, observed_y, fx, fy, cx, cy, sigma_pixels));
    }

    double observed_x, observed_y;
    double fx, fy, cx, cy;
    double inv_sigma;
};

class SlidingWindowBA {
public:
    SlidingWindowBA(double fx, double fy, double cx, double cy, double sigma_pixels = 1.0)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), sigma_pixels_(sigma_pixels) {}

    OptimizationResult optimize(const std::vector<KeyframeData>& keyframes, const std::vector<Landmark>& landmarks, const std::vector<Observation>& observations, int max_iterations = 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        OptimizationResult result;
        result.success = false;
        result.frames_optimized = keyframes.size();
        result.landmarks_optimized = landmarks.size();
        
        if (keyframes.empty() || landmarks.empty() || observations.empty()) {
            result.message = "Empty input data";
            result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            return result;
        }
        
        try {
            std::map<int, std::shared_ptr<CameraPose>> frame_poses;
            std::map<uint64_t, std::shared_ptr<Landmark>> landmark_map;
            
            ceres::Problem problem;
            
            // Set up camera poses
            for (const auto& kf : keyframes) {
                auto pose = std::make_shared<CameraPose>();
                pose->fromRt(kf.R, kf.t);
                frame_poses[kf.frame_id] = pose;
                
                problem.AddParameterBlock(pose->rotation, 4);
                problem.AddParameterBlock(pose->translation, 3);
                problem.SetManifold(pose->rotation, new ceres::EigenQuaternionManifold);
            }
            
            // Fix first camera pose
            if (!keyframes.empty()) {
                auto first_pose = frame_poses[keyframes[0].frame_id];
                problem.SetParameterBlockConstant(first_pose->rotation);
                problem.SetParameterBlockConstant(first_pose->translation);
            }
            
            // Set up landmarks
            for (const auto& landmark : landmarks) {
                auto lm = std::make_shared<Landmark>(landmark);
                landmark_map[landmark.id] = lm;
                
                problem.AddParameterBlock(lm->position, 3);
                
                if (lm->fixed) {
                    problem.SetParameterBlockConstant(lm->position);
                }
            }
            
            // Add observations
            int valid_observations = 0;
            for (const auto& obs : observations) {
                auto frame_it = frame_poses.find(obs.frame_id);
                auto landmark_it = landmark_map.find(obs.landmark_id);
                
                if (frame_it != frame_poses.end() && landmark_it != landmark_map.end()) {
                    ceres::CostFunction* cost_function = WeightedSquaredReprojectionError::Create(
                        obs.pixel[0], obs.pixel[1], fx_, fy_, cx_, cy_, sigma_pixels_
                    );
                    
                    problem.AddResidualBlock(
                        cost_function,
                        new ceres::HuberLoss(1.345),
                        frame_it->second->rotation,
                        frame_it->second->translation,
                        landmark_it->second->position
                    );
                    
                    valid_observations++;
                }
            }
            
            if (valid_observations == 0) {
                result.message = "No valid observations";
                result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time);
                return result;
            }
            
            // Solve
            ceres::Solver::Options options;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.num_threads = 4;
            options.max_num_iterations = max_iterations;
            options.function_tolerance = 1e-6;
            options.gradient_tolerance = 1e-10;
            options.parameter_tolerance = 1e-8;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Extract results
            result.success = (summary.termination_type == ceres::CONVERGENCE);
            result.final_cost = summary.final_cost;
            result.iterations_completed = summary.num_successful_steps;
            result.optimization_time = optimization_time;
            
            if (result.success) {
                result.message = "Optimization converged successfully";
            } else {
                result.message = "Optimization did not converge";
            }
            
            // Extract optimized poses
            for (const auto& [frame_id, pose] : frame_poses) {
                cv::Mat R, t;
                pose->toRt(R, t);
                result.optimized_poses[frame_id] = {R.clone(), t.clone()};
            }
            
            // Extract optimized landmarks
            for (const auto& [landmark_id, landmark] : landmark_map) {
                result.optimized_landmarks[make_pair(landmark_id, landmark->category)] = cv::Point3d(
                    landmark->position[0],
                    landmark->position[1], 
                    landmark->position[2]
                );
            }
            
        } catch (const std::exception& e) {
            result.message = "Exception during optimization: " + std::string(e.what());
            result.optimization_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
        }
        
        return result;
    }

private:
    double fx_, fy_, cx_, cy_;
    double sigma_pixels_;
};

#endif // BUNDLE_ADJUSTMENT_HPP