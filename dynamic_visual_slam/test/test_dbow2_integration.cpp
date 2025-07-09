#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "DBoW2/DBoW2.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

// Correct typedefs for ORB descriptors
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbDatabase;

class DBoW2IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create dummy image with features
        dummy_image_ = cv::Mat::zeros(480, 640, CV_8UC1);
        cv::circle(dummy_image_, cv::Point(100, 100), 50, cv::Scalar(255), -1);
        cv::circle(dummy_image_, cv::Point(300, 200), 30, cv::Scalar(255), -1);
        cv::circle(dummy_image_, cv::Point(500, 300), 40, cv::Scalar(255), -1);
        
        orb_ = cv::ORB::create(100);
    }

    cv::Mat dummy_image_;
    cv::Ptr<cv::ORB> orb_;
};

TEST_F(DBoW2IntegrationTest, CreateDBoW2Objects) {
    EXPECT_NO_THROW({
        OrbVocabulary vocabulary;
        OrbDatabase database;
    });
}

TEST_F(DBoW2IntegrationTest, ExtractORBDescriptors) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    ASSERT_NO_THROW({
        orb_->detectAndCompute(dummy_image_, cv::Mat(), keypoints, descriptors);
    });
    
    EXPECT_GT(descriptors.rows, 0) << "Should extract at least some descriptors";
    EXPECT_EQ(descriptors.cols, 32) << "ORB descriptors should be 32 bytes";
}

TEST_F(DBoW2IntegrationTest, ConvertDescriptorsToBoWFormat) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(dummy_image_, cv::Mat(), keypoints, descriptors);
    
    ASSERT_GT(descriptors.rows, 0);
    
    // Convert to DBoW2 format
    std::vector<cv::Mat> descriptor_vector;
    EXPECT_NO_THROW({
        for (int i = 0; i < descriptors.rows; i++) {
            descriptor_vector.push_back(descriptors.row(i));
        }
    });
    
    EXPECT_EQ(descriptor_vector.size(), static_cast<size_t>(descriptors.rows));
}

TEST_F(DBoW2IntegrationTest, BasicDatabaseOperations) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(dummy_image_, cv::Mat(), keypoints, descriptors);
    
    ASSERT_GT(descriptors.rows, 0);
    
    // Convert to DBoW2 format
    std::vector<cv::Mat> descriptor_vector;
    for (int i = 0; i < descriptors.rows; i++) {
        descriptor_vector.push_back(descriptors.row(i));
    }
    
    // Load the pre-trained vocabulary using ROS package path
    OrbVocabulary vocabulary;
    
    // Use ament_index to find the package share directory
    std::string package_share_dir;
    try {
        package_share_dir = ament_index_cpp::get_package_share_directory("dynamic_visual_slam");
    } catch (const std::exception& e) {
        package_share_dir = "/home/kwolek/Northwestern/FinalProject/ws/install/dynamic_visual_slam/share/dynamic_visual_slam";
    }
    
    std::string vocab_path = package_share_dir + "/config/ORBvoc.txt";
    
    bool vocab_loaded = false;
    try {
        vocabulary.loadFromTextFile(vocab_path);
        vocab_loaded = true;
        std::cout << "Loaded vocabulary from: " << vocab_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to load vocabulary from: " << vocab_path << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    EXPECT_TRUE(vocab_loaded) << "Failed to load vocabulary from: " << vocab_path;
    EXPECT_GT(vocabulary.size(), 0) << "Vocabulary should not be empty";
    
    // Create database with the loaded vocabulary
    OrbDatabase database(vocabulary);
    DBoW2::EntryId entry_id;
    
    EXPECT_NO_THROW({
        entry_id = database.add(descriptor_vector);
    });
    
    EXPECT_GE(entry_id, 0) << "Entry ID should be non-negative";
    
    // Test querying
    DBoW2::QueryResults results;
    EXPECT_NO_THROW({
        database.query(descriptor_vector, results, 1);
    });
    
    // We should get at least one result (the entry we just added)
    EXPECT_GT(results.size(), 0) << "Should return at least one query result";
    
    // The first result should be our own entry with a high score
    if (!results.empty()) {
        EXPECT_EQ(results[0].Id, entry_id) << "First result should be the entry we just added";
        EXPECT_GT(results[0].Score, 0.0) << "Score should be positive";
    }
}

// Test to verify DBoW2 headers are accessible
TEST_F(DBoW2IntegrationTest, HeadersAccessible) {
    // This test just verifies we can create the basic DBoW2 objects
    EXPECT_NO_THROW({
        OrbVocabulary vocab;
        OrbDatabase db;
    });
}

// Simple test that doesn't require vocabulary training
TEST_F(DBoW2IntegrationTest, VocabularyCreation) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(dummy_image_, cv::Mat(), keypoints, descriptors);
    
    ASSERT_GT(descriptors.rows, 0);
    
    // Convert to DBoW2 format
    std::vector<cv::Mat> descriptor_vector;
    for (int i = 0; i < descriptors.rows; i++) {
        descriptor_vector.push_back(descriptors.row(i));
    }
    
    // Test vocabulary creation
    OrbVocabulary vocabulary;
    std::vector<std::vector<cv::Mat>> training_features;
    training_features.push_back(descriptor_vector);
    
    // This should not crash
    EXPECT_NO_THROW({
        vocabulary.create(training_features, 2, 1);
    });
    
    // Vocabulary should not be empty after creation
    EXPECT_GT(vocabulary.size(), 0) << "Vocabulary should have nodes after creation";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}