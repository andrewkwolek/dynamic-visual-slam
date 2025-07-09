#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "DBoW2/DBoW2.h"
#include "DBoW2/FORB.h"

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
    
    std::vector<cv::Mat> descriptor_vector;
    EXPECT_NO_THROW({
        for (int i = 0; i < descriptors.rows; i++) {
            descriptor_vector.push_back(descriptors.row(i));
        }
    });
    
    EXPECT_EQ(descriptor_vector.size(), static_cast<size_t>(descriptors.rows));
}

TEST_F(DBoW2IntegrationTest, DatabaseOperations) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(dummy_image_, cv::Mat(), keypoints, descriptors);
    
    ASSERT_GT(descriptors.rows, 0);
    
    std::vector<cv::Mat> descriptor_vector;
    for (int i = 0; i < descriptors.rows; i++) {
        descriptor_vector.push_back(descriptors.row(i));
    }
    
    OrbDatabase database;
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
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}