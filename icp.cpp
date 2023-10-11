/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/
#include "icp.h"
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h> //include Kdtree header
#include "parameters.h"
#include <vector>
#include <cmath>


typedef pcl::KdTreeFLANN<pcl::PointXYZ> KDTree;

void findCorrespondences(const PointCloud::Ptr src_cloud, const PointCloud::Ptr tar_cloud, std::vector<int>& correspondences, pcl::KdTreeFLANN<PointType>& kdtree) {
    for (const PointType& src_point : *src_cloud) {
        std::vector<int> indices(1);
        std::vector<float> distances(1);

        PointType src_point_xy;
        src_point_xy.x = src_point.x;
        src_point_xy.y = src_point.y;
        src_point_xy.z = 0.0;  // Ignore the Z coordinate

        // Use KD tree to find the index of the nearest neighbor point
        kdtree.nearestKSearch(src_point_xy, 1, indices, distances);

        // Add the index of the nearest neighbor to the correspondences array
        correspondences.push_back(indices[0]);
    }
}

TransformationMatrix computeTransformation(const PointCloud::Ptr src_cloud, const PointCloud::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // Initialize transformation matrix
    TransformationMatrix transformation = TransformationMatrix::Identity();

    // Compute the number of corresponding points
    size_t num_correspondences = correspondences.size();

    if (num_correspondences == 0) {
        return transformation;
    }

    // Compute the centroids of source and target point clouds
    Eigen::Vector2d src_centroid(0, 0);
    Eigen::Vector2d tar_centroid(0, 0);

    for (size_t i = 0; i < num_correspondences; ++i) {
        const PointType& src_point = src_cloud->points[i];
        const PointType& tar_point = tar_cloud->points[correspondences[i]];

        src_centroid.x() += src_point.x;
        src_centroid.y() += src_point.y;
        tar_centroid.x() += tar_point.x;
        tar_centroid.y() += tar_point.y;
    }

    src_centroid /= num_correspondences;
    tar_centroid /= num_correspondences;

    // Compute the covariance matrix
    Eigen::Matrix2d covariance_matrix = Eigen::Matrix2d::Zero();

    for (size_t i = 0; i < num_correspondences; ++i) {
        PointType src_point = src_cloud->points[i];
        PointType tar_point = tar_cloud->points[correspondences[i]];

        src_point.x -= src_centroid.x();
        src_point.y -= src_centroid.y();

        tar_point.x -= tar_centroid.x();
        tar_point.y -= tar_centroid.y();

        
    Eigen::Vector2d src_vec(src_point.x, src_point.y);
    Eigen::Vector2d tar_vec(tar_point.x, tar_point.y);

    covariance_matrix += src_vec * tar_vec.transpose();
    }

    covariance_matrix /= num_correspondences;

    // Compute the Singular Value Decomposition (SVD) to find rotation
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    // Compute the translation vector
    Eigen::Vector2d translation = tar_centroid - rotation_matrix * src_centroid;

    // Fill in the transformation matrix
    transformation.block(0, 0, 2, 2) = rotation_matrix;
    transformation.block(0, 3, 2, 1) = translation;

    return transformation;
}

TransformationMatrix icp_registration(const PointCloud::Ptr src_cloud, const PointCloud::Ptr tar_cloud, TransformationMatrix init_guess) {
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(tar_cloud);

    TransformationMatrix transformation = init_guess;
    int iteration = 0;

    while (iteration < params::max_iterations) {
        std::vector<int> correspondences;
        findCorrespondences(src_cloud, tar_cloud, correspondences, kdtree);

        // Compute the transformation matrix
        TransformationMatrix delta_transformation = computeTransformation(src_cloud, tar_cloud, correspondences);

        // Update the transformation matrix
        transformation = delta_transformation * transformation;

        // Calculate the change in the transformation matrix
        double transformation_change = (delta_transformation - TransformationMatrix::Identity()).norm();

        // Convergence check: If the change in the transformation matrix is below a threshold, exit the iteration
        if (transformation_change < 1e-6) {
            break;
        }

        iteration++;
    }

    return transformation;
}
