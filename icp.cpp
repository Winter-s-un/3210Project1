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

void findCorrespondences(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, std::vector<int>& correspondences, KDTree& kdtree) {
    for (const pcl::PointXYZ& src_point : *src_cloud) {
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        kdtree.nearestKSearch(src_point, 1, indices, distances);
        correspondences.push_back(indices[0]);
    }
}

typedef Eigen::Matrix4d TransformationMatrix;

// Compute the transformation matrix (including rotation and translation)
TransformationMatrix computeTransformation(const pcl::PointCloud<PointType>::Ptr src_cloud, const pcl::PointCloud<PointType>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // Compute the centroid of the source cloud
    Eigen::Vector3d src_centroid(0, 0, 0);
    for (size_t i = 0; i < src_cloud->size(); ++i) {
        src_centroid.x() += src_cloud->points[i].x;
        src_centroid.y() += src_cloud->points[i].y;
        src_centroid.z() += src_cloud->points[i].z;
    }
    src_centroid /= src_cloud->size();

    // Compute the centroid of the target cloud
    Eigen::Vector3d tar_centroid(0, 0, 0);
    for (size_t i = 0; i < correspondences.size(); ++i) {
        tar_centroid.x() += tar_cloud->points[correspondences[i]].x;
        tar_centroid.y() += tar_cloud->points[correspondences[i]].y;
        tar_centroid.z() += tar_cloud->points[correspondences[i]].z;
    }
    tar_centroid /= correspondences.size();

    // Compute the covariance matrix
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < correspondences.size(); ++i) {
        Eigen::Vector3d src_point, tar_point;
        src_point.x() = src_cloud->points[i].x - src_centroid.x();
        src_point.y() = src_cloud->points[i].y - src_centroid.y();
        src_point.z() = src_cloud->points[i].z - src_centroid.z();
        tar_point.x() = tar_cloud->points[correspondences[i]].x - tar_centroid.x();
        tar_point.y() = tar_cloud->points[correspondences[i]].y - tar_centroid.y();
        tar_point.z() = tar_cloud->points[correspondences[i]].z - tar_centroid.z();

        covariance_matrix += src_point * tar_point.transpose();
    }
    covariance_matrix /= correspondences.size();

    // Compute the Singular Value Decomposition (SVD) to find rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    // Compute the translation vector
    Eigen::Vector3d translation = tar_centroid - rotation_matrix * src_centroid;

    // Construct the transformation matrix
    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block(0, 0, 3, 3) = rotation_matrix;
    transformation.block(0, 3, 3, 1) = translation;

    return transformation;
}


Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    int iteration = 0;
    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);
    Eigen::Matrix4d transformation = init_guess;

    while (iteration < params::max_iterations) {
        std::vector<int> correspondences;
        findCorrespondences(src_cloud, tar_cloud, correspondences, kdtree);
        Eigen::Matrix4d delta_transformation = computeTransformation(src_cloud, tar_cloud, correspondences);
        transformation = delta_transformation * transformation;
        double transformation_change = (delta_transformation - Eigen::Matrix4d::Identity()).norm();

        if (transformation_change < 1e-6) {
            break;
        }

        iteration++;
    }

    std::cout << "Final Transformation:" << std::endl;
    std::cout << transformation << std::endl;

    return transformation;
}



