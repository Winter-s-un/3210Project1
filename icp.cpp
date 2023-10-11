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

// 计算均值移除后的点对列表
std::vector<Eigen::Vector3d> computeMeanRemovedPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    std::vector<Eigen::Vector3d> mean_removed_points;

    for (size_t i = 0; i < correspondences.size(); ++i) {
        pcl::PointXYZ src_point = src_cloud->points[i];
        pcl::PointXYZ tar_point = tar_cloud->points[correspondences[i]];

        Eigen::Vector3d mean_removed_point;
        mean_removed_point(0) = src_point.x - tar_point.x;
        mean_removed_point(1) = src_point.y - tar_point.y;
        mean_removed_point(2) = src_point.z - tar_point.z;

        mean_removed_points.push_back(mean_removed_point);
    }

    return mean_removed_points;
}

// 使用SVD计算变换矩阵
TransformationMatrix computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // 步骤1：计算均值移除后的点对列表
    std::vector<Eigen::Vector3d> mean_removed_points = computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences);

    // 步骤2：计算均值偏移
    Eigen::Vector3d mean_offset = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        mean_offset += point;
    }
    mean_offset /= mean_removed_points.size();

    // 步骤3：计算协方差矩阵
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        Eigen::Vector3d centered_point = point - mean_offset;
        covariance_matrix += centered_point * centered_point.transpose();
    }
    covariance_matrix /= mean_removed_points.size();

    // 步骤4：SVD分解协方差矩阵
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    // 步骤5：构建变换矩阵
    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block<3, 1>(0, 3) = mean_offset;
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



