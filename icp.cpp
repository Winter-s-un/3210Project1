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

        // 使用KD树查找最近邻点的索引
        kdtree.nearestKSearch(src_point, 1, indices, distances);

        // 添加最近邻点的索引到correspondences数组
        correspondences.push_back(indices[0]);
    }
}


// 使用Eigen库的Matrix4d定义变换矩阵
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

// 使用最小二乘法计算平移向量
Eigen::Vector3d computeTranslation(const std::vector<Eigen::Vector3d>& mean_removed_points) {
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();
    
    // 如果点对的数量小于3，无法估计平移向量，返回零向量
    if (mean_removed_points.size() < 3) {
        return translation_vector;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> A(mean_removed_points.size(), 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> B(mean_removed_points.size());

    // 填充矩阵 A 和向量 B
    for (size_t i = 0; i < mean_removed_points.size(); ++i) {
        A.row(i) = mean_removed_points[i].transpose();
        B(i) = -mean_removed_points[i].squaredNorm();
    }

    // 使用最小二乘法求解线性系统 Ax = B
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(B);

    // 估计的平移向量为 x
    translation_vector = x;

    return translation_vector;
}

// 使用SVD计算变换矩阵
TransformationMatrix computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // 步骤1：计算均值移除后的点对列表
    std::vector<Eigen::Vector3d> mean_removed_points = computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences);

    // 步骤2：计算平移向量
    Eigen::Vector3d translation_vector = computeTranslation(mean_removed_points);

    // 步骤3：计算协方差矩阵
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        covariance_matrix += point * point.transpose();
    }
    covariance_matrix /= mean_removed_points.size();

    // 步骤4：SVD分解协方差矩阵
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    // 步骤5：构建变换矩阵
    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block<3, 1>(0, 3) = translation_vector; // 添加平移部分到变换矩阵

    return transformation;
}

Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);
    Eigen::Matrix4d transformation;

    transformation = init_guess;

    int iteration = 0;
    while (iteration < params::max_iterations) {
        std::vector<int> correspondences;
        
        // 在迭代过程中应用真实变换矩阵
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*src_cloud, *transformed_src_cloud, transformation);

        findCorrespondences(transformed_src_cloud, tar_cloud, correspondences, kdtree);
        
        // 计算变换矩阵
        Eigen::Matrix4d delta_transformation = computeTransformation(transformed_src_cloud, tar_cloud, correspondences);

        // 更新变换矩阵
        transformation = delta_transformation * transformation;

        // 计算变换矩阵的变化量
        double transformation_change = (delta_transformation - Eigen::Matrix4d::Identity()).norm();

        // 收敛检查：如果变换矩阵的变化量小于收敛阈值，退出迭代
        if (transformation_change < 1e-6) {
            break;
        }

        iteration++;
    }

    return transformation;
}


