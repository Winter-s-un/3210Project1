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
typedef Eigen::Matrix4d TransformationMatrix;
void findCorrespondences(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, std::vector<int>& correspondences, KDTree& kdtree, const TransformationMatrix& transformation) {
    for (const pcl::PointXYZ& src_point : *src_cloud) {
        Eigen::Vector4f src_point_homogeneous;
        src_point_homogeneous << src_point.x, src_point.y, src_point.z, 1.0;

        // 应用变换矩阵，包括平移部分
        Eigen::Vector4f transformed_point_homogeneous = transformation * src_point_homogeneous;

        // 创建 pcl::PointXYZ 对象来表示变换后的点
        pcl::PointXYZ transformed_src_point;
        transformed_src_point.x = transformed_point_homogeneous[0];
        transformed_src_point.y = transformed_point_homogeneous[1];
        transformed_src_point.z = transformed_point_homogeneous[2];

        std::vector<int> indices(1);
        std::vector<float> distances(1);

        kdtree.nearestKSearch(transformed_src_point, 1, indices, distances);

        correspondences.push_back(indices[0]);
    }
}


// 计算旋转矩阵和平移向量
void computeRotationAndTranslation(const Eigen::Matrix4d& transformation, Eigen::Matrix3d& rotation, Eigen::Vector3d& translation) {
    rotation = transformation.block<3, 3>(0, 0);
    translation = transformation.block<3, 1>(0, 3);
}

// 计算变换矩阵（包括旋转和平移）
TransformationMatrix computeTransformation(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;
    return transformation;
}

// 计算均值移除后的点对列表
std::vector<Eigen::Vector3d> computeMeanRemovedPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences, const TransformationMatrix& transformation) {
    std::vector<Eigen::Vector3d> mean_removed_points;

    for (size_t i = 0; i < correspondences.size(); ++i) {
        pcl::PointXYZ src_point = src_cloud->points[i];
        pcl::PointXYZ tar_point = tar_cloud->points[correspondences[i]];

        Eigen::Vector4d src_point_homogeneous, tar_point_homogeneous;
        src_point_homogeneous << src_point.x, src_point.y, src_point.z, 1;
        tar_point_homogeneous << tar_point.x, tar_point.y, tar_point.z, 1;

        // 应用变换矩阵
        Eigen::Vector4d transformed_src_point = transformation * src_point_homogeneous;

        Eigen::Vector3d mean_removed_point;
        mean_removed_point(0) = transformed_src_point(0) - tar_point_homogeneous(0);
        mean_removed_point(1) = transformed_src_point(1) - tar_point_homogeneous(1);
        mean_removed_point(2) = transformed_src_point(2) - tar_point_homogeneous(2);

        mean_removed_points.push_back(mean_removed_point);
    }

    return mean_removed_points;
}

// 使用SVD计算旋转矩阵
Eigen::Matrix3d computeRotationMatrix(const std::vector<Eigen::Vector3d>& mean_removed_points) {
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        covariance_matrix += point * point.transpose();
    }
    covariance_matrix /= mean_removed_points.size();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    return rotation_matrix;
}

// 使用最小二乘法计算平移向量
Eigen::Vector3d computeTranslation(const std::vector<Eigen::Vector3d>& mean_removed_points) {
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();

    if (mean_removed_points.size() < 3) {
        return translation_vector;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> A(mean_removed_points.size(), 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> B(mean_removed_points.size());

    for (size_t i = 0; i < mean_removed_points.size(); ++i) {
        A.row(i) = mean_removed_points[i].transpose();
        B(i) = -mean_removed_points[i].squaredNorm();
    }

    Eigen::Vector3d x = A.colPivHouseholderQr().solve(B);
    translation_vector = x;

    return translation_vector;
}

// 使用ICP算法计算变换矩阵（包括旋转和平移）
TransformationMatrix icp_registration(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const TransformationMatrix& init_guess) {
       int iteration = 0;
    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);

    TransformationMatrix transformation = init_guess;

    while (iteration < params::max_iterations) {
        std::vector<int> correspondences;
        findCorrespondences(src_cloud, tar_cloud, correspondences, kdtree);

        // 计算旋转矩阵
        Eigen::Matrix3d rotation_matrix = computeRotationMatrix(computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences, transformation));

        // 计算平移向量
        Eigen::Vector3d translation_vector = computeTranslation(computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences, transformation));

        // 构建新的变换矩阵
        transformation = computeTransformation(rotation_matrix, translation_vector);

        // 计算变换矩阵的变化量
        double transformation_change = (transformation - init_guess).norm();

        // 收敛检查：如果变换矩阵的变化量小于收敛阈值，退出迭代
        if (transformation_change < 1e-6) {
            break;
        }

        iteration++;
    }

    return transformation;
}
