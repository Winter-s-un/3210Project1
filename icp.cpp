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

void findCorrespondences(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, std::vector<int>& correspondences, KDTree& kdtree，const TransformationMatrix& transformation) {
    for (const pcl::PointXYZ& src_point : *src_cloud) {
        Eigen::Vector4f transformed_point;
        transformed_point = transformation * src_point.getVector4fMap();

        pcl::PointXYZ transformed_src_point;
        transformed_src_point.x = transformed_point[0];
        transformed_src_point.y = transformed_point[1];
        transformed_src_point.z = transformed_point[2];

        std::vector<int> indices(1);
        std::vector<float> distances(1);

        kdtree.nearestKSearch(transformed_src_point, 1, indices, distances);

        correspondences.push_back(indices[0]);
    }
}

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

TransformationMatrix computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    std::vector<Eigen::Vector3d> mean_removed_points = computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences);
    Eigen::Vector3d translation_vector = computeTranslation(mean_removed_points);
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();

    for (const Eigen::Vector3d& point : mean_removed_points) {
        covariance_matrix += point * point.transpose();
    }

    covariance_matrix /= mean_removed_points.size();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block<3, 1>(0, 3) = translation_vector;

    return transformation;
}

Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);
    Eigen::Matrix4d transformation = init_guess;

    int iteration = 0;

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

    return transformation;
}
