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
         std::cout << "Source point: (" << src_point.x << ", " << src_point.y << ", " << src_point.z << ") ";
        std::cout << "Target point: (" << tar_cloud->points[indices[0]].x << ", " << tar_cloud->points[indices[0]].y << ", " << tar_cloud->points[indices[0]].z << ")" << std::endl;
    }
}

typedef Eigen::Matrix4d TransformationMatrix;

// 改进计算平移向量的方法
Eigen::Vector3d computeTranslation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < correspondences.size(); ++i) {
        pcl::PointXYZ src_point = src_cloud->points[i];
        pcl::PointXYZ tar_point = tar_cloud->points[correspondences[i]];

        translation_vector += tar_point.getVector3fMap() - src_point.getVector3fMap();
    }
    translation_vector /= correspondences.size();

    return translation_vector;
}

// 改进计算旋转矩阵的方法
Eigen::Matrix4d computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // 计算平移向量
    Eigen::Vector3d translation_vector = computeTranslation(src_cloud, tar_cloud, correspondences);

    // 计算旋转矩阵
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity(); // 初始化为单位矩阵

    // 添加更多代码来计算旋转矩阵（例如SVD方法）
    
    // 构建变换矩阵
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block(0, 3, 3, 1) = translation_vector;

    return transformation;
}


Eigen::Matrix4d icp_registration(PointCloud::Ptr src_cloud, PointCloud::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);

    Eigen::Matrix4d transformation = init_guess;
    int max_iterations = 50;
    double transformation_epsilon = 1e-6;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::vector<int> correspondences;
        std::vector<float> distances;

        for (const PointType& src_point : *src_cloud) {
            PointType src_point_2d;
            src_point_2d.x = src_point.x;
            src_point_2d.y = src_point.y;
            src_point_2d.z = 0.0;  // Ignore Z coordinate

            std::vector<int> indices(1);
            kdtree.nearestKSearch(src_point_2d, 1, indices, distances);

            correspondences.push_back(indices[0]);
        }

        pcl::registration::TransformationEstimationSVD<PointType, PointType> svd;
        Eigen::Matrix4d delta_transformation;
        svd.estimateRigidTransformation(*src_cloud, *tar_cloud, correspondences, delta_transformation);

        transformation = delta_transformation * transformation;

        double transformation_change = (delta_transformation - Eigen::Matrix4d::Identity()).norm();

        if (transformation_change < transformation_epsilon) {
            break;
        }
    }

    return transformation;
}



