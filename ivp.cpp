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
//use kdtree to find nearest point
void findCorrespondences(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, std::vector<int>& correspondences, KDTree& kdtree) {
    for (const pcl::PointXYZ& src_point : *src_cloud) {
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        kdtree.nearestKSearch(src_point, 1, indices, distances);
        correspondences.push_back(indices[0]);
    }
}



typedef Eigen::Matrix4d TransformationMatrix;


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

// use svd to cal the transformation
TransformationMatrix computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {

    std::vector<Eigen::Vector3d> mean_removed_points = computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences);


    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        covariance_matrix += point * point.transpose();
    }
    covariance_matrix /= mean_removed_points.size();


    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();


    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block(0, 0, 3, 3) = rotation_matrix;

    return transformation;
}

Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    // This is an example of using pcl::IterativeClosestPoint to align two point clouds
    // In your project, you should implement your own ICP algorithm!!!
    // In your implementation, you can use KDTree in PCL library to find nearest neighbors
    // Use chatGPT, google and github to learn how to use PCL library and implement ICP. But do not copy totally. TA will check your code using advanced tools.
    // If you use other's code, you should add a reference in your report. https://registry.hkust.edu.hk/resource-library/academic-integrity
    /*
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tar_cloud);
    icp.setMaximumIterations(params::max_iterations);  // set maximum iteration
    icp.setTransformationEpsilon(1e-6);  // set transformation epsilon
    icp.setMaxCorrespondenceDistance(params::max_distance);  // set maximum correspondence distance
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp.align(aligned_cloud, init_guess.cast<float>());

    Eigen::Matrix4d transformation = icp.getFinalTransformation().cast<double>();
    return transformation;*/
    Eigen::Matrix4d transformation = init_guess;
    int iteration = 0;

    KDTree kdtree;
    kdtree.setInputCloud(tar_cloud);

    while (iteration < params::max_iterations) {
        std::vector<int> correspondences;
        
        findCorrespondences(src_cloud, tar_cloud, correspondences, kdtree);

        Eigen::Matrix4d delta_transformation = computeTransformation(src_cloud, tar_cloud, correspondences);

        transformation = delta_transformation * transformation;
        
        double transformation_change = (delta_transformation - Eigen::Matrix4d::Identity()).norm();


        if (transformation_change < 1e-8) {
            break;
        }
        iteration++;
    }

    return transformation;
}
