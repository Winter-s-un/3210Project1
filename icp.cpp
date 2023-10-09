#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

// 定义点类型
struct Point {
    double x, y, z;
};

// 定义点云类型
typedef std::vector<Point> PointCloud;

// 使用PCL定义的点云类型
typedef pcl::PointXYZ PointPCL;
typedef pcl::PointCloud<PointPCL> PointCloudPCL;

// 创建示例点云数据
PointCloud source_cloud, target_cloud;
// 在实际应用中，你需要从文件或传感器中读取点云数据

// 定义真实的变换矩阵
Eigen::Matrix4d true_transformation;
true_transformation << 0.981995, 0.188913, -0.00021171, -0.920981,
                      -0.188913, 0.981996, 0.00015824, 0.160313,
                       0.000237872, -0.000115197, 1, 0.0013342,
                       0, 0, 0, 1;

// 定义KD树类型
typedef pcl::KdTreeFLANN<PointPCL> KDTree;

// 寻找最近邻点对应关系
void findCorrespondences(const PointCloudPCL::Ptr source, const PointCloudPCL::Ptr target, std::vector<int>& correspondences, KDTree& kdtree) {
    pcl::Correspondences all_correspondences;
    pcl::registration::CorrespondenceEstimation<PointPCL, PointPCL> est;
    est.setInputSource(source);
    est.setInputTarget(target);
    est.determineReciprocalCorrespondences(all_correspondences);

    correspondences.clear();
    correspondences.reserve(all_correspondences.size());
    for (const auto& corr : all_correspondences) {
        correspondences.push_back(corr.index_query);
    }
}


// 计算变换矩阵
Eigen::Matrix4d computeTransformation(const PointCloudPCL::Ptr source, const PointCloudPCL::Ptr target, const std::vector<int>& correspondences) {
    // 创建源点云和目标点云的中心化版本
    Eigen::Matrix<double, 3, Eigen::Dynamic> centered_source(3, correspondences.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> centered_target(3, correspondences.size());

    // 填充中心化点云矩阵
    for (size_t i = 0; i < correspondences.size(); ++i) {
        int index = correspondences[i];
        centered_source.col(i) << source->points[index].x, source->points[index].y, source->points[index].z;
        centered_target.col(i) << target->points[index].x, target->points[index].y, target->points[index].z;
    }

    // 计算质心
    Eigen::Vector3d source_centroid = centered_source.rowwise().mean();
    Eigen::Vector3d target_centroid = centered_target.rowwise().mean();

    // 中心化点云
    centered_source.colwise() -= source_centroid;
    centered_target.colwise() -= target_centroid;

    // 计算协方差矩阵
    Eigen::Matrix3d covariance_matrix = centered_source * centered_target.transpose();

    // 执行SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // 构建估计的变换矩阵
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Vector3d translation_vector = target_centroid - rotation_matrix * source_centroid;

    Eigen::Matrix4d estimated_transformation = Eigen::Matrix4d::Identity();
    estimated_transformation.block<3, 3>(0, 0) = rotation_matrix;
    estimated_transformation.block<3, 1>(0, 3) = translation_vector;

    return estimated_transformation;
}

// 主要ICP函数
Eigen::Matrix4d icp(const PointCloudPCL::Ptr source, const PointCloudPCL::Ptr target, int max_iterations = 100, double convergence_threshold = 1e-6) {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    int iteration = 0;

    KDTree kdtree;
    kdtree.setInputCloud(target);

    while (iteration < max_iterations) {
        std::vector<int> correspondences;
        findCorrespondences(source, target, correspondences, kdtree);

        // 计算变换矩阵
        Eigen::Matrix4d delta_transformation = computeTransformation(source, target, correspondences);

        // 更新变换矩阵
        transformation = delta_transformation * transformation;

        // 计算变换矩阵的变化量
        double transformation_change = (delta_transformation - Eigen::Matrix4d::Identity()).norm();

        // 收敛检查：如果变换矩阵的变化量小于收敛阈值，退出迭代
        if (transformation_change < convergence_threshold) {
            break;
        }

        iteration++;
    }

    return transformation;
}


int main() {
    // 初始化PCL点云
    PointCloudPCL::Ptr source_cloud_pcl = createPCLPointCloud(source_cloud);
    PointCloudPCL::Ptr target_cloud_pcl = createPCLPointCloud(target_cloud);

    // 执行ICP配准
    Eigen::Matrix4d estimated_transform = icp(source_cloud_pcl, target_cloud_pcl);

    // 输出估计的变换矩阵
    std::cout << "Estimated Transformation Matrix:" << std::endl << estimated_transform << std::endl;

    return 0;
}
