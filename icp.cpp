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

// 使用SVD计算变换矩阵
TransformationMatrix computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, const std::vector<int>& correspondences) {
    // 步骤1：计算均值移除后的点对列表
    std::vector<Eigen::Vector3d> mean_removed_points = computeMeanRemovedPoints(src_cloud, tar_cloud, correspondences);

    // 步骤2：计算协方差矩阵
    Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& point : mean_removed_points) {
        covariance_matrix += point * point.transpose();
    }
    covariance_matrix /= mean_removed_points.size();

    // 步骤3：SVD分解协方差矩阵
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation_matrix = svd.matrixU() * svd.matrixV().transpose();

    // 步骤4：构建变换矩阵
    TransformationMatrix transformation = TransformationMatrix::Identity();
    transformation.block(0, 0, 3, 3) = rotation_matrix;
    // 如果需要，还可以添加平移部分

    return transformation;
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
