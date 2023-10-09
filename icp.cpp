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



// 计算变换矩阵
Eigen::Matrix4d computeTransformation(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud,
                                      const std::vector<int>& correspondences) {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();

    // 计算源点云和目标点云的质心
    Eigen::Vector4d src_centroid(0.0, 0.0, 0.0, 0.0);
    Eigen::Vector4d tar_centroid(0.0, 0.0, 0.0, 0.0);

    for (int idx : correspondences) {
        src_centroid += src_cloud->at(idx).getVector4fMap();
        tar_centroid += tar_cloud->at(idx).getVector4fMap();
    }

    src_centroid /= correspondences.size();
    tar_centroid /= correspondences.size();

    // 计算中心化后的点云
    Eigen::Matrix3Xd centered_src(3, correspondences.size());
    Eigen::Matrix3Xd centered_tar(3, correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i) {
        pcl::PointXYZ& src_point = src_cloud->at(correspondences[i]);
        pcl::PointXYZ& tar_point = tar_cloud->at(correspondences[i]);

        centered_src.col(i) = src_point.getVector3fMap().head(3) - src_centroid.head(3);
        centered_tar.col(i) = tar_point.getVector3fMap().head(3) - tar_centroid.head(3);
    }

    // 计算协方差矩阵 H
    Eigen::Matrix3d H = centered_src * centered_tar.transpose();

    // 使用奇异值分解 (SVD) 计算旋转矩阵 R
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * (svd.matrixV().transpose());

    // 如果矩阵 R 的行列式小于零，需要进行修正
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1; // 反转最后一列
        R = svd.matrixU() * V.transpose();
    }

    // 计算平移矩阵 t
    Eigen::Vector3d t = tar_centroid.head(3) - R * src_centroid.head(3);

    // 构建最终的变换矩阵
    transformation.block<3, 3>(0, 0) = R;
    transformation.block<3, 1>(0, 3) = t;
    
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
