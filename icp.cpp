#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// 定义点类型
struct Point {
    double x, y, z;
};

// 定义点云类型
typedef std::vector<Point> PointCloud;

// 定义变换矩阵类型
typedef Eigen::Matrix4d TransformationMatrix;

// 创建示例点云数据
PointCloud source_cloud, target_cloud;
// 在实际应用中，你需要从文件或传感器中读取点云数据

// 定义真实的变换矩阵
TransformationMatrix true_transformation;
true_transformation << 0.981995, 0.188913, -0.00021171, -0.920981,
                      -0.188913, 0.981996, 0.00015824, 0.160313,
                       0.000237872, -0.000115197, 1, 0.0013342,
                       0, 0, 0, 1;

// 定义点云配准函数
TransformationMatrix icp(const PointCloud& source, const PointCloud& target, int max_iterations = 100, double convergence_threshold = 1e-6) {
    TransformationMatrix estimated_transformation = TransformationMatrix::Identity();

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // 步骤1：寻找对应关系，这里简化为一对一的对应关系
        std::vector<Point> matched_source, matched_target;
        for (size_t i = 0; i < source.size(); ++i) {
            // 将源点云中的点根据估计的变换映射到目标点云中
            Point transformed_source_point;
            transformed_source_point.x = estimated_transformation(0, 0) * source[i].x + estimated_transformation(0, 1) * source[i].y + estimated_transformation(0, 3);
            transformed_source_point.y = estimated_transformation(1, 0) * source[i].x + estimated_transformation(1, 1) * source[i].y + estimated_transformation(1, 3);
            transformed_source_point.z = estimated_transformation(2, 3);  // 假设只有平移

            // 在目标点云中查找最近邻点
            Point closest_target_point = target[0];  // 初始化为目标点云的第一个点
            double min_distance = std::sqrt(std::pow(transformed_source_point.x - target[0].x, 2) +
                                            std::pow(transformed_source_point.y - target[0].y, 2));

            for (size_t j = 1; j < target.size(); ++j) {
                double distance = std::sqrt(std::pow(transformed_source_point.x - target[j].x, 2) +
                                            std::pow(transformed_source_point.y - target[j].y, 2));
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_target_point = target[j];
                }
            }

            // 将对应的点添加到匹配点集
            matched_source.push_back(source[i]);
            matched_target.push_back(closest_target_point);
        }

        // 步骤2：计算变换矩阵，这里使用最小二乘法
        TransformationMatrix delta_transformation = TransformationMatrix::Identity();
        int num_correspondences = matched_source.size();

        // 计算均值
        double mean_source_x = 0.0, mean_source_y = 0.0, mean_target_x = 0.0, mean_target_y = 0.0;
        for (int i = 0; i < num_correspondences; ++i) {
            mean_source_x += matched_source[i].x;
            mean_source_y += matched_source[i].y;
            mean_target_x += matched_target[i].x;
            mean_target_y += matched_target[i].y;
        }
        mean_source_x /= num_correspondences;
        mean_source_y /= num_correspondences;
        mean_target_x /= num_correspondences;
        mean_target_y /= num_correspondences;

        // 计算变换的各项
        double sxx = 0.0, sxy = 0.0, syx = 0.0, syy = 0.0, tx = 0.0, ty = 0.0;
        for (int i = 0; i < num_correspondences; ++i) {
            double source_x = matched_source[i].x - mean_source_x;
            double source_y = matched_source[i].y - mean_source_y;
            double target_x = matched_target[i].x - mean_target_x;
            double target_y = matched_target[i].y - mean_target_y;

            sxx += source_x * target_x;
            sxy += source_x * target_y;
            syx += source_y * target_x;
            syy += source_y * target_y;
            tx += target_x;
            ty += target_y;
        }

        double det = sxx * syy - sxy * syx;
        if (det != 0.0) {
            delta_transformation(0, 0) = (sxx * tx - sxy * ty) / det;
            delta_transformation(0, 1) = (sxy * tx - syy * ty) / det;
            delta_transformation(1, 0) = (syx * tx - sxx * ty) / det;
            delta_transformation(1, 1) = (syy * tx - sxy * ty) / det;
            delta_transformation(0, 3) = mean_target_x - (delta_transformation(0, 0) * mean_source_x + delta_transformation(0, 1) * mean_source_y);
            delta_transformation(1, 3) = mean_target_y - (delta_transformation(1, 0) * mean_source_x + delta_transformation(1, 1) * mean_source_y);
        }

        // 步骤3：更新估计的变换矩阵
        estimated_transformation = delta_transformation * estimated_transformation;

        // 步骤4：检查收敛条件
        // 计算估计的变换前后的差异
        TransformationMatrix delta = delta_transformation - TransformationMatrix::Identity();
        double delta_norm = delta.norm();

        // 如果差异小于收敛阈值，退出迭代
        if (delta_norm < convergence_threshold) {
            break;
        }
    }

    return estimated_transformation;
}

int main() {
    // 执行ICP配准
    TransformationMatrix estimated_transform = icp(source_cloud, target_cloud);

    // 输出估计的变换矩阵
    std::cout << "Estimated Transformation Matrix:" << std::endl << estimated_transform << std::endl;

    return 0;
}
