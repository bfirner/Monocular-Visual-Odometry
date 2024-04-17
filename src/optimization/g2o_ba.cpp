/* This script is mainly copied and then modified from Chapter 7 of Dr. Xiang Gao's book. Link is here:
https://github.com/gaoxiang12/slambook/blob/master/ch7/pose_estimation_3d2d.cpp
*/

#include "my_slam/optimization/g2o_ba.h"

#include "my_slam/basics/eigen_funcs.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

#include <chrono> // timer

namespace my_slam
{
namespace optimization
{

Eigen::Matrix2d mat2eigen(const cv::Mat &mat)
{
    Eigen::Matrix2d mat_eigen;
    mat_eigen << mat.at<double>(0, 0), mat.at<double>(0, 1), mat.at<double>(1, 0), mat.at<double>(1, 1);
    return mat_eigen;
}

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

void bundleAdjustment(
    const vector<vector<cv::Point2f *>> &v_pts_2d,
    const vector<vector<int>> &v_pts_2d_to_3d_idx,
    const cv::Mat &K,
    std::unordered_map<int, cv::Point3f *> &pts_3d,
    vector<cv::Mat *> &v_camera_g2o_poses,
    const cv::Mat &information_matrix,
    bool is_fix_map_pts, bool is_update_map_pts)
{

    // Change pose format from OpenCV to Sophus::SE3<double>
    int num_frames = v_camera_g2o_poses.size();
    // vector<Sophus::SE3<double>, aligned_allocator<Sophus::SE3<double>>> v_T_cam_to_world;
    vector<Sophus::SE3<double>> v_T_cam_to_world;
    for (int i = 0; i < num_frames; i++)
    {
        v_T_cam_to_world.push_back(
            basics::transT_cv2sophus((*v_camera_g2o_poses[i]).inv()));
    }

    // Init g2o
    /*Old
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block; // dim(pose) = 6, dim(landmark) = 3
    Block::LinearSolverType *linearSolver;
    // linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // solver for linear equation
    linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(linearSolver); // solver for matrix block
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    */

    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg solver(std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
    solver.setUserLambdaInit(1);
    // If the G2O_DELETE_IMPLICITLY_OWNED_OBJECTS define was set then g2o may try to free the
    // solver.
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(&solver);

    // -- Add vertex: parameters to optimize
    int vertex_id = 0;
    // Camera pose
    vector<g2o::VertexSE3Expmap *> g2o_poses;
    for (int ith_frame = 0; ith_frame < num_frames; ith_frame++)
    {
        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose
        pose->setId(vertex_id++);
        // if (num_frames > 1 && ith_frame == num_frames - 1)
        // pose->setFixed(true); // Fix the last one -- which is the earliest frame
        pose->setEstimate(g2o::SE3Quat(
            v_T_cam_to_world[ith_frame].rotationMatrix(),
            v_T_cam_to_world[ith_frame].translation()));
        optimizer.addVertex(pose);
        g2o_poses.push_back(pose);
    }
    // Parameter: camera intrinsics
    g2o::CameraParameters *camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // Points pos in world frame
    std::unordered_map<int, g2o::VertexPointXYZ *> g2o_points_3d;
    std::unordered_map<int, int> pts3dID_to_vertexID;
    for (auto it = pts_3d.begin(); it != pts_3d.end(); it++) // landmarks
    {
        int pt3d_id = it->first;
        cv::Point3f *p = it->second;

        g2o::VertexPointXYZ *point = new g2o::VertexPointXYZ();
        point->setId(vertex_id);
        if (is_fix_map_pts)
            point->setFixed(true);

        pts3dID_to_vertexID[pt3d_id] = vertex_id;
        vertex_id++;
        point->setEstimate(Eigen::Vector3d(p->x, p->y, p->z));
        point->setMarginalized(true); // g2o must set marg
        optimizer.addVertex(point);
        g2o_points_3d[pt3d_id] = point;
    }

    // -- Add edges, which define the error/cost function.

    // Set information matrix
    int edge_id = 0;
    Eigen::Matrix2d information_matrix_eigen = mat2eigen(information_matrix);
    for (int ith_frame = 0; ith_frame < num_frames; ith_frame++)
    {
        int num_pts_2d = v_pts_2d[ith_frame].size();
        for (int j = 0; j < num_pts_2d; j++)
        {
            const cv::Point2f *p = v_pts_2d[ith_frame][j];
            int pt3d_id = v_pts_2d_to_3d_idx[ith_frame][j];

            g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
            edge->setId(edge_id++);
            edge->setVertex(0, // XYZ point
                            dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(pts3dID_to_vertexID[pt3d_id])));
            edge->setVertex(1, // camera pose
                            dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(ith_frame)));

            edge->setMeasurement(Eigen::Vector2d(p->x, p->y));
            edge->setParameterId(0, 0);
            edge->setInformation(information_matrix_eigen);
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }
    }

    // -- Optimize
    constexpr bool is_print_time_and_res = false;
    int optimize_iters = 50;
    if (is_print_time_and_res)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        optimizer.optimize(optimize_iters);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    }
    else
    {
        optimizer.initializeOptimization();
        optimizer.optimize(optimize_iters);
    }

    // --------------------------------------------------
    // -- Final: get the result from solver

    printf("BA: Number of frames = %d, 3d points = %d\n", num_frames, vertex_id - num_frames);

    // 1. Camera pose
    for (int i = 0; i < num_frames; i++)
    {
        Sophus::SE3<double> T_cam_to_world = Sophus::SE3<double>(
            g2o_poses[i]->estimate().rotation(),
            g2o_poses[i]->estimate().translation());
        cv::Mat pose_src = basics::transT_sophus2cv(T_cam_to_world).inv(); // Change data format back to OpenCV
        pose_src.copyTo(*v_camera_g2o_poses[i]);
    }

    // 2. Points 3d world pos // This makes the performance bad
    for (auto it = pts_3d.begin(); is_update_map_pts && it != pts_3d.end(); it++)
    {
        int pt3d_id = it->first;
        cv::Point3f *p = it->second;
        Eigen::Vector3d p_res = g2o_points_3d[pt3d_id]->estimate();
        p->x = p_res(0, 0);
        p->y = p_res(1, 0);
        p->z = p_res(2, 0);
    }
}

} // namespace optimization
} // namespace my_slam
