#ifndef MULTI_STEREO_H
#define MULTI_STEREO_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using namespace cv;
using namespace std;
using namespace Eigen;

// ================= NCC =================
double NormalizedCrossCorrelation(const Mat& C0, const Mat& C1);

// ================= Patch Projection =================
bool ComputeConsistencyPatch(
    const Mat& image,
    const Matrix<double, 3, 4>& P,
    const vector<Vector3d>& patch3D,
    Mat& patchOut,
    int window);

// ================= Projection =================
vector<Point2f> project_points(
    const Matrix<double, 3, 4>& P,
    const vector<Vector3d>& points);

// ================= Camera Loader =================
map<string, Matrix<double,3,4>> load_camera_params(const string& filepath);

// ================= Point Cloud =================
void save_point_cloud_obj(
    const string& filename,
    const Mat& depth_map,
    const Mat& image,
    const Vector3d& cam_center,
    const vector<vector<Vector3d>>& rays,
    const Vector3d& min_pt,
    const Vector3d& max_pt);

#endif