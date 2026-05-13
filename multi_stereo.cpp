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
double NormalizedCrossCorrelation(const Mat& C0, const Mat& C1){
    Mat c0, c1;
    C0.convertTo(c0, CV_32F);
    C1.convertTo(c1, CV_32F);

    Scalar mean0 = mean(c0);
    Scalar mean1 = mean(c1);

    c0 -= mean0;
    c1 -= mean1;

    double n0 = norm(c0);
    double n1 = norm(c1);

    if (n0 < 1e-6 || n1 < 1e-6)
        return -1.0;

    Mat nccMat = (c0 / n0).mul(c1 / n1);
    Scalar s = sum(nccMat);

    return (s[0] + s[1] + s[2]);
}

// ================= Patch Projection =================
bool ComputeConsistencyPatch(
    const Mat& image,
    const Matrix<double, 3, 4>& P,
    const vector<Vector3d>& patch3D,
    Mat& patchOut,
    int window)
{
    int h = image.rows;
    int w = image.cols;

    patchOut = Mat(window, window, CV_8UC3);

    int idx = 0;

    for (int y = 0; y < window; y++)
    {
        for (int x = 0; x < window; x++, idx++)
        {
            if (idx >= patch3D.size()) return false;

            Vector4d X;
            X << patch3D[idx](0), patch3D[idx](1), patch3D[idx](2), 1.0;

            Vector3d p = P * X;

            if (p(2) <= 1e-6) return false;

            float u = float(p(0) / p(2));
            float v = float(p(1) / p(2));

            int ui = int(u);
            int vi = int(v);

            if (ui < 0 || ui >= w || vi < 0 || vi >= h)
                return false;

            patchOut.at<Vec3b>(y, x) = image.at<Vec3b>(vi, ui);
        }
    }
    return true;
}

// ================= Projection =================
vector<Point2f> project_points(
    const Matrix<double, 3, 4>& P,
    const vector<Vector3d>& points)
{
    vector<Point2f> pts2D;
    pts2D.reserve(points.size());

    for (const auto& X : points)
    {
        Vector4d Xh;
        Xh << X(0), X(1), X(2), 1.0;

        Vector3d p = P * Xh;

        if (p(2) <= 1e-6) continue;

        pts2D.emplace_back(float(p(0)/p(2)), float(p(1)/p(2)));
    }

    return pts2D;
}

// ================= Camera Loader =================
map<string, Matrix<double,3,4>> load_camera_params(const string& filepath)
{
    map<string, Matrix<double,3,4>> cameras;
    ifstream file(filepath);

    if (!file.is_open()) {
        cerr << "Failed to open camera file\n";
        return cameras;
    }

    string name;

    while (file >> name)
    {
        Matrix3d K, R;
        Vector3d t;

        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                file >> K(i,j);

        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                file >> R(i,j);

        for(int i=0;i<3;i++)
            file >> t(i);

        Matrix<double,3,4> P;
        P.block<3,3>(0,0) = K * R;
        P.col(3) = K * t;

        cameras[name] = P;
    }

    return cameras;
}

// ================= Point Cloud =================
void save_point_cloud_obj(
    const string& filename,
    const Mat& depth_map,
    const Mat& image,
    const Vector3d& cam_center,
    const vector<vector<Vector3d>>& rays,
    const Vector3d& min_pt,
    const Vector3d& max_pt)
{
    ofstream f(filename);
    f << "# Point Cloud\n";

    int count = 0;

    for(int y=0; y<depth_map.rows; y++)
    {
        for(int x=0; x<depth_map.cols; x++)
        {
            double d = depth_map.at<double>(y,x);
            if (d <= 0) continue;

            Vector3d X = cam_center + d * rays[y][x];

            if ((X.array() >= (min_pt.array()-0.005)).all() &&
                (X.array() <= (max_pt.array()+0.005)).all())
            {
                Vec3b color = image.at<Vec3b>(y,x);

                f << "v "
                  << X(0) << " "
                  << -X(1) << " "
                  << -X(2) << " "
                  << color[2]/255.0 << " "
                  << color[1]/255.0 << " "
                  << color[0]/255.0 << "\n";

                count++;
            }
        }
    }

    cout << "Saved " << count << " points\n";
}

// ================= MAIN =================
int main() {

    string data_dir = "../data/";
    string cam_file = "../data/templeR_par.txt";

    auto cameras = load_camera_params(cam_file);

    Vector3d min_pt(-0.023121, -0.038009, -0.091940);
    Vector3d max_pt( 0.078626,  0.121636, -0.017395);

    vector<Vector3d> corners = {
        {min_pt[0], min_pt[1], min_pt[2]},
        {max_pt[0], min_pt[1], min_pt[2]},
        {max_pt[0], max_pt[1], min_pt[2]},
        {min_pt[0], max_pt[1], min_pt[2]},
        {min_pt[0], min_pt[1], max_pt[2]},
        {max_pt[0], min_pt[1], max_pt[2]},
        {max_pt[0], max_pt[1], max_pt[2]},
        {min_pt[0], max_pt[1], max_pt[2]}
    };

    vector<string> image_names = {
        "templeR0013.png",
        "templeR0014.png",
        "templeR0016.png",
        "templeR0043.png",
        "templeR0045.png"
    };

    vector<Mat> images;
    vector<Matrix<double,3,4>> projections;

    for (const auto& n : image_names)
    {
        Mat img = imread(data_dir + n);
        if (img.empty()) {
            cerr << "Failed to load " << n << endl;
            continue;
        }

        cvtColor(img, img, COLOR_BGR2RGB);

        images.push_back(img);

        if (cameras.find(n) != cameras.end())
            projections.push_back(cameras[n]);
        else
            cerr << "Missing camera for " << n << endl;
    }

    for (int i = 0; i < images.size(); i++)
    {
        Mat debug = images[i].clone();

        auto pts2d = project_points(projections[i], corners);

        vector<pair<int,int>> edges =
        {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };

        for(auto& e : edges)
        {
            if (e.first < pts2d.size() && e.second < pts2d.size())
            {
                line(debug,
                     pts2d[e.first],
                     pts2d[e.second],
                     Scalar(255,0,0), 2);
            }
        }

        imwrite("../results/corners_" + to_string(i) + ".png", debug);
    }
    // ===================================
    // DEPTH MAP GENERATION
    // ===================================

    int ref_idx = 0;
    Mat I0 = images[ref_idx];
    Matrix<double, 3, 4> P0 = projections[ref_idx];
    int h = I0.rows;
    int w = I0.cols;
    int window = 5;
    int half = window / 2;
    
    Mat depth_map = Mat::zeros(h, w, CV_64F);
    Mat best_score(h, w, CV_64F, Scalar(-1e9));
    double conf_threshold = 0.70;

    Matrix3d M_inv = (P0.block<3,3>(0,0)).inverse();
    Vector3d cam_center = -M_inv * P0.col(3);

    vector<vector<Vector3d>> rays(h, vector<Vector3d>(w));
    for(int y=0; y<h; ++y) {
        for(int x=0; x<w; ++x) {
            Vector3d p_h(x, y, 1.0);
            Vector3d r = M_inv * p_h;
            rays[y][x] = r.normalized();
        }
    }

    auto pts_ref_2d = project_points(P0, corners);
    float min_u = 1e9, max_u = -1e9;
    float min_v = 1e9, max_v = -1e9;
    for(auto& pt : pts_ref_2d) {
        if(pt.x < min_u) min_u = pt.x;
        if(pt.x > max_u) max_u = pt.x;
        if(pt.y < min_v) min_v = pt.y;
        if(pt.y > max_v) max_v = pt.y;
    }

    int u_min = max(half, (int)min_u);
    int u_max = min(w - half, (int)max_u);
    int v_min = max(half, (int)min_v);
    int v_max = min(h - half, (int)max_v);

    vector<double> depth_values;
    int num_depths = 100;
    for(int i=0; i<num_depths; ++i) {
        depth_values.push_back(0.2 + i * (0.6 - 0.2) / (double)(num_depths - 1));
    }

    cout << "Generating Depth Map for I0...\n";
    
    for(int y = v_min; y < v_max; ++y) {
        if((y - v_min) % 25 == 0) {
            printf("  Progress: %.1f%%\r", (double(y - v_min) / (v_max - v_min) * 100.0));
            fflush(stdout);
        }
        for(int x = u_min; x < u_max; ++x) {
            Vec3b intensity = I0.at<Vec3b>(y, x);
            if((intensity[0] + intensity[1] + intensity[2]) / 3.0 < 20.0) continue;

            Mat C0 = I0(Rect(x - half, y - half, window, window));
            
            vector<Vector3d> patch_rays;
            patch_rays.reserve(window * window);
            for(int dy = -half; dy <= half; ++dy) {
                for(int dx = -half; dx <= half; ++dx) {
                    patch_rays.push_back(rays[y + dy][x + dx]);
                }
            }

            for(double d : depth_values) {
                Vector3d X_center = cam_center + d * rays[y][x];
                if (!(X_center.array() >= (min_pt.array() - 0.005)).all() || 
                    !(X_center.array() <= (max_pt.array() + 0.005)).all()) {
                    continue;
                }

                vector<Vector3d> X_patch;
                X_patch.reserve(patch_rays.size());
                for(auto& r : patch_rays) {
                    X_patch.push_back(cam_center + d * r);
                }

                vector<double> scores;
                for(int k=0; k<images.size(); ++k) {
                    if(k == ref_idx) continue;
                    Mat C1;
                    if(ComputeConsistencyPatch(images[k], projections[k], X_patch, C1, window)) {
                        double score = NormalizedCrossCorrelation(C0, C1);
                        if(score > 0.4) {
                            scores.push_back(score);
                        }
                    }
                }

                if(scores.size() >= 2) {
                    double avg_s = 0;
                    for(double s : scores) avg_s += s;
                    avg_s /= scores.size();

                    if(avg_s > best_score.at<double>(y, x) && avg_s >= conf_threshold) {
                        best_score.at<double>(y, x) = avg_s;
                        depth_map.at<double>(y, x) = d;
                    }
                }
            }
        }
    }
    cout << "\nFinal Processing Complete.\n";

    save_point_cloud_obj("../results/temple_from_I0.obj", depth_map, I0, cam_center, rays, min_pt, max_pt);

    Mat depth_vis;
    normalize(depth_map, depth_vis, 0, 255, NORM_MINMAX, CV_8U);
    applyColorMap(depth_vis, depth_vis, COLORMAP_PLASMA);
    imwrite("../results/depth_map_I0_color.png", depth_vis);

    return 0;
}