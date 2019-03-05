#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h> 
#include <stdio.h>
#include <vector>
#include <chrono>
#include <math.h>


using namespace std;
using namespace cv;

Mat SgbmTest(Mat left_ud,Mat right_ud);             //使用SGBM算法进行立体重构，返回视差矩阵
Mat getDepth(Mat disp,Mat Q);                       //获取视差矩阵对应的像素的深度
Mat getXYZ(Mat img, Mat depth, Mat K, int offset = 128);  //获得图像在相机坐标系下的坐标
void joinMap(Mat color, Mat xyz, Mat R, Mat t);     //进行pcl点云拼接
double poseEstimation_3d3d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat xyz_1, Mat xyz_2, vector<DMatch> matches,Mat K, Mat* R, Mat* t);
double poseEstimation_3d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat xyz, vector<DMatch> matches, Mat K, Mat* R, Mat* t);
double poseEstimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches,Mat K, Mat xyz, Mat* R, Mat* t);
double calcReprojectError_2d2d(vector<Point2d> point_1, vector<Point2d> point_2,Mat K, Mat R, Mat t, Mat xyz);
double calcReprojectError_3d3d(vector<Point3d> points_1, vector<Point3d> points_2,Mat K, Mat R, Mat t);    
double calcReprojectError3d2d(vector<Point3d> points_1,vector<Point2d> points_2, Mat K, Mat R, Mat t);      
vector<DMatch> matchKeypoints(Mat reference, Mat current, vector<KeyPoint>* keypoints_1, vector<KeyPoint>* keypoints_2);
void calcRTfromHomo(Mat H,Mat K,Mat* R, Mat *t);


int main(int argc,char** argv){
    VideoCapture capture(1);                        //有时候是0有时候是1，奇了怪了
    Mat frame;                                      //相机获取的一帧
    Mat left,right;                                 //左右图像
    Mat left_ud,right_ud;                           //经过畸变矫正的左右图像
    Mat camera_left = Mat::eye(3,3,CV_64F);         //左侧相机内参数矩阵
    Mat camera_right = Mat::eye(3,3,CV_64F);        //右侧相机内参数矩阵
    Mat dist_coeffs_left = Mat::eye(5,1,CV_64F);    //左侧相机畸变参数向量
    Mat dist_coeffs_right = Mat::eye(5,1,CV_64F);   //右侧相机畸变参数向量
    Mat R = Mat::eye(3,1,CV_64F);
    Mat t = Mat::eye(3,1,CV_64F);
    Mat map1,map2,map3,map4,R1,R2,P1,P2,Q;
    Rect rio1,rio2;
    Size img_size;

    camera_left.at<double>(0, 0) = 406.86448; 
    camera_left.at<double>(0, 2) = 149.08732; 
    camera_left.at<double>(1, 1) = 407.91008; 
    camera_left.at<double>(1, 2) = 118.67917;

    camera_right.at<double>(0, 0) = 404.99717; 
    camera_right.at<double>(0, 2) = 152.64756; 
    camera_right.at<double>(1, 1) = 407.34931; 
    camera_right.at<double>(1, 2) = 121.40527;

    dist_coeffs_left.at<double>(0, 0) = -0.41906;
	dist_coeffs_left.at<double>(1, 0) = -0.05569;
	dist_coeffs_left.at<double>(2, 0) = -0.00108;
	dist_coeffs_left.at<double>(3, 0) = 0.00425;
    dist_coeffs_left.at<double>(4, 0) = 0;

    dist_coeffs_right.at<double>(0, 0) = -0.50873;
	dist_coeffs_right.at<double>(1, 0) = 0.51490;
	dist_coeffs_right.at<double>(2, 0) = 0.00193;
	dist_coeffs_right.at<double>(3, 0) = -0.01215;
    dist_coeffs_right.at<double>(4, 0) = 0;

    R.at<double>(0, 0) = 0.01800;
	R.at<double>(1, 0) = -0.01484;
	R.at<double>(2, 0) = 0.00276;

	t.at<double>(0, 0) = -172.89628;
	t.at<double>(1, 0) = 1.79758;
    t.at<double>(2, 0) = 4.28381;

    capture>>frame;
    int col = frame.cols;
    int row = frame.rows;
     
    left = frame(Rect(0, 0, col/2, row));       //分割为左右画面
    right = frame(Rect(col/2, 0, col/2, row));  
    img_size = left.size();
    
    Mat R_R;
    Rodrigues(R,R_R);   //使用罗德里格斯公式把R变为旋转矩阵			
    //输入双目的内参等，输出R1 R2 P1 P2 Q roi1 roi2 用于矫正畸变和计算深度	
	stereoRectify(camera_left, dist_coeffs_left, camera_right, dist_coeffs_right, img_size, R_R, t, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &rio1, &rio2);

	initUndistortRectifyMap(camera_left, dist_coeffs_left, R1, P1, img_size, CV_16SC2, map1, map2);	

    initUndistortRectifyMap(camera_right, dist_coeffs_right, R2, P2, img_size, CV_16SC2, map3, map4); 

    chrono::steady_clock::time_point start,end;
    chrono::duration<double> time_used;
    Mat disp, xyz, depth;           //disp是视差图，xyz是视差图在相机坐标系下的坐标，depth是视差图每一点的深度 
    Mat xyz_reference;              //上一帧的xyz坐标  
    Mat reference, current;
    vector<DMatch> matches;         //经过匹配和筛选的特征点
    vector<KeyPoint> keypoints_1,keypoints_2;   //关键点 1对应reference 2对应current
    Mat rotation_matrix = Mat::eye(3,3,CV_64F);
    Mat translation_matrix = Mat::eye(3,1,CV_64F);    //旋转矩阵和平移矩阵R，t
    Mat best_rotation,best_translation;     //最佳的旋转矩阵和平移矩阵
    double error, min_error;
    int key;
    int i = 0;
    while(true){
        capture>>frame;
        imshow("实时图像",frame);
        imshow("可以测量深度的区域",frame.colRange(128,320));
        key = waitKey(1);
        if(key == ' '){
            left = frame(Rect(0, 0, col/2, row));       //分割为左右画面
            right = frame(Rect(col/2, 0, col/2, row));
            remap(left,left_ud,map1,map2,INTER_LINEAR);
            remap(right,right_ud,map3,map4,INTER_LINEAR);

            // imshow("left_ud",left_ud);
            // imshow("right_ud",right_ud);
            imshow("reference",left.colRange(128,320));

            cvtColor(left_ud,left_ud,CV_BGR2GRAY);      //转换成灰度图
            cvtColor(right_ud,right_ud,CV_BGR2GRAY);

            disp = SgbmTest(left_ud,right_ud);     //使用SGBM方法进行深度分析,获得视差矩阵
            depth = getDepth(disp, Q);
            xyz = getXYZ(left.colRange(128,left.cols/2),depth,camera_left);

            if(i == 0 ){
                current = left.clone();     //用深拷贝，如果浅拷贝会导致current和reference一样
                xyz_reference = xyz.clone();
                joinMap(left.colRange(128,left.cols),xyz,Mat::eye(3,3,CV_64F),Mat::ones(3,1,CV_64F));
            }
            else{
                reference = current;
                current = left.clone();
                /* 2d2d之间通过对极几何求解位姿*/
                matches = matchKeypoints(reference,current,&keypoints_1,&keypoints_2);
                //keypoint顺序要注意，求解的R t是从current到reference的
                error = poseEstimation_2d2d(keypoints_1, keypoints_2, matches,camera_left, xyz, &rotation_matrix,&translation_matrix);
                
                best_rotation = rotation_matrix.clone();
                best_translation = translation_matrix.clone();
                min_error = error;

                
                matches = matchKeypoints(reference.colRange(128,reference.cols),current.colRange(128,current.cols),&keypoints_1,&keypoints_2);
                /*3d2d之间通过EPnP求解*/
                error = poseEstimation_3d2d(keypoints_1,keypoints_2,xyz,matches,camera_left,&rotation_matrix,&translation_matrix);
                if(error < min_error){
                    best_rotation = rotation_matrix.clone();
                    best_translation = translation_matrix.clone();
                    min_error = error;
                }
                /*3d3d之间通过SVD方法求解ICP问题*/
                // 1对应reference 2对应current
                error = poseEstimation_3d3d(keypoints_1, keypoints_2, xyz_reference, xyz , matches,camera_left, &rotation_matrix, &translation_matrix);
                if(error < min_error){
                    best_rotation = rotation_matrix;
                    best_translation = translation_matrix;
                    min_error = error;
                }
                cout<<"/*****最佳的*****/"<<endl;
                cout<<"旋转矩阵"<<endl<<best_rotation<<endl;
                cout<<"平移矩阵"<<endl<<best_translation<<endl;
                cout<<"重投影误差 "<<min_error<<endl;

                joinMap(left.colRange(128,left.cols),xyz,best_rotation,best_translation);
            }
            i++;
            xyz_reference = xyz.clone();
        }
        
    }
    

}



Mat SgbmTest(Mat left_ud,Mat right_ud){ 
    int SADWindowSize = 11;//必须是奇数
    Ptr<StereoSGBM> sgbm = StereoSGBM::create();
    sgbm->setBlockSize(SADWindowSize);
    sgbm->setP1(8 *1*SADWindowSize*SADWindowSize);
	sgbm->setP2(32 *1*SADWindowSize*SADWindowSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(128);//128//num_disp good
	sgbm->setUniquenessRatio(5);//good
	sgbm->setSpeckleWindowSize(100);//good
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setPreFilterCap(64);// good
    sgbm->setMode(StereoSGBM::MODE_HH);//good
    Mat disp,disp_8;
    sgbm->compute(left_ud,right_ud,disp);

    disp = disp.colRange(128,disp.cols);        //转换后左侧有一部分没有用，删去
    disp.convertTo(disp_8,CV_8U,255/(128*16.));

    imshow("disp",disp_8);
    return disp;

}

//输入:视差矩阵disp，重投影矩阵Q
//返回值：视差图的深度
Mat getDepth(Mat disp,Mat Q){
    Mat xyz;
    reprojectImageTo3D(disp,xyz,Q);     //生成视差图在像极坐标系下的xyz坐标
    xyz *= 1.6;
    vector<Mat> xyz_split; 
    Mat depth;
    split(xyz,xyz_split);  
    depth = xyz_split[2];  
    depth.convertTo(depth,CV_64F);      //转化为double精度 
    return depth;    
}

//输入：图片img，图片对应的深度图，相机内参,偏移量
//输出：图中每一点在相机坐标系下的坐标
//注意：偏移量是指深度图不能覆盖全部的图片，只能从第offset列开始才有深度
Mat getXYZ(Mat img, Mat depth, Mat K, int offset){
    Mat xyz(depth.rows,depth.cols,CV_64FC3);
    for(int u = 0; u < xyz.rows; u++){
        for(int v = 0; v < xyz.cols; v++){
            Mat p_uv = (Mat_<double>(3,1)<< u + offset, v, 1);      //像素坐标(齐次坐标)
            Mat p_c(3,1,CV_64F);
            // Z * p_uv = K * p_c
            double z = depth.at<double>(u,v);
            if(z > 160 || z < 40) 
                z = 0;                      //z代表深度，如果深度过大或过小就认为测量错误，把深度置零
            p_c = z * K.inv() * p_uv;
            xyz.at<Vec3d>(u,v)[0] = p_c.at<double>(0,0);
            xyz.at<Vec3d>(u,v)[1] = p_c.at<double>(1,0);
            xyz.at<Vec3d>(u,v)[2] = p_c.at<double>(2,0);
        }
    }
    // for(int u = 0; u < xyz.rows; u++)
    //     for(int v = 0; v < xyz.cols; v++){
    //         cout<<xyz.at<Vec3d>(u,v)[0]<<" "<<xyz.at<Vec3d>(u,v)[1]<<" "<<xyz.at<Vec3d>(u,v)[2]<<endl;
    //     }
    return xyz;
}

//输入：上一张图像reference，当前图像current
//输出：两张图片的关键点keypoint_1,keypoint_2，匹配的特征点matches
vector<DMatch> matchKeypoints(Mat reference, Mat current, vector<KeyPoint>* keypoints_1, vector<KeyPoint>* keypoints_2){
    
    Mat descriptors_1,descriptors_2;            //描述子
    //orb参数解释
    //https://docs.opencv.org/3.2.0/db/d95/classcv_1_1ORB.html#adc371099dc902a9674bd98936e79739c
    Ptr<ORB> orb = ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    //检测Oriented FAST关键点
    orb->detect(reference,*keypoints_1);
    orb->detect(current,*keypoints_2);
    //计算描述子
    orb->compute(reference,*keypoints_1,descriptors_1);
    orb->compute(current,*keypoints_2,descriptors_2);
    //对描述子匹配，使用汉明距离
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1,descriptors_2,matches);
    //对匹配好的描述子进行筛选，去除大多数误匹配
    double min_dist = 10000,max_dist = 0, dist;
    for(int i = 0; i<descriptors_1.rows; i++){    //找到描述子之间的最大和最小距离
        dist = matches[i].distance;
        if(dist > max_dist) max_dist = dist;
        if(dist < min_dist) min_dist = dist;
    }
    vector<DMatch> good_matches;
    for(int i = 0; i<descriptors_1.rows; i++){
        if(matches[i].distance <= max(2*min_dist,30.0))
            good_matches.push_back(matches[i]);
    }
    return good_matches;
}

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
//输入：彩色图color，坐标xyz，旋转矩阵R，平移矩阵t，相机内参K
//输出：无
void joinMap(Mat color, Mat xyz, Mat R, Mat t){
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();   //变换矩阵T
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();    //旋转矩阵
    Eigen::AngleAxisd rotation_vector;  //旋转向量
    
    cv2eigen(R, rotation_matrix);       //opencv的Mat转化为Eigen的Matrix
    rotation_vector.fromRotationMatrix(rotation_matrix);    //旋转矩阵生成旋转向量
    
    T.rotate(rotation_vector);  
    T.pretranslate(Eigen::Vector3d(t.at<double>(Point(0,0)), t.at<double>(Point(1,0)), t.at<double>(Point(2,0))));
    static Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();    //这个才是真正的坐标转换需要的变换矩阵
    T_cw = T_cw * T;

    //新建一个点云
    static PointCloud::Ptr point_cloud(new PointCloud);
    for(int v = 0; v < color.rows; v++)
        for(int u = 0; u < color.cols; u++){
            
            //测得的深度小于400mm大于1500mm就不准了  不同的摄像头这个值不同，自己调整
            if((xyz.at<Vec3d>(v,u)[2] < 40)||(xyz.at<Vec3d>(v,u)[2] > 150))   
                continue;                   
            //把每一点坐标转换为vector3d类型，方便后面计算
            Eigen::Vector3d point_camera;
            point_camera[0] = xyz.at<Vec3d>(v,u)[0];
            point_camera[1] = xyz.at<Vec3d>(v,u)[1];
            point_camera[2] = xyz.at<Vec3d>(v,u)[2];

            //cout<<"x "<<point_camera[0]<<" y "<<point_camera[1]<<" z "<<point_camera[2]<<endl;
            Eigen::Vector3d point_world = T_cw * point_camera;

            PointT p;
            p.x = point_world[0];
            p.y = point_world[1];
            p.z = point_world[2];
           
            // cout<<"p.z "<<p.z<<"  point_world "<<point_world[2]<<endl;
            p.b = color.at<Vec3b>(v,u)[0];
            p.g = color.at<Vec3b>(v,u)[1];
            p.r = color.at<Vec3b>(v,u)[2];
            point_cloud->points.push_back(p);
        }

    for(int i = 0; i < 100; i++){
        PointT p;
        p.x = i;
        p.y = 0;
        p.z = 0;
        p.r = 255;
        p.g = 0;
        p.b = 0;
        point_cloud->points.push_back(p);
    }
    for(int i = 0; i < 100; i++){
        PointT p;
        p.x = 0;
        p.y = i;
        p.z = 0;
        p.r = 0;
        p.g = 255;
        p.b = 0;
        point_cloud->points.push_back(p);
    }
    for(int i = 0; i < 100; i++){
        PointT p;
        p.x = 0;
        p.y = 0;
        p.z = i;
        p.r = 0;
        p.g = 0;
        p.b = 255;
        point_cloud->points.push_back(p);
    }

    cout<<"共有"<<point_cloud->size()<<"个点"<<endl;
    
   
    point_cloud->is_dense = false;
    cout<<"存储点云中........"<<endl;
    pcl::io::savePCDFileBinary("map.pcd",*point_cloud);
    cout<<"存储完成"<<endl;
    
    cout<<endl<<endl<<endl<<endl<<endl;
}

//输入：匹配的关键点
//输出：旋转矩阵和平移矩阵
//注意：R和t是从keypoint2到keypoint1的，也就是 P1 = R×P2 + t
double poseEstimation_3d3d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat xyz_1, Mat xyz_2, vector<DMatch> matches,Mat K, Mat* R, Mat* t){
    vector<Point3d> pts_1,pts_2;    //对应的关键点转化成point3f形式
    for(DMatch m:matches){
        int x1 = keypoints_1[m.queryIdx].pt.x;
        int y1 = keypoints_1[m.queryIdx].pt.y;
        int x2 = keypoints_2[m.trainIdx].pt.x;
        int y2 = keypoints_2[m.trainIdx].pt.y;

        double z1 = xyz_1.at<Vec3d>(x1,y1)[2];
        double z2 = xyz_2.at<Vec3d>(x2,y2)[2];

        if(xyz_1.at<Vec3d>(x1,y1)[2]<40 || xyz_1.at<Vec3d>(x1,y1)[2] > 150) continue;
        if(xyz_2.at<Vec3d>(x2,y2)[2]<40 || xyz_2.at<Vec3d>(x2,y2)[2] > 150) continue;
        
        pts_1.push_back(Point3d(xyz_1.at<Vec3d>(x1,y1)[0],xyz_1.at<Vec3d>(x1,y1)[1],xyz_1.at<Vec3d>(x1,y1)[2]));
        pts_2.push_back(Point3d(xyz_2.at<Vec3d>(x2,y2)[0],xyz_2.at<Vec3d>(x2,y2)[1],xyz_2.at<Vec3d>(x2,y2)[2]));
    }
    cout<<"匹配点的数目： "<<pts_1.size()<<"  "<<pts_2.size()<<endl;
    Point3d p1,p2;	//两张图片的质心
    int N = pts_1.size();
    for(int i=0;i<N;i++)
    {
        p1 += pts_1[i];
        p2 += pts_2[i];
    }
    p1 /= N;
    p2 /= N;
    // cout<<"p1 "<<p1.x<<" "<<p1.y<<" "<<p1.z<<endl;
    // cout<<"p2 "<<p2.x<<" "<<p2.y<<" "<<p2.z<<endl;

    vector<Point3d> q1(N),q2(N);	//去质心坐标
    for(int i=0;i<N;i++)
    {
        q1[i] = pts_1[i] - p1;
        q2[i] = pts_2[i] - p2;
    }
    //计算 q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i=0;i<N;i++)
    {
	    W += Eigen::Vector3d(q1[i].x , q1[i].y , q1[i].z)*Eigen::Vector3d(q2[i].x , q2[i].y , q2[i].z).transpose();
    }
    // cout<<"W = "<<endl<<W<<endl;
    
    //使用SVD方法
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    // cout<<"U = "<<endl<<U<<endl;
    // cout<<"V = "<<endl<<V<<endl;
    
    Eigen::Matrix3d R_ = U*(V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x,p1.y,p1.z) - R_*Eigen::Vector3d(p2.x, p2.y, p2.z);
    eigen2cv(R_,*R);
    t->at<double>(0,0) = t_(0,0);
    t->at<double>(1,0) = t_(1,0);
    t->at<double>(2,0) = t_(2,0);

    cout<<"通过3d3d得到"<<endl;
    cout<<"旋转矩阵"<<endl<<*R<<endl;
    cout<<"平移矩阵"<<endl<<*t<<endl;

    double error = calcReprojectError_3d3d(pts_2, pts_1, K, *R, *t);//注意point的顺序
    cout<<"重投影误差  "<<error<<endl;
    Eigen::AngleAxisd rotation_vector(R_);
    return error;
    
    
}

//计算重投影误差 3d3d
//输入：对应的特征点，相机内参K,旋转矩阵R，平移矩阵t
//输出：重投影误差
//注意：R，t是从1到2的，也就是 P2 = R×P1 + t
double calcReprojectError_3d3d(vector<Point3d> points_1, vector<Point3d> points_2,Mat K, Mat R, Mat t){
    double x_error, y_error, error = 0;
    int n = 0;
    Mat p1 = Mat::eye(3,1,CV_64F);
    Mat p2 = Mat::eye(3,1,CV_64F);

    for(int i = 0; i < points_1.size(); i++){
        if(points_1[i].x < 1) continue;
        //point1和point2存的是两张图关键点的三维坐标
        //从point3d形式转化为mat
        p1.at<double>(0,0) = points_1[i].x;
        p1.at<double>(1,0) = points_1[i].y;
        p1.at<double>(2,0) = points_1[i].z;
        //p2是第二张图的关键点的uv坐标
        p2.at<double>(0,0) = points_2[i].x;
        p2.at<double>(1,0) = points_2[i].y;
        p2.at<double>(2,0) = points_2[i].z;

        p2 = K*p2;
        p2 /= p2.at<double>(2,0);


        Mat tmp1 = (R*p1 + t);
        Mat tmp2 = K * tmp1;            
        tmp2.at<double>(0,0) /= tmp1.at<double>(2,0); //矩阵变为其次坐标，x y转化为像素坐标系u v
        tmp2.at<double>(1,0) /= tmp1.at<double>(2,0);   //tmp2就是 K*(R*p1 + t)/Z
        Mat tmp = p2 - tmp2;
        x_error = tmp.at<double>(0,0);
        y_error = tmp.at<double>(1,0);
        error += sqrt(x_error*x_error + y_error*y_error);
        n++;
    }
    return error/n;
}

//输入：匹配的关键点，相机内参矩阵K
//输出：旋转矩阵R，平移矩阵t,最小的重投影误差
//注意：R和t是从keypoint2到keypoint1的，也就是 P1 = R×P2 + t
double poseEstimation_2d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches,Mat K, Mat xyz, Mat* R, Mat* t){
    //把匹配点转换成vector<Point2f>形式
    vector<Point2d> points_1;
    vector<Point2d> points_2;
    Mat rotation_matrix = Mat::eye(3,3,CV_64F);
    Mat translation_matrix = Mat::eye(3,1,CV_64F);
    double error, min_error;
    for(int i = 0; i < (int)matches.size(); i++){
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_1[matches[i].trainIdx].pt);
    }
    
    Mat H;
    H = findHomography(points_1,points_2,RANSAC,3,noArray(),2000,0.99);
    calcRTfromHomo(H, K, R, t);
    error = calcReprojectError_2d2d(points_1, points_2, K, *R, *t,xyz);
    rotation_matrix = R->clone();
    translation_matrix = t->clone();
    min_error = error;
    cout<<"自己计算的H"<<endl;
    cout<<"旋转矩阵"<<endl<<*R<<endl;
    cout<<"平移矩阵"<<endl<<*t<<endl;
    cout<<"重投影误差  "<<error<<endl<<endl;

    vector<Mat> rotation,translation,n;
    decomposeHomographyMat(H,K,rotation,translation,n);
    cout<<"通过H计算得到"<<endl;
    for(int i = 0; i < rotation.size(); i++){
        for (int j = 0; j <translation.size(); j++){
            error = calcReprojectError_2d2d(points_1, points_2, K, rotation[i], translation[j],xyz);
            cout<<"旋转矩阵"<<endl<<rotation[i]<<endl;
            cout<<"平移矩阵"<<endl<<translation[j]<<endl;
            cout<<"重投影误差  "<<error<<endl<<endl;
            if(error < min_error){
                rotation_matrix = rotation[i];
                translation_matrix = translation[j];
                min_error = error;
            }
        }
    }

    //计算本质矩阵E
    Mat E;
    E = findEssentialMat(points_1, points_2, K, RANSAC);
    //从本质矩阵计算R t
    recoverPose(E,points_1,points_2, K, *R, *t);
    error = calcReprojectError_2d2d(points_1, points_2, K, *R, *t,xyz);
    cout<<"通过E计算得到"<<endl;
    cout<<"旋转矩阵"<<endl<<R->inv()<<endl;
    cout<<"平移矩阵"<<endl<<*t<<endl;
    cout<<"重投影误差  "<<error<<endl<<endl;
    if(error < min_error){
        rotation_matrix = *R;
        translation_matrix = *t;
        min_error = error;
    }
    //要取反，因为需要的R t是从keypoint2到keypoint1的
    *R = rotation_matrix.inv();
    *t = -translation_matrix;
    return min_error;
}

//计算重投影误差
//输入：对应的特征点，相机内参K,旋转矩阵R，平移矩阵t,特征点1对应的xyz坐标 
//输出：重投影误差
//注意：R，t是从1到2的，也就是 P2 = R×P1 + t
double calcReprojectError_2d2d(vector<Point2d> point_1, vector<Point2d> point_2,Mat K, Mat R, Mat t, Mat xyz){
    double x_error,y_error,error=0;
    int n = 0;
    Mat p1 = Mat::eye(3,1,CV_64F);
    Mat p2 = Mat::eye(3,1,CV_64F);

    for(int i = 0; i < (int)point_1.size(); i++){
        if(point_1[i].x < 128) continue;        //位于彩色图前128列的点是没有对应的xyz坐标的
        //p1是关键点1对应的xyz坐标
        p1.at<double>(0,0) = xyz.at<Vec3d>(point_1[i].x - 128,  point_1[i].y)[0];      
        p1.at<double>(1,0) = xyz.at<Vec3d>(point_1[i].x - 128,  point_1[i].y)[1];
        p1.at<double>(2,0) = xyz.at<Vec3d>(point_1[i].x - 128,  point_1[i].y)[2];
        if(p1.at<double>(0,0) == 0) 
            continue;       
        //p2是关键点2对应的uv坐标
        p2.at<double>(0,0) = point_2[i].x; 
        p2.at<double>(1,0) = point_2[i].y;
        p2.at<double>(2,0) = 1;
        //计算重投影误差 Z*p2 = K*(R*p1 + t)
        Mat tmp1 = (R*p1 + t);
        Mat tmp2 = K * tmp1;            
        tmp2.at<double>(0,0) /= tmp1.at<double>(2,0); //矩阵变为其次坐标，x y转化为像素坐标系u v
        tmp2.at<double>(1,0) /= tmp1.at<double>(2,0);   //tmp2就是 K*(R*p1 + t)/Z
        Mat tmp = p2 - tmp2;
        x_error = tmp.at<double>(0,0);
        y_error = tmp.at<double>(1,0);
        error += sqrt(x_error*x_error + y_error*y_error);
        n++;
    }
    return error/n;
}

//求解单应矩阵H
//输入：单应矩阵H，相机内参K
// 输出：旋转矩阵和平移矩阵 
void calcRTfromHomo(Mat H,Mat K,Mat* R, Mat *t){	
    Mat matrix_ones = Mat::ones(3,1,CV_64F);	
	//for SVD
	Mat U = Mat(3, 3, CV_64F);
	Mat W = Mat(3, 3, CV_64F);
	Mat V = Mat(3, 3, CV_64F);
	Mat invK = Mat(3, 3, CV_64F);
	// three columns of Homography matrix
	Mat h1 = Mat(3, 1, CV_64F);
	Mat h2 = Mat(3, 1, CV_64F);
	Mat h3 = Mat(3, 1, CV_64F);
	// three columns of rotation matrix
	Mat r1 = Mat(3, 1, CV_64F);
	Mat r2 = Mat(3, 1, CV_64F);
	Mat r3 = Mat(3, 1, CV_64F);
	// translation vector
    h1 = H.colRange(0,1);
    h2 = H.colRange(1,2);
    h3 = H.colRange(2,3);
    r1 = R->colRange(0,1);
    r2 = R->colRange(1,2);
	r3 = R->colRange(2,3);

    invert(K, invK);
    r1 = (invK) * (h1);
    r2 = (invK) * (h2);
    *t = (invK) * (h3);
	
    normalize(r1,r1);
    multiply(r2,matrix_ones,r2,1/norm(r1));
    multiply(*t,matrix_ones,*t,1/norm(r1));
    r3 = r1.cross(r2);

    SVD::compute(*R, W, U, V, CV_SVD_V_T);
    *R = (U) * (V);
}
//使用EPnP求解相机位姿
//输入：匹配的关键点，相机内参矩阵K
//输出：旋转矩阵R，平移矩阵t,最小的重投影误差
//
double poseEstimation_3d2d(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat xyz, vector<DMatch> matches, Mat K, Mat* R, Mat* t){
    vector<Point3d> points_1;
    vector<Point2d> points_2;
    for(int i = 0; i < matches.size(); i++){
        int x = keypoints_1[matches[i].queryIdx].pt.x;
        int y = keypoints_1[matches[i].queryIdx].pt.y;
        if(xyz.at<Vec3d>(x,y)[2] < 40 || xyz.at<Vec3d>(x,y)[2] > 150)
            continue;
        points_1.push_back(Point3d(xyz.at<Vec3d>(x,y)[0],xyz.at<Vec3d>(x,y)[1],xyz.at<Vec3d>(x,y)[2]));
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    Mat r;
    solvePnP(points_1,points_2,K,Mat(),r,*t,false,cv::SOLVEPNP_EPNP);
    cv::Rodrigues(r,*R);
    float error = calcReprojectError3d2d(points_1,points_2,K,*R,*t);
    cout<<"通过3d2d得到"<<endl;
    cout<<"匹配点数目 "<<points_1.size()<<endl;
    cout<<"旋转矩阵"<<endl<<*R<<endl;
    cout<<"平移矩阵"<<endl<<*t<<endl;
    cout<<"重投影误差 "<<error<<endl<<endl;
    return error;
}
//计算重投影误差 3d2d
//输入：对应的特征点，相机内参K,旋转矩阵R，平移矩阵t
//输出：重投影误差
//注意：R，t是从1到2的，也就是 P2 = R×P1 + t
double calcReprojectError3d2d(vector<Point3d> points_1,vector<Point2d> points_2, Mat K, Mat R, Mat t){
    double error = 0, x_error, y_error;
    int n = 0;
    Mat p2 = Mat::eye(3,1,CV_64F);
    Mat p1 = Mat::eye(3,1,CV_64F);
    for(int i = 0; i < points_1.size(); i++){
        if(points_1[i].x < 1) continue;
        //point1和point2存的是两张图关键点的三维坐标
        //从point3d形式转化为mat
        p1.at<double>(0,0) = points_1[i].x;
        p1.at<double>(1,0) = points_1[i].y;
        p1.at<double>(2,0) = points_1[i].z;
        //p2是第二张图的关键点的uv坐标
        p2.at<double>(0,0) = points_2[i].x;
        p2.at<double>(1,0) = points_2[i].y;
        p2.at<double>(2,0) = 1;

        Mat tmp1 = (R*p1 + t);
        Mat tmp2 = K * tmp1;            
        tmp2.at<double>(0,0) /= tmp1.at<double>(2,0); //矩阵变为其次坐标，x y转化为像素坐标系u v
        tmp2.at<double>(1,0) /= tmp1.at<double>(2,0);   //tmp2就是 K*(R*p1 + t)/Z
        Mat tmp = p2 - tmp2;
        x_error = tmp.at<double>(0,0);
        y_error = tmp.at<double>(1,0);
        error += sqrt(x_error*x_error + y_error*y_error);
        n++;
    }
    return error/n;
}