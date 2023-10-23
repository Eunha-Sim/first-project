#include <vector> 
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Bool.h>
#include <tf2/utils.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <array>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <Eigen/Core>

#include <casadi/casadi.hpp>


using namespace std;  
using namespace Eigen;
using namespace casadi;

// Helpers
#define iteration 100000


#define KMpH2MpS 0.277777  
#define Mps2KMpH 3.6
#define SIGN(x) ((x >= 0) ? 1 : -1)
#define DISTANCE(x1, y1, x2, y2) sqrt((x1 - x2)*(x1 - x2) +(y1 - y2)*(y1 - y2)) 
#define _RAD2DEG 180 / M_PI
#define _DEG2RAD M_PI / 180

// Tracker Module // 1: Pure-pursuit, 2: Stanley, 3: Kanayama  4:MPC
#define LOCAL_TRACKER 4

// Vehicle Parameters
#define TURN_RADIUS 6.0 
#define CURVATURE_MAX 1.0/TURN_RADIUS 
#define WHEEL_BASE  2.8325 
#define MIN_VEL_INPUT 1.0
#define MAX_VEL_INPUT 6.0
#define LimitSteering 540 
#define SteerRatio 15.43 
#define LimitDegPerSec 360.0
#define LimitVelPerSec 5.0 

ros::Subscriber Sub_localization, Sub_refPath; 
ros::Publisher Pub_MarkerCar, Pub_poseVehicle, Pub_ControlCmd, Pub_vehicleTraj, Pub_finish, Pub_Point, Pub_mpcTraj; 
visualization_msgs::Marker m_CarPos, m_CarPosLine; 
visualization_msgs::Marker m_point_marker;
bool m_pathFlag{false};
bool m_finishFlag{false};

struct CARPOSE {   
    double x, y, th, vel; 
};                       
CARPOSE m_car;  
double m_dir_mode = 1.0;  
double m_Steer_cmd = 0.0;
double m_Velocity_cmd = 0.0;
ros::Time ros_time; 
ros::Time ros_time2;
ros::Time start_ros_time;

bool is_init_path = true; 
vector<Vector3d> m_ref_path; 

int m_carIdx = 0; 
int m_remainIdx = 0; 
double m_curvature_max = 1.0 / TURN_RADIUS; 

Vector3d m_init_pose = Vector3d(0.0, 0.0, 0.0); 
Vector3d m_goal_pose = Vector3d(0.0, 0.0, 0.0);
bool almost_done_flag = false; 
bool m_finish_flag = false;

void Local2Global(double Lx, double Ly, double &Gx, double &Gy) { 
    double tmpX = Lx;
    double tmpY = Ly;
    Gx = m_car.x + (tmpX * cos(m_car.th) - tmpY * sin(m_car.th));
    Gy = m_car.y + (tmpX * sin(m_car.th) + tmpY * cos(m_car.th));
}

SX rhs(SX states, SX controls)
{
    SXVector rhs = {controls(0)*cos(states(2)),
                    controls(0)*sin(states(2)),
                    controls(0)*tan(states(3))/WHEEL_BASE,
                    controls(1)};
    return vertcat(rhs);
}
double dt = 0.01;

std::ofstream outputFile;   // 여기다 둬도 되나???

//% target_{x,y}: position with respect to the global frame
//% coord_{x,y,th}: pose with respect to the global frame
//% rel_{x,y}: relative position of target with respect to coord_{x,y,th}
void GetRelativePosition(double target_x, double target_y, double target_th, 
                         double coord_x, double coord_y, double coord_th, 
                         double &rel_x, double &rel_y, double &rel_th) {
    double rel_position_x = target_x - coord_x;
    double rel_position_y = target_y - coord_y;
    double D = sqrt(pow(rel_position_x, 2) + pow(rel_position_y, 2));
    double alpha = atan2(rel_position_y, rel_position_x);
    rel_x = D * cos(alpha - coord_th);
    rel_y = D * sin(alpha - coord_th);
    rel_th = atan2(sin(target_th - coord_th), cos(target_th - coord_th));
}

double three_pt_curvature(double x1, double y1, double x2, double y2, double x3, double y3) {  
    double fAreaOfTriangle = fabs((x1 * (y2 - y3) +\
                                    x2 * (y3 - y1) +\
                                    x3 * (y1 - y2)) / 2);
    double fDist12 = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    double fDist23 = sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3));
    double fDist13 = sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));
    double fKappa = 4 * fAreaOfTriangle / (fDist12 * fDist23 * fDist13); 

    //cross product
    double first_vec_x = x2 - x1;
    double first_vec_y = y2 - y1;
    double second_vec_x = x3 - x2;
    double second_vec_y = y3 - y2;
    double cross_product = first_vec_x * second_vec_y - second_vec_x * first_vec_y;
    int sign_ = (cross_product>=0) ? 1 : -1;

    if (isnan(fKappa) != 0)
        fKappa = 0.0;

    return sign_ * fKappa; 
}

double CalculateCurvature(int idx) {   
    int path_size = m_ref_path.size(); 
    if (idx != 0 && idx <= path_size-2)
        return three_pt_curvature(m_ref_path[idx-1][0], m_ref_path[idx-1][1],  
                                m_ref_path[idx][0], m_ref_path[idx][1],
                                m_ref_path[idx+1][0], m_ref_path[idx+1][1]);
    else
        return m_curvature_max;  
}

double m_dist_tol = 2.0;
double parkingVeloController(double velo_input) {  //parking controller?  
    if (velo_input == 0.0) {
        return 0.0;
    }
    else {
        double ClosestPtX = m_ref_path[m_carIdx][0];  
        double ClosestPtY = m_ref_path[m_carIdx][1];  
        double dist = 0.0;
        dist = DISTANCE(ClosestPtX, ClosestPtY, m_ref_path.back()[0], m_ref_path.back()[1]);
        if (dist < m_dist_tol) //[m]
            return (dist/m_dist_tol) * velo_input;//
            // return MIN_VEL_INPUT;//
        else
            return velo_input;
    }
}


void PublishCarPose(double car_x, double car_y, double car_th) {  
    // Car Red (Car)
    m_CarPos.header.stamp = ros::Time::now();
    m_CarPos.pose.position.x = car_x; 
    m_CarPos.pose.position.y = car_y; 

    // Create the vehicle's AXIS
    geometry_msgs::PoseStamped poseStamped;  
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "map";
    poseStamped.pose.position = m_CarPos.pose.position;
    poseStamped.pose.position.z = 5.5;
    poseStamped.pose.orientation = tf::createQuaternionMsgFromYaw(car_th);
    poseStamped.header = header;
    if (Pub_poseVehicle.getNumSubscribers() > 0)
        Pub_poseVehicle.publish(poseStamped);
}


void PublishPoint(double x, double y) {
    m_point_marker.pose.position.x = x;
    m_point_marker.pose.position.y = y;
    m_point_marker.pose.position.z = 2.0;
    Pub_Point.publish(m_point_marker);
}


/* Pure-pursuit algorithm
** Input: Look-ahead distance, Velocity 
** Output: steering angle and velocity for tracking the path
** Variables: (All information is with respect to the global frame (Global Axes of RVIZ))
LookAheadPtX, LookAheadPtY: LookAheadPoint with respect to the global frame. These are local variables.
m_ref_path: A global path which the vehicle should track. Its structure is [(x0,y0,index(=0)), (x1,y1,1), (x2,y2,2), ..., (x_{goal},y_{goal},N)] (global variables).
m_carIdx: The index of the closest waypoint on the global path (m_ref_path) from the vehicle (global variables).
m_car.x, m_car.y, m_car.th: The pose (position [m] + orientation [rad]) of the vehicle's center of the rear axle (global variables).
*/
pair<double, double> PurePursuit(double look_ahead_dist, double constant_velo_) { 
    double LookAheadPtX, LookAheadPtY;
    double steerAngle{0.0}; // [deg]
    double velo_result{0.0}; // [km/h]
    
    /* ========================================
    TODO: Code the pure-pursuit steering controller
    =========================================== */
    double pp_alpha, pp_R;
    int LookAheadIdx = 0;
    double pp_dist = 0;
    double gx,gy,gth;
    //find look ahead point 
    for(int i = m_carIdx; i < m_ref_path.size() - 1; i++) {
        pp_dist = DISTANCE(m_ref_path[i][0], m_ref_path[i][1], m_car.x, m_car.y);
        if(pp_dist > look_ahead_dist) {
            LookAheadIdx = i;
            break;
        }
    }
    LookAheadPtX = m_ref_path[LookAheadIdx][0];
    LookAheadPtY = m_ref_path[LookAheadIdx][1];

    //calc steerangle
    GetRelativePosition(LookAheadPtX, LookAheadPtY, m_car.th, 
                        m_car.x, m_car.y, m_car.th, 
                        gx, gy, gth);

    pp_alpha = atan2(gy, gx);
    pp_R = look_ahead_dist / (2 * sin(pp_alpha) );
    // steerAngle = - atan2(WHEEL_BASE, pp_R) * _RAD2DEG * SteerRatio ; //=>R became -inf~~ //
    steerAngle = - atan2(2*WHEEL_BASE*sin(pp_alpha), look_ahead_dist) * _RAD2DEG * SteerRatio ;

    if(abs(steerAngle) > LimitSteering)
        steerAngle = SIGN(steerAngle) * LimitSteering;
    if(abs(steerAngle) > LimitSteering)
        steerAngle = SIGN(steerAngle) * LimitSteering;
    velo_result = constant_velo_;
    velo_result = max(velo_result, MIN_VEL_INPUT);
    velo_result = min(velo_result, MAX_VEL_INPUT);
    PublishCarPose(m_car.x, m_car.y, m_car.th);
    PublishPoint(LookAheadPtX, LookAheadPtY);

    outputFile << LookAheadPtX << " " << LookAheadPtY << " " << m_car.th << " " << m_car.x << " " <<  m_car.y <<" " <<  m_car.th << std::endl; // 파일에 데이터 입력

    return std::make_pair(steerAngle, velo_result);
}




/* Stanley algorithm
** Input: Gain, Velocity, Cross-track err.
** Output: steering angle and velocity for tracking the path
** Variables: (All information is with respect to the global frame (Global Axes of RVIZ))
ClosestFrontPtX, ClosestFrontPtY: The closest point from the vehicle's center of the front wheel w.r.t the global frame. These are local variables.
m_ref_path: A global path which the vehicle should track. Its structure is [(x0,y0,index(=0)), (x1,y1,1), (x2,y2,2), ..., (x_{goal},y_{goal},N)] (global variables).
m_carIdx: The index of the closest waypoint on the global path (m_ref_path) from the vehicle (global variables).
m_car.x, m_car.y, m_car.th: The pose (position [m] + orientation [rad]) of the vehicle's center of the rear axle (global variables).
PublishPoint function: Visualize the closest point on Rviz.
*/
pair<double, double> Stanley(double gainK, double constant_velo_, double cross_track_err) {
    double ClosestFrontPtX, ClosestFrontPtY;
    double steerAngle{0.0}; // [deg]
    double velo_result{0.0}; // [km/h]    

    /* ========================================
    TODO: Code the stanley steering controller
    =========================================== */ 
 
    //m_ref_path[m_carIdx] : closeset point from front axle
    //m_car : center of rear axle

    double front_x, front_y;
    double th_e, ref_th;
    int sign_var;

    //calc theta error
    front_x = m_car.x + WHEEL_BASE*cos(m_car.th);
    front_y = m_car.y + WHEEL_BASE*sin(m_car.th);
    //theta of reference path
    ref_th = atan2(m_ref_path[m_carIdx+1][1]-m_ref_path[m_carIdx][1], m_ref_path[m_carIdx+1][0]-m_ref_path[m_carIdx][0]); //2)
    // ref_th = atan2(m_ref_path[m_carIdx][1]-m_car.y, m_ref_path[m_carIdx][0]-m_car.x); //1)
    th_e = atan2(sin(ref_th - m_car.th), cos(ref_th - m_car.th));

    double cx, cy, cth;
    GetRelativePosition(m_ref_path[m_carIdx][0], m_ref_path[m_carIdx][1], ref_th, 
                        front_x, front_y, m_car.th, 
                        cx, cy, cth);
    //signbit: true when negative      
    if(signbit(cy)){
        sign_var = -1;
    }
    else{
        sign_var = +1;
    }
    
    // steerAngle = ( -th_e  - atan2(gainK * sign_var * cross_track_err, constant_velo_ ) )* _RAD2DEG * SteerRatio;  //1)
    // steerAngle = ( -th_e  - atan2(gainK * sign_var * cross_track_err, constant_velo_ ) )* _RAD2DEG * SteerRatio; //2)


    // steerAngle = ( -th_e -atan2(gainK * sign_var *cross_track_err, constant_velo_ ) )* _RAD2DEG * SteerRatio; // KMpH2MpS
    steerAngle = ( -th_e -atan2(gainK * sign_var *cross_track_err, constant_velo_ *  KMpH2MpS) )* _RAD2DEG * SteerRatio; 


    //publish closeset point
    ClosestFrontPtX = m_ref_path[m_carIdx][0];
    ClosestFrontPtY = m_ref_path[m_carIdx][1];

    if(abs(steerAngle) > LimitSteering)
        steerAngle = SIGN(steerAngle) * LimitSteering;
    if(abs(steerAngle) > LimitSteering)
        steerAngle = SIGN(steerAngle) * LimitSteering;
    velo_result = constant_velo_;
    velo_result = max(velo_result, MIN_VEL_INPUT);
    velo_result = min(velo_result, MAX_VEL_INPUT);
    PublishCarPose(m_car.x + WHEEL_BASE*cos(m_car.th), m_car.y + WHEEL_BASE*sin(m_car.th), m_car.th);
    PublishPoint(ClosestFrontPtX, ClosestFrontPtY);

    outputFile << ClosestFrontPtX << " " << ClosestFrontPtY << " " << ref_th << " " << m_car.x + WHEEL_BASE*cos(m_car.th) << " " <<  m_car.y + WHEEL_BASE*sin(m_car.th) <<" " <<  m_car.th << std::endl; // 파일에 데이터 입력 // 이게 맞는건지 잘 모르겠다.


    return std::make_pair(steerAngle, velo_result);
}

/* Kanayama algorithm
** Input: Lateral gain (K_y), Orientation gain(K_th), RefVelocity
** Output: steering angle and velocity for tracking the path
** Variables: (All information is with respect to the global frame (Global Axes of RVIZ))
ClosestPtX, ClosestPtY: The closest point from the vehicle's center of the rear wheel w.r.t the global frame. These are local variables.
RefPtX, RefPtY: The reference point in the Kanayama controller paper w.r.t the global frame. These are local variables. 
                In this example, the reference point is determined as a fixed point 1 m away along the path from the closest point.
FF_term: Kanayama feed-forward term (angular velocity).
FF_gain: Gain for feed-forward term. As the reference path's curvature has some problems, so you need to adjust the FF_gain first before adjusting the feed-forward gain.
         In this step (the problem #1 for homework 3), set the feed-back gain (K_y, K_th) to zeros before adjusting the FF_gain.
FB_term: Kanayama feed-back term (angular velocity).
m_ref_path: A global path which the vehicle should track. Its structure is [(x0,y0,index(=0)), (x1,y1,1), (x2,y2,2), ..., (x_{goal},y_{goal},N)] (global variables).
m_carIdx: The index of the closest waypoint on the global path (m_ref_path) from the vehicle (global variables).
m_car.x, m_car.y, m_car.th: The pose (position [m] + orientation [rad]) of the vehicle's center of the rear axle (global variables).
PublishPoint function: Visualize the closest point on Rviz.
*/
pair<double, double> Kanayama(double K_y, double K_th, double ref_velo) {
    double ClosestPtX, ClosestPtY;
    double RefPtX, RefPtY;
    double steerAngle{0.0}; // [deg]
    double velo_result{0.0}; // [km/h]  
    double FF_gain = 0.0; // Adjust this gain for setting the feed-forward term (related to the problem #1).
    double x_err, y_err, heading_err;
    
    double ClosestTh;
    double RefTh;

    double refIdx, tmp_dist;
    double curvature;

    /* ======================================== 
    TODO: Code the Kanayama steering controller */ 
    double FF_term;
    double FB_term;
    /* YOU JUST NEED TO COMPUTE THE FF_term and FB_term above.
    =========================================== */ 

    ClosestPtX = m_ref_path[m_carIdx][0];
    ClosestPtY = m_ref_path[m_carIdx][1];

    //theta of reference path
    ClosestTh = atan2(m_ref_path[m_carIdx+1][1]-m_ref_path[m_carIdx][1], m_ref_path[m_carIdx+1][0]-m_ref_path[m_carIdx][0]);

    //find ref point 
    for(int i = m_carIdx; i < m_ref_path.size() - 1; i++) {
        tmp_dist = DISTANCE(m_ref_path[i][0], m_ref_path[i][1], m_car.x, m_car.y);
        if(tmp_dist > 1.0) { // lookahead distance가 1m라는거 아닌가? 그럼 항상 거리가 1m가 차이나야한다는 소리잖아. 
            refIdx = i;
            break;
        }
    }

    RefPtX = m_ref_path[refIdx][0];
    RefPtY = m_ref_path[refIdx][1];
    RefTh = atan2(m_ref_path[refIdx+1][1]-m_ref_path[refIdx][1], m_ref_path[refIdx+1][0]-m_ref_path[refIdx][0]);

    GetRelativePosition(RefPtX, RefPtY, RefTh, 
                        m_car.x, m_car.y, m_car.th, 
                        x_err, y_err, heading_err);

    cout<< "x_err: " << x_err << endl;
    cout<< "y_err: " << y_err << endl;

    curvature = CalculateCurvature(refIdx);
    
    FF_term = curvature * ref_velo;
    FF_gain = 0.175; //no fb or ky=1, kth=0
    // FF_gain = 0.2;

    FB_term = ref_velo * ( K_y * y_err + K_th * sin(heading_err));
    
// Lateral Control
    double kanayama_angular_velo = FF_gain * FF_term + FB_term;
    steerAngle = -(_RAD2DEG*SteerRatio)*atan2(WHEEL_BASE*kanayama_angular_velo, ref_velo*KMpH2MpS);
    if(abs(steerAngle) > LimitSteering)
        steerAngle = SIGN(steerAngle) * LimitSteering;

// Longitudinal Control
    velo_result = (ref_velo * cos(heading_err)); // basic KANAYAMA // [km/h]
    // velo_result = (ref_velo * cos(heading_err) + K_x * x_e); // basic KANAYAMA
    // As the fixed ref_pose is used, neglect the 'K_x*x_e' term.

    /* ========== The problem #4 for homework 3. Longitudinal controller considering vehicle curvature errors ==========
    Kim, Minsoo, et al. "A Comparative Analysis of Path Planning and Tracking Performance According to the Consideration 
    of Vehicle's Constraints in Automated Parking Situations." The Journal of Korea Robotics Society 16.3 (2021): 250-259. */
    double K_v = 0.3;
    double vehicle_curv = velo_result*KMpH2MpS* tan(m_Steer_cmd / (_RAD2DEG*SteerRatio)) / WHEEL_BASE;
    velo_result = (velo_result != 0) ? velo_result * (1.0 - K_v * abs(CalculateCurvature(refIdx) - vehicle_curv) / (2.0 * m_curvature_max)) : 0.0; // [m/s]
    velo_result = parkingVeloController(velo_result);
    //

    velo_result = max(velo_result, MIN_VEL_INPUT);
    velo_result = min(velo_result, MAX_VEL_INPUT);
    PublishCarPose(m_car.x, m_car.y, m_car.th);
    PublishPoint(RefPtX, RefPtY);

    outputFile << RefPtX << " " << RefPtY << " " << RefTh << " " << m_car.x << " " <<  m_car.y <<" " <<  m_car.th << std::endl; // 파일에 데이터 입력


    return std::make_pair(steerAngle, velo_result);
}


/* MPC algorithm *//////////////////////////////////////////////////
//N: prediction horizon

    // int T = 1;
    double T = 1;
    int N = 20;
    // double v_max = LimitVelPerSec/60.0*0.05 , v_min = 0;
    double v_max = 5/3.6*dt , v_min =-v_max; //1.0/3.5/60.0;     
    // double phi_dot_max = LimitDegPerSec*_DEG2RAD/60.0*0.05, phi_dot_min = -phi_dot_max;
    double phi_dot_max = LimitDegPerSec*_DEG2RAD*dt*0.08, phi_dot_min = -phi_dot_max;

    double n_states =4, n_controls=2;
    double velo_result{0.0}; // [km/h]  
    double RefPtX, RefPtY ,RefPtth;
    double steerAngle{0.0}; // [deg]
    double duration;


    SX U= SX::sym("U",n_controls,N );
    SX P= SX::sym("P",n_states + N*(n_states+n_controls) ); //decision variable
    SX X= SX::sym("X",n_states,(N+1) );  //아하. 여기서 symbolic으로 선언해도 한번 하고 나면 얘가 값이 갱신되면서 바꾸니까 더이상 심볼릭이 아니게 되는거야
    // 한번 밖으로 빼려면 함수 선언하는 곳까지 다 빼야할 듯
    SX obj = 0.0;
    SXVector g;

    SX Q = SX::diag({35, 35, 38, 0.0006});  //26, 26, 38, 0.0006
    SX R = SX::diag({0.0, 0.0});
    SX R_u = SX::diag({0.2, 0.1}); //200, 100
    
    SXVector OPT_variables;
    SX st = X(Slice(), 0);

    SX con, A, B, st_next, st_next_euler, push, con_before;
    SXDict nlp_prob;
    Dict opts;


    // Function solver = nlpsol("solver", "ipopt", nlp_prob, opts);  // 여기서 오류난다. 안에 들어가는 인자들 크기 안맞아서

    int N_control_points = N + 1;
    int num_decision_variables = 4 * N_control_points + 2 * N;

    SX args_lbg(4*(N_control_points),1), args_ubg(4*(N_control_points),1);
    SX args_lbx(num_decision_variables,1), args_ubx(num_decision_variables,1); 
    
    Function solver;

    double t0 = 0.0;
    SX x0, xs;
    int mpciter =0;   //여기 mpciter를 전역변수로 해서 기억하게 두니까 while문에서도 이게 기억돼서 돌아가는거다. 
    SX xx(4,iteration);
    SX t(1,iteration); 
    SX u0 = SX::zeros(N, 2);
    SX X0(4, N+1);




    DM xx1, u_cl;
    SX v_index=0, a=0;
    SXVector b= {0};
    double index_change = 0;

    vector<double> m_ref_path_theta;

    int ref_index_ =0;
    
    // double ref_index_ =0;

    double phi_ref = 0.0;
    double u_ref = 0.0, phi_dot_ref = 0.0;

    




pair<double,double> MPC() { //////////////////////////////////////////////////////////////////////////////////////////////////

       // 여기다 둬도 되나???
    // outputFile.open("data_seh.txt", std::ios::app);



    // std::ofstream outputFile("data.txt");

    // outputFile << m_ref_path[m_carIdx][0] << " " << m_ref_path[m_carIdx][1] << " " << m_ref_path_theta[m_carIdx] << " " << m_car.x << " " <<  m_car.y <<" " <<  m_car.th << std::endl;


    // 파일 스트림 닫기
    // outputFile.close();




    clock_t start, finish;

    start = clock();


    std::cout << std::endl <<"MPC 시작합니다" << std::endl;

    int sim_tim = m_ref_path.size();  
;

    // finish = clock();
    // duration = (finish - start);
    // std::cout <<"duration(변수 선언 시간): " << duration << std::endl;

//////////////////// while문 시작 ///////////////////////////////////////////////////////////////////////////////////////////////////
        // double current_time = mpciter * T;
        //std::cout << "x0: " << x0 << std::endl;
        // std::cout << "obj: " << obj << std::endl;

        P(Slice(0, 4)) = x0;
        P(0) = m_car.x;  // 여기 현재 값을 넣어주는 게 맞는거 같은데 
        P(1) = m_car.y;
        P(2) = m_car.th;
        // P(3) = -x0(3);
        //cout << "P(3): " << P(3) << endl;

        

        int ref_index =0;
        for (int k = 0; k < N; ++k) {    //k가 0부터 들어가서 1씩 증가해서 N까지 들어가는거다. 
            // double t_predict = current_time + k * T;

     //cout<<"m_carIdx: " << m_carIdx << endl;       

            
            // int ref_index = std::round(t_predict);  //round는 그냥 반올림함수.     // 여기 ref_path 넣어주는 곳을 잘 설계하면 될 것 같다. 
            // ref_index = mpciter + k;
            // ref_index = ref_index_ + round(duration/CLOCKS_PER_SEC*velo_result*100000/60/60*k)+1;
            // cout << "duration/CLOCKS_PER_SEC : " << duration/CLOCKS_PER_SEC <<endl;
            // double ref_increment = velo_result*k*1000/60/60;
            // ref_index = ref_index_ + round(ref_increment);
 

            ref_index = m_carIdx  + 6*k;

            //ref_index = ref_index_ + k ;
            //cout << "m_carIdx: " << m_carIdx << endl;

           
            if (k==1){
                ref_index_ = ref_index;
            }

            //cout << "ref_index: " << ref_index << endl;


            //cout << "ref_index 증가분: " << ref_increment<< endl;
            //cout << "ref_index 증가분(round): " << round(duration/CLOCKS_PER_SEC*velo_result*k) << endl;
            
            if (ref_index >= m_ref_path.size()) {         // 내가 그 다음 ref로 잡은 곳을 N등분한 index를 넣어준다.
                ref_index = m_ref_path.size() - 1;
            }

            //std::cout << "ref_index: " << ref_index << std::endl;      // 1~10 ,2~11, 3~12, 4~13 ref path가 이렇게 들어가도록

            // DM x_ref = ref_path(ref_index, 0); 
            DM x_ref = m_ref_path[ref_index][0]; 

            //std::cout << "x_ref: " << x_ref << std::endl;
            DM y_ref = m_ref_path[ref_index][1];
            //std::cout << "y_ref: " << y_ref << std::endl;
            DM theta_ref = m_ref_path_theta[ref_index];///////////////////////////-> 여기 잠깐만
            //std::cout << "theta_ref: " << theta_ref << std::endl;

            double phi_ref = 0.0;
            double u_ref = 0.0, phi_dot_ref = 0.0;

            P(6 * k + 4) = x_ref; P(6 * k + 5) = y_ref; P(6 * k + 6) = theta_ref; P(6 * k + 7) = m_Steer_cmd; //여기에 이상한 값이 들어가나?
            P(6 * k + 8) = m_car.vel; P(6 * k + 9) = phi_dot_ref;
         }

        std::cout << "x_ref: " <<  m_ref_path[ref_index_][0] << std::endl;
        std::cout << "y_ref: " <<  m_ref_path[ref_index_][1] << std::endl;
        std::cout << "theta_ref: " <<  m_ref_path_theta[ref_index] << std::endl;




        SX args_x0 = SX::vertcat({reshape(X0.T(), 4 * N_control_points, 1),  reshape(u0.T(), 2 * N, 1) }); //여기 다시해야겠다.
 
        DMDict arg = { {"x0", args_x0}, {"lbx", args_lbx}, {"ubx", args_ubx}, {"lbg", args_lbg},{"ubg", args_ubg}, {"p", P} };

        DMDict sol = solver(arg); /////////////////////////////////////////////////////솔버 푸는 부분!!!
        



        // cout<<"솔버 결과값: " << sol["x"](0) << endl;
        // cout<<"솔버 크기: " << sol["x"].size() << endl;

        DM u = reshape(sol["x"](Slice(4 * (N + 1), sol["x"].size1()) ).T(), 2,N).T(); //이거 크기 10*2 이다. 
        // ////////////////////////
        nav_msgs::Path gui_path;
        gui_path.poses.resize(N + 1);
        gui_path.header.frame_id = "map";
        // gui_path.header.stamp = ros::Time::now();


        for(int i=0; i < N+1; i++)
        {
            gui_path.poses[i].header.frame_id = "map"; // Edit here if tf name is different
            // gui_path.poses[i].header.stamp = ros::Time::now();
            gui_path.poses[i].pose.position.x = double(sol["x"](4 * i));
            gui_path.poses[i].pose.position.y = double(sol["x"](4 * i + 1));
            gui_path.poses[i].pose.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY( 0, 0, double(sol["x"](4 * i + 2)) );
            tf2::convert(q, gui_path.poses[i].pose.orientation);
        }
        Pub_mpcTraj.publish(gui_path);
        // //////////////////////

        // cout << "u(0,0) => V값: " << u(0,0) << endl;

        // velo_result = double(u(0,0))*60*60/1000;
        // velo_result = double(u(0,0))*200; //이거 200을 곱하면 안되는거 같다
        velo_result = double(u(0,0))*60*3.6;
        cout << "velo_result[km/h]: " << velo_result << endl;


        u_cl = vertcat(u_cl, u(0,Slice()));  // 이렇게 하면 안될 거 같은데 
        t(mpciter) = t0;
        //velo_result = double(u_cl(u_cl.size1()-1,0))*1000; 
        
        
        //std::cout << "u_cl.size(): " << u_cl.size1() << std::endl;


/////////////////////// shift 함수 정의하기//////////////////////////////////////////////

        st = x0; 


        // cout << "tan(st(3)): " << tan(st(3)) << endl;

        con = u(0, Slice()).T();
        //st = st + (T*rhs(st, con))/ (CLOCKS_PER_SEC/duration); // 이게 되네??
        st = st + (T*rhs(st, con)); 
        // st(2) = st(2) - floor(st(2)/6.2832)*6.2832; // 얘가 몫이니까 
        //st = st + (T*rhs(st, con))*dt;
        // cout <<"dt: " << dt << endl;

        // st_sh = st_sh + (T*rhs(st_sh, con_sh));
        // cout << "con: " <<con <<endl;
        // cout << "st: " <<st <<endl;
        // cout << "con(1),phi_dot = " << con(1) << endl;

        // cout << "T*rhs(st, con): " << T*rhs(st, con) <<endl;
        x0 = st;
        t0 = t0 + T;
        u0 = vertcat(u(Slice(1,u.size1()),Slice()),  u(u.size1()-1,Slice())); 

//////////////////////////////////////////////////////////////////////////////////////////


        //std::cout << "x0: " << x0 <<std::endl;  //
        xx(Slice(),mpciter+1) = x0;
        RefPtX = m_ref_path[ref_index_][0];
        RefPtY = m_ref_path[ref_index_][1];
        // steerAngle = -double(x0(3))*_RAD2DEG;
        // cout << "double(x0(3)),phi 값[degree]: " << double(x0(3))*_RAD2DEG << endl;
        // steerAngle = -double(x0(3))*_RAD2DEG*9;
        steerAngle = -(double(x0(3))) *_RAD2DEG *SteerRatio ;
        // steerAngle = -(double(x0(3))) *_RAD2DEG ;
        cout << "steerAngle: [degree* SteerRatio] " << steerAngle << endl;

        cout << "mpciter: " << mpciter << std::endl;

        // cout << "steerAngle: " << steerAngle << std::endl; 
        // cout << "theta[degree],x0(2): " << double(x0(2))*_RAD2DEG << std::endl; 

        X0 = reshape(sol["x"](Slice(0,4*(N+1))).T() ,4 ,N+1).T();
        //std::cout << "X0: " << X0 << std::endl;  //이거 full만 해주면 될 듯!
        //std::cout << "X0.size(): " << X0.size() << std::endl;

        X0 = vertcat(X0(Slice(1,X0.size1()),Slice()),  X0(X0.size1()-1,Slice()));

        
        mpciter++;


        
        // velo_result = max(velo_result, MIN_VEL_INPUT);
        // velo_result = min(velo_result, MAX_VEL_INPUT);

        // cout<< "velo_result: " <<  velo_result << endl;
        cout << "dt: " << dt << endl;


        PublishCarPose(m_car.x, m_car.y, m_car.th); // 현재의 car의 위치랑
        PublishPoint(RefPtX, RefPtY); //ref_point들의 위치 

        finish = clock();

        duration = (finish - start);
        std::cout <<"duration: " << duration/CLOCKS_PER_SEC << std::endl;

        outputFile << m_ref_path[m_carIdx][0] << " " << m_ref_path[m_carIdx][1] << " " << m_ref_path_theta[m_carIdx] << " " << m_car.x << " " <<  m_car.y <<" " <<  m_car.th << std::endl; // 파일에 데이터 입력
        // outputFile << dt << " " << duration/CLOCKS_PER_SEC << std::endl; // 파일에 데이터 입력

        
        


        // outputFile.close();
    
        return std::make_pair(steerAngle, velo_result);


    }



////////////////////////////////////// while문 끝 /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    




// Assumes, forward motion only
void CallbackRefPath(const nav_msgs::Path::ConstPtr& msg) {   //ROS�� ���ȸ� subscribe�ؼ� ���ο� ��� ������ �������� �� ȣ��Ǵ� �Լ�
    cout << "reference path callback!" << endl;      //nav_msgs�� path �޼����� �޾ƿͼ� ��� ������ ó���ϰ�, �� ������ 
    if (is_init_path) {
        is_init_path = false; // MK add
        start_ros_time = ros::Time::now();
    }
    
    // MemberValInit();
    if (msg->poses.size() != 0) {
        // check forward/backward switching
        double first_th_, second_th_;
        int pathIdx = 0;

        m_init_pose = Vector3d(msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 0);
        double goal_th = atan2(msg->poses.back().pose.position.y - msg->poses[msg->poses.size()-2].pose.position.y,
                               msg->poses.back().pose.position.x - msg->poses[msg->poses.size()-2].pose.position.x);
        cout << "GOAL: " << msg->poses.back().pose.position.x << ", " << msg->poses.back().pose.position.y << endl;
        m_goal_pose = Vector3d(msg->poses.back().pose.position.x, msg->poses.back().pose.position.y, goal_th);

        // Substitute the first value
        m_ref_path.push_back(Vector3d(msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 0)); //���⺸�� ������ ���� ��� ������ m_ref_path�� ������ �ְ� �ִ� (push)
        cout << "Callback ReferencePath # Size (# of way-points) : " << msg->poses.size() << endl;
        // Each pathIdx is represented as the index of path segment, which is distinguished by Gear Switching.
        for(int i = 0; i < msg->poses.size(); i++)
            m_ref_path.push_back(Vector3d(msg->poses[i].pose.position.x, msg->poses[i].pose.position.y, i));
        m_pathFlag = true;
    }
///////////////////////MPC 추가한 부분 /////////////////////////////////////
    // std::cout <<"m_ref_path.size()여기가 되나 3 ??: " << m_ref_path.size() << std::endl;
    m_ref_path_theta.push_back(0);   // 이렇게 해서 최대한 시간 단축했는데 그래도 여기서 2000정도 쓰인다. 나중에 시간되면 더 단축시키는게 좋겠다. 
    for (size_t i = 0; i < m_ref_path.size(); ++i) {
        m_ref_path_theta.push_back( atan2(m_ref_path[i+1][1] - m_ref_path[i][1] , m_ref_path[i+1][0] - m_ref_path[i][0]));
    }
    // m_ref_path[m_ref_path.size()-2]
    m_ref_path_theta[0] = m_ref_path_theta[3];
    m_ref_path_theta[1] = m_ref_path_theta[3];

    x0 = {m_ref_path[0][0], m_ref_path[0][1], m_ref_path_theta[0], 0.0}; // 이게 while문 앞에 있으면 x0를 기억해서 못 쓰게 돼. 
    // xs = {m_ref_path[m_ref_path.size()-2][0], m_ref_path[m_ref_path.size()-2][1], m_ref_path_theta[m_ref_path.size()-2], 0.0}; 
    xs = {m_ref_path[0][0], m_ref_path[0][1], m_ref_path_theta[0], 0.0};
    t(0) = t0; 

    for (int i = 0; i < N+1; i++) {
        for (int j = 0; j < 4; j++) {
            X0(j, i) = x0(j);
        }
    }

    

    // outputFile.open("data_seh.txt", std::ios::app);
    // std::ofstream outputFile("data.txt");

    // cout << "m_ref_path_theta : " << m_ref_path_theta << endl;
//////////////////////////////////////////////////////////////////////////////////

}


void vehicleTrajectory(int time) {// Vehicle's future trajectory //��.. ����� ���ϴ� ���ϱ�?
    double dt = 0.1;
    nav_msgs::Path msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "map";
    msg.poses.resize((int)(time/dt));

    double yaw = 0.0, xL = 0.0, yL = 0.0;
    for (int i = 0 ; i < (int)(time/dt) ; i++) {  // a unit of the time is second.
        xL += dt*m_dir_mode*m_car.vel*cos(yaw);
        yL += dt*m_dir_mode*m_car.vel*sin(yaw);
        yaw += dt*dt*m_dir_mode*m_car.vel*atan2(-m_Steer_cmd*_DEG2RAD, 1.0);///2.85;
        Local2Global(xL, yL, msg.poses[i].pose.position.x, msg.poses[i].pose.position.y);
        msg.poses[i].pose.position.z = 0.0;
    }
    Pub_vehicleTraj.publish(msg);
    msg.poses.clear();
}

// Publish control commands as a topic for controlling the vehicle
void VehicleControl(double steer, double velocity) {
    // Physical limit
    dt = (ros::Time::now() - ros_time2).toSec();
    
    if (LimitDegPerSec*dt < abs(steer - m_Steer_cmd)) // 그니까 저번 steer_cmd랑 이번에 계산한 steer랑 차이가 심할 경우 이렇게 한다는 거지. 
       //  phi_dot으로 constraint 걸어서 하고 있는데 이게 이렇게 되는 이유가 뭐지?
        m_Steer_cmd += (double)(SIGN(steer - m_Steer_cmd)*LimitDegPerSec*dt);   // 지금 여기에 걸려서 값이 바뀌고 있는거지? 
    else
        m_Steer_cmd = steer;
    
    if (LimitVelPerSec*dt < abs(velocity - m_Velocity_cmd))
        m_Velocity_cmd += (double)(SIGN(velocity - m_Velocity_cmd)*LimitVelPerSec*dt);
    else
        m_Velocity_cmd = velocity;

    // cout << "                                  dt: " << dt << endl;

    
    // cout << "                                  steer: " << steer << endl;
    // cout << "                                  m_Steer_cmd: " << m_Steer_cmd << endl;
    // cout << "                                  m_Velocity_cmd: " << m_Velocity_cmd << endl;

    // For CARLA   
    std_msgs::Float32MultiArray msg_;
    msg_.data.push_back(m_Steer_cmd);
    msg_.data.push_back(m_Velocity_cmd);
    Pub_ControlCmd.publish(msg_);
}

void PublishTopic() {
    static int pub_cnt;
    pub_cnt++;
    if (pub_cnt == 7) {
        m_CarPosLine.points.push_back(m_CarPos.pose.position);
        pub_cnt = 0;
    }
    if (Pub_MarkerCar.getNumSubscribers() > 0)
        Pub_MarkerCar.publish(m_CarPosLine);
}

// Find the closest way-point from the vehicle
double CalculateClosestPt(int& pWIdx, double xx, double yy) {
    double minDist = 99999, dist = 0;
    for(int i = 0; i < m_ref_path.size() - 1; i++) {
        dist = DISTANCE(m_ref_path[i][0], m_ref_path[i][1], xx, yy);
        if(dist < minDist) {
            minDist = dist;
            pWIdx = i;
        }
    }
    m_carIdx = pWIdx;
    
    return minDist;
}

// Check that the vehilce reaches the goal pose. If the distance between the vehicle's position and goal is lower than 'check_thres', then the vehicle reaches the goal.
void GoalCheck(double check_thres, int which_tracker) {
    bool goal_checking_front = (which_tracker==2) ? true : false;
    if (!goal_checking_front && almost_done_flag && (DISTANCE(m_car.x, m_car.y, m_goal_pose[0], m_goal_pose[1]) < check_thres)) {
        std::cout << "Goal!" << std::endl;
        double tracking_time = (ros::Time::now() - start_ros_time).toSec();


        outputFile.close();
        if (!outputFile.is_open()) {
        std::cout << "파일이 정상적으로 닫혔습니다." << std::endl;
        } else {
        std::cout << "파일을 닫을 수 없습니다." << std::endl;
        }

        
        double rel_x, rel_y, rel_th;
        GetRelativePosition(m_car.x, m_car.y, m_car.th, m_goal_pose[0], m_goal_pose[1], m_goal_pose[2], rel_x, rel_y, rel_th);
        std::cout << "Tracking time [s]: " << tracking_time << std::endl;
        std::cout << "Lateral error [m]: " << abs(rel_y) << std::endl;
        std::cout << "Orientation error [deg]: " << _RAD2DEG * atan2(sin(rel_th), cos(rel_th)) << std::endl;

        // For CARLA
        std_msgs::Float32MultiArray msg_;
        msg_.data.push_back(0.0);
        msg_.data.push_back(0.0);
        Pub_ControlCmd.publish(msg_);
        m_finish_flag = true;
    }
    double car_front_x = m_car.x+WHEEL_BASE*cos(m_car.th);
    double car_front_y = m_car.y+WHEEL_BASE*sin(m_car.th);
    double goal_front_x = m_goal_pose[0]+WHEEL_BASE*cos(m_goal_pose[2]);
    double goal_front_y = m_goal_pose[1]+WHEEL_BASE*sin(m_goal_pose[2]);

    if (goal_checking_front && almost_done_flag && (DISTANCE(car_front_x, car_front_y, goal_front_x, goal_front_y) < check_thres)) {
        std::cout << "Goal!" << std::endl;
        double tracking_time = (ros::Time::now() - start_ros_time).toSec();

        outputFile.close();
        if (!outputFile.is_open()) {
        std::cout << "파일이 정상적으로 닫혔습니다." << std::endl;
        } else {
        std::cout << "파일을 닫을 수 없습니다." << std::endl;
        }
        
        double rel_x, rel_y, rel_th;
        GetRelativePosition(car_front_x, car_front_y, m_car.th, goal_front_x, goal_front_y , m_goal_pose[2], rel_x, rel_y, rel_th);
        std::cout << "Tracking time [s]: " << tracking_time << std::endl;
        std::cout << "Lateral error [m]: " << abs(rel_y) << std::endl;
        std::cout << "Orientation error [deg]: " << _RAD2DEG * atan2(sin(rel_th), cos(rel_th)) << std::endl;

        // For CARLA
        std_msgs::Float32MultiArray msg_;
        msg_.data.push_back(0.0);
        msg_.data.push_back(0.0);
        Pub_ControlCmd.publish(msg_);
        m_finish_flag = true;
    }




    return;
}

// Check that how much distance to the goal is remained 
void remain_path_check(double close_to_goal_thres) {
    if (almost_done_flag)
        return;

    m_remainIdx = m_ref_path.size() - m_carIdx;
    if (close_to_goal_thres > (double)m_remainIdx * 0.01) {
        int path_size = m_ref_path.size();
        double first_th = atan2(m_ref_path[path_size-2][1] - m_ref_path[path_size-3][1], m_ref_path[path_size-2][0] - m_ref_path[path_size-3][0]);
        double second_th = atan2(m_ref_path[path_size-1][1] - m_ref_path[path_size-2][1], m_ref_path[path_size-1][0] - m_ref_path[path_size-2][0]);
        double th_diff = second_th - first_th;
        if( th_diff > M_PI) th_diff = th_diff - 2*M_PI;
        else if( th_diff < -M_PI ) th_diff = 2*M_PI - abs(th_diff);
        double extend_distance = DISTANCE(m_ref_path[path_size-2][0], m_ref_path[path_size-2][1],
                                    m_ref_path[path_size-1][0], m_ref_path[path_size-1][1]);
        for(int j = 1 ; j <= 2000; j++)// 20 m extension
            m_ref_path.push_back(Vector3d(m_ref_path[path_size-2+j][0] + extend_distance*cos(second_th + j*th_diff), 
                                            m_ref_path[path_size-2+j][1] + extend_distance*sin(second_th + j*th_diff),
                                            path_size));
        almost_done_flag = true;
        return;
    }
}

// Compute the control commands for tracking
void Compute() {
    int pWIdx = 0;
    if (!m_finish_flag && m_ref_path.size() > 1) {
        pair<double, double> _control;
        switch(LOCAL_TRACKER) {
            case 1: {// PurePursuit
                CalculateClosestPt(pWIdx, m_car.x, m_car.y); // Computing the closest point from the rear axle
                _control = PurePursuit(2.0, 4.0);  ///
                break;
            }
            case 2: {// Stanley
                double min_dist = CalculateClosestPt(pWIdx, m_car.x + WHEEL_BASE*cos(m_car.th), m_car.y + WHEEL_BASE*sin(m_car.th)); //Computing the closest point from the front axle
                _control = Stanley(5.0, 4.0, min_dist);           
                break;
            }

            case 3: {// Kanayama
                CalculateClosestPt(pWIdx, m_car.x, m_car.y); // Computing the closest point from the rear axle
                _control =  Kanayama(0.75, 0.01, 6.0);
                break;
            }

            case 4: {//MPC
                //나는 무슨 값을 계산해서 받아와야 할까.
                CalculateClosestPt(pWIdx, m_car.x, m_car.y);
                _control = MPC();
                break;
            
            }


            default: {
                _control = PurePursuit(1.0, 3.0);
                break;
            }
        }
        remain_path_check(5.0);
        
        VehicleControl(_control.first, m_dir_mode*_control.second); //[km/h]
        ros_time2 = ros::Time::now();
        GoalCheck(0.2, LOCAL_TRACKER); // When you use Stanley, set it to 'true', otherwise 'false'.
    }
    PublishTopic();
}


void CallbackLocalizationData(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    m_car.x = msg->data.at(0);      // x
    m_car.y = msg->data.at(1);      // y
    m_car.th = msg->data.at(2);     // theta
    m_car.vel = msg->data.at(3)*KMpH2MpS;    // to [m/s]
    
    // Set a center to rear axis from CoM
    double c2r = -1.2865;
    m_car.x += c2r*cos(m_car.th);
    m_car.y += c2r*sin(m_car.th);

    ////////////////////// MPC path 계산하는 부분 ///////////////////// 근데 여기다 넣어도 계속 반복해서 계산하는건 똑같은거 같은데 
    // std::cout <<"m_ref_path.size()여기가 되나??: " << m_ref_path.size() << std::endl;
    // m_ref_path_theta.push_back(0);   // 이렇게 해서 최대한 시간 단축했는데 그래도 여기서 2000정도 쓰인다. 나중에 시간되면 더 단축시키는게 좋겠다. 
    // for (size_t i = 0; i < m_ref_path.size(); ++i) {
    //     m_ref_path_theta.push_back( atan2(m_ref_path[i+1][1] - m_ref_path[i][1] , m_ref_path[i+1][0] - m_ref_path[i][0]));
    // }
    //////////////////////////////////////////////////////////////////
    ros_time = ros::Time::now();
    if (m_pathFlag && !m_finishFlag) {//PATH IS GENERATED, WHICH CONSISTS OF PATH SEGMENTS
    
        Compute();
    }
    else {
        VehicleControl(0.0, 0.0);
    }
    vehicleTrajectory(1.5);    // based on the vehicle's kinematic model
}


void MarkerInit() {
    m_CarPosLine.points.clear();
    m_CarPos.header.frame_id =  m_CarPosLine.header.frame_id = "map";
    m_CarPos.ns = m_CarPosLine.ns = "RegionOfInterest";
    m_CarPos.id = 0;
    m_CarPosLine.id = 1;
    
    m_CarPos.type = visualization_msgs::Marker::ARROW;
    m_CarPosLine.type = visualization_msgs::Marker::LINE_STRIP;
    m_CarPos.action = m_CarPosLine.action = visualization_msgs::Marker::ADD;

    m_CarPos.scale.x = 4.5/2;
    m_CarPos.scale.y = 1.5;
    m_CarPos.scale.z = 0.1;
    m_CarPos.color.a = 0.7;
    m_CarPos.color.r = 1.0f;
    m_CarPos.color.g = 0.0f;
    m_CarPos.color.b = 0.0f;
    m_CarPos.pose.position.z = 0.5;

    m_CarPosLine.scale.x = 0.15;
    m_CarPosLine.pose.position.z = 1.2;
    m_CarPosLine.color.r = 1.0;
    m_CarPosLine.color.a = 0.7;

    
    m_point_marker.header.frame_id = "map";
    m_point_marker.header.stamp = ros::Time();
    m_point_marker.ns = "point";
    m_point_marker.id = 0;
    m_point_marker.type = visualization_msgs::Marker::SPHERE;
    m_point_marker.action = visualization_msgs::Marker::ADD;
    m_point_marker.pose.position.x = 1;
    m_point_marker.pose.position.y = 1;
    m_point_marker.pose.position.z = 1;
    m_point_marker.pose.orientation.x = 0.0;
    m_point_marker.pose.orientation.y = 0.0;
    m_point_marker.pose.orientation.z = 0.0;
    m_point_marker.pose.orientation.w = 1.0;
    m_point_marker.scale.x = 0.5;
    m_point_marker.scale.y = 0.5;
    m_point_marker.scale.z = 0.0;
    m_point_marker.color.a = 1.0;
    m_point_marker.color.r = 1.0;
    m_point_marker.color.g = 0.0;
    m_point_marker.color.b = 0.0;
}

void LOCALPLANNERTHREAD() {   //여기는 ros랑 통신하는 부분
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "Tracker");   
    ros::NodeHandle nh_, anytime_node;
    ros::NodeHandle priv_nh("~");
    MarkerInit();

    ros::CallbackQueue anytime_queue;
    anytime_node.setCallbackQueue(&anytime_queue);


    Pub_MarkerCar = nh_.advertise<visualization_msgs::Marker>("markerVehicleTraj", 1); 
    Pub_poseVehicle = nh_.advertise<geometry_msgs::PoseStamped>("poseVehicle", 1);  
    Pub_vehicleTraj = nh_.advertise<nav_msgs::Path>("vehicle_traj", 1);  
    Pub_mpcTraj = nh_.advertise<nav_msgs::Path>("mpc_traj", 1);  
    Pub_ControlCmd = nh_.advertise<std_msgs::Float32MultiArray>("Control_Command", 1);  
    Pub_finish = nh_.advertise<std_msgs::Float32MultiArray>("FinishFlag", 1);  
    Pub_Point = nh_.advertise<visualization_msgs::Marker>("visualization_marker", 0 );
    
    Sub_localization = nh_.subscribe("/LocalizationData", 10, &CallbackLocalizationData);
    Sub_refPath = nh_.subscribe("ref_path", 1, &CallbackRefPath);// From doRRT node

    // MemberValInit();
    cout << "START CONTROLLER !!!" << endl;
//////여기다 mpc init 함수 넣어서 
    


//////////////////
    ros::AsyncSpinner spinner(0, &anytime_queue);
    spinner.start();
    ros::spin();
}


void MPC_init() {  
    
    g = {st - P(Slice(0, 4))};
    

    for (int k = 0; k < N; ++k) {    
        st = X(Slice(), k); // initial state 값. 
        con = U(Slice(), k);
        if( k != 0)
        {
            con_before = U(Slice(), k-1);
        }

        // st = X(Slice(), k);   
        // con = U(Slice(), k); // 이걸 차량의 현재 값을 넣어야하는거 아닌가?

        SX error_pose;

        A = P(Slice(6*k+4,6*k+8)); 
        B = P(Slice(6*k+8,6*k+10));
        error_pose = st - A;
        error_pose(2) = atan2(sin(error_pose(2)), cos(error_pose(2)));
        obj = obj + ( mtimes(mtimes((error_pose).T(),Q) , (error_pose))) +  (mtimes(mtimes((con-B).T() , R) ,(con-B) ));
        if (k!=0){
            obj = obj + mtimes(mtimes((con - con_before).T() , R_u) ,(con - con_before));

        }

        st_next = X(Slice(), k+1);
        st_next_euler = st+ (T*rhs(st, con));
        push = st_next-st_next_euler;  

        g.push_back(st_next-st_next_euler); 
    }

    OPT_variables.push_back(reshape(X, 4 * (N+1), 1));
    OPT_variables.push_back(reshape(U, 2 * (N), 1));


    // SXDict nlp_prob;
    nlp_prob["f"] = obj;
    nlp_prob["x"] = vertcat(OPT_variables);
    nlp_prob["g"] = vertcat(g);
    //std::cout << "vertcat(g):" << nlp_prob["g"] << std::endl;
    //std::cout << "vertcat(g).size:" << nlp_prob["g"].size() << std::endl;
    nlp_prob["p"] = P;


    // Dict opts;
    opts["ipopt.max_iter"] = 5000;
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    opts["ipopt.acceptable_tol"] = 1e-8;
    opts["ipopt.acceptable_obj_change_tol"] = 1e-6;

    solver = nlpsol("solver", "ipopt", nlp_prob, opts);

    int N_control_points = N + 1;
    int num_decision_variables = 4 * N_control_points + 2 * N;

    // SX args_lbg(4*(N_control_points),1), args_ubg(4*(N_control_points),1);
    // SX args_lbx(num_decision_variables,1), args_ubx(num_decision_variables,1); 
-
    args_lbg(Slice(0,4*(N + 1)),0) = 0.0;    //Equality constraints.
    args_ubg(Slice(0,4*(N + 1)),0) = 0.0;

     //                    행         열
    args_lbx(Slice(0, 4*(N + 1), 4) ,0) = -inf; // state x lower bound  
    args_ubx(Slice(0, 4*(N + 1), 4) ,0) = inf; // state x upper bound
    args_lbx(Slice(1, 4*(N + 1), 4) ,0) = -inf; // state y lower bound
    args_ubx(Slice(1, 4*(N + 1), 4) ,0) = inf; // state y upper bound
    args_lbx(Slice(2, 4*(N + 1), 4) ,0) = -inf; // state theta lower bound
    args_ubx(Slice(2, 4*(N + 1), 4) ,0) = inf; // state theta upper bound
    // args_lbx(Slice(2, 4*(N + 1), 4) ,0) = -6.2832; // state theta lower bound
    // args_ubx(Slice(2, 4*(N + 1), 4) ,0) = 6.2832; // state theta upper bound
    // args_lbx(Slice(3, 4*(N + 1), 4) ,0) = -inf; // state phi lower bound
    // args_ubx(Slice(3, 4*(N + 1), 4) ,0) = inf; // state phi upper bound
    args_lbx(Slice(3, 4*(N + 1), 4) ,0) = -35*_DEG2RAD; // state phi lower bound
    args_ubx(Slice(3, 4*(N + 1), 4) ,0) = 35*_DEG2RAD; // state phi upper bound


    args_lbx(Slice(4 * (N+1), 4*(N+1)+2*N, 2) ,0) = v_min; // v lower bound
    args_ubx(Slice(4 * (N+1), 4*(N+1)+2*N, 2) ,0) = v_max; // v upper bound
    args_lbx(Slice(4 * (N+1)+1, 4*(N+1)+2*N, 2) ,0) = phi_dot_min; // phi_dot lower bound
    args_ubx(Slice(4 * (N+1)+1, 4*(N+1)+2*N, 2) ,0) = phi_dot_max; // phi_dot upper bound

    //std::cout << "args_lbx:" << args_lbx << std::endl;
    //std::cout << "g(moving horizon의 constraint):" << g << std::endl;

    std::cout << "MPC_init 루프 끝났습니다. " << std::endl;


    // for (int i = 0; i < N+1; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         X0(j, i) = x0(j);
    //     }
    // }

    


}



int main(int argc, char* argv[]) //여기가 main함수니까 제일 먼저 실행되고
{   
    cout << "main 루프 시작합니다 " << endl;
    char tmp[256];
    getcwd(tmp, 256);
    cout<< tmp<<endl;

    outputFile.open("/home/dyros/data_seh_mpc_data_4.txt", std::ios::app); //파일 여는 건 여기다 두자.

    if (!outputFile.is_open()) {
    std::cout << "파일을 열 수 없습니다." << std::endl; //즉 파일은 열렸다는 건데.. 왜 저장이 안되지?
    }

    // outputFile <<" ~~~~~~ 파일에 적히나ㅏㅏ"  << std::endl; // 파일에 데이터 입력

    // outputFile.close();
    // if (!outputFile.is_open()) {
    // std::cout << "파일이 정상적으로 닫혔습니다." << std::endl;
    // } else {
    // std::cout << "파일을 닫을 수 없습니다." << std::endl;
    // }



    MPC_init();
    LOCALPLANNERTHREAD();  // localplannerthread 라는 함수가 실행된다.

    return 0;
} 