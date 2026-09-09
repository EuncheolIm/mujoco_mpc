#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>


#include <iostream>

#include <algorithm>
#include <atomic>
#include "PRBDL_l/Robot_config.h"

#define cuda_K 128  // Num of Sample
// #define cuda_K 1200  // Num of Sample
// #define cuda_K 128  // Num of Sample
#define cuda_T 30  /// Time Horizon

#define J_NUM 7
#define _lambda 1

#define cuda_dt  0.001 // time step

#define POS 3
#define ROT 9


#define _dt2_time_window cuda_T-10
#define _dt2_max 0.1

#define MainIndex cuda_K
#define Samplingndex cuda_K*J_NUM
#define ExpandingIndex cuda_K*cuda_T*J_NUM

#define ThreadNum 256

#define RESET "\033[0m"
#define RED "\033[31m"  /* Red */
#define BLUE "\033[34m" /* Blue */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define CYAN "\033[36m" /* Cyan */
#define MAGENTA "\033[35m" /* Magenta */

// test
#define IDX_VEC(j, t) ((j) * cuda_T + (t))

// time major order
// K=0, t =0  , j 0~6
// K=0, t=1   , j=0~6, ...
// K=0, t=T-1 , j=0~6,
// K=1, t=0,  , j=0~6
#define IDX_Q(j, tid, t) ((tid) * cuda_T * J_NUM + (t) * J_NUM + j)

// float* d_final_cost_all;  // 크기: J_NUM * cuda_K
#define IDX_FINAL_COST(j, tid) ((j) * cuda_K + (tid))

// extern __constant__ float d_tau_max[7];
// extern __constant__ float d_q_limit_min[7];
// extern __constant__ float d_q_limit_max[7];
// extern __constant__ float d_qdot_limit_max[7];

// extern __constant__ float d_Q;
// extern __constant__ float d_Q2;

class MPPI_qacc
{
    public:
    MPPI_qacc();
    virtual ~MPPI_qacc();

    int seed_count=0;
    int calculateBlocks(int totalThreads, int threadsPerBlock);

    float quadratic(float x);
    RobotBody RModel;

    int Sampling_block;
    int Sampling_block_all;
    int Expanding_block;
    int Main_block;

    float return_dq[J_NUM] = {0.0};
    float return_q[J_NUM] = {0.0};

    // float returnArray[2*2];
    float returnArray[J_NUM];

    float *d_des_pos_l;
    float *d_des_pos_r;

    float *dev_q, *dev_qdot;

    // 사실은 이게 next qacc 들임!

    float *d_cost_pos_l, *dev_cost_rpy_l;
    float *d_cost_u;
    float *d_cost_pos_r, *dev_cost_rpy_r;

    // float *_qdot;
    float u_vector[J_NUM*cuda_T]={0.0f,};

    float *dev_sigma;


    float x[1]={0.0f};
    float y[1]={0.0f};

    float des_pos_l[3]={0.0f};
    float des_pos_r[3]={0.0f};


    float cost_x[cuda_K*cuda_T]={0,};
    float cost_y[cuda_K*cuda_T]={0,};



    float cost_u[cuda_K*cuda_T]={0,};

    float _u_star[J_NUM]={0,};
    float u_star[J_NUM]={0,};
    // TODO:floating base case
    float _qdot[J_NUM]={0,};
    float _q[J_NUM]={0,};
    // TODO:floating base case
    float integrate_u_star[7]={0,};

    float u[cuda_K*cuda_T*J_NUM]={0,};
    float u_all[cuda_K*cuda_T*J_NUM]={0,};
    float u_single[cuda_K*J_NUM]={0,};



    // void  mppi_host(float *state1, float *state2, float* M, float* Cqdot, float* G, float* tau, float *returnArray);
    // void  mppi_host(float *state1, float *state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_rot, float *returnArray);

    // void  mppi_host(float *state1, float *state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_rot,
    //                 float *prev_position, float* prev_rotation, float *returnArray);

    // // MPPI -tau
    // void  mppi_host(float *state1, float *state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_pos, float* des_EE_rot,
    //                 float *prev_position, float* prev_rotation, float *returnArray);

    // I-MPPI
    void  mppi_host(float *state1, float *state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_pos, float* des_EE_rot,
                    float *prev_position, float* prev_rotation, float* qddot, float *returnArray);

    // MPPI no qacc
    void  mppi_host(float *state1, float *state2, float *prev_state1, float *prev_state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_pos, float* des_EE_rot,
                    float *prev_position, float* prev_rotation, float *returnArray);

    // MPPI no qacc - 2
    void  mppi_host(float *state1, float *state2, float *prev_state1, float *prev_state2, float* M, float* Cqdot, float* G, float* tau, float* des_EE_pos, float* des_EE_rot,
                    float *prev_position, float* prev_rotation, float* qddot, float *returnArray);

    // MPPI qacc
    void  mppi_host(float *state1, float *state2,
                    float* des_EE_pos_l, float* des_EE_rot_l,
                    float *prev_position_l, float* prev_rotation_l, float *returnArray);




    // TODO:floating base case
    float _q1[1];
    float _q2[1];
    float _q3[1];
    float _q4[1];
    float _q5[1];
    float _q6[1];
    float _q7[1];

    void state_init();
    void EstimateFK();


    void ComputeTransformationMatrix2_host(int horizon, int link_num, float* link_pos, float* Rotmat, float theta, float* _rho ,  // input
        float* B2Link_pos , float* B2Link_Rot, float* Pos_CoM);      // output
    void Multiply3x3_host(float* Mat1, float* Mat2, float* Mat3);

    void ComputeEachLinkPose_host(float* Rot_B2Lnk_parent, float* rpos_lnk_parent,                            // Input
        float* Pos_Link_child , float* Rot_Link_child, float* Pos_CoM_child,        // Input
        float* Rot_B2Lnk_child, float* rpos_lnk_child, float* rpos_lnk_CoM_child);   // output

    void GetPose2_host(float* B2Link_pos, float* B2Link_Rot,
        float* pos_offset, float* rot_offset, float* EE_pos);
    // TODO:floating base case
    float _pos[3], _gap[2], _pos_kju[3];
    float *d_u_all;
    float *d_u_single, _u_single[cuda_K *J_NUM];
    float *dev_next_q, *dev_next_qdot;

    float _sol_cost, _cuda_dt2[1];
    float *dev_cuda_dt2;

    float *dev_u_star;




    bool tau;

    float *dev_M, *dev_Cqdot, *dev_G, *dev_tau;


    float *dev_next_del_tau1, *dev_next_del_tau2, *dev_next_del_tau3, *dev_next_del_tau4, *dev_next_del_tau5, *dev_next_del_tau6, *dev_next_del_tau7;
    // float Mass[2];
    // float Length[2];
    // float gravi[0];


    cudaStream_t mppi_stream;

    //////////////////////////////////////////////
    ////////        Dynamics    //////////////////
    //////////////////////////////////////////////
    void cuda_Initialize();
    void cuda_memory();
    void cuda_data_free();
  
    // host 쪽
    float h_PosLink[J_NUM * POS];
    float h_RotLink[J_NUM * ROT];

    float h_rho[J_NUM * POS];
    float h_RotCoM[J_NUM * ROT];

    float h_axis[J_NUM * POS];
    float h_I_G[J_NUM * ROT];

    float h_mass[J_NUM];
    float h_Tot_mass[1];
    float h_id_parents[J_NUM];

    // device 쪽
    float *dev_PosLink;
    float *dev_RotLink;

    float *dev_rho;
    float *dev_RotCoM;

    float *dev_axis;
    float *dev_I_G;

    float* dev_mass;
    float* dev_Tot_mass;
    float *dev_id_parents;


    float *dev_des_EE_rot_l, *dev_des_EE_rot_r;

    float exp_sv[J_NUM];

    float fixed_u_star[J_NUM];

    float id_parents[J_NUM] = {0.0};

    int reset_check = 0;
    float des_state1[J_NUM]={0.0};
    float des_state2[J_NUM]={0.0};


    // test
    int total_KT = J_NUM * cuda_K * cuda_T;
    int total_T = J_NUM * cuda_T;
    int total_K = J_NUM * cuda_K;

    float* dev_next_tau_all;      // J_NUM * K * T<
    float* dev_next_del_tau_all;  // J_NUM * K * T

    float* dev_next_q_all;  // J_NUM * K * T
    float* dev_next_qdot_all;  // J_NUM * K * T

    float* dev_u_vector_all;  // J_NUM *  T
    float* dev_final_cost_all;  // J_NUM * K


    float u_vector_all[J_NUM * cuda_T] = {0.0f};

    // for generating random values
    curandState *d_states;
};