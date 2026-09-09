#include "MPPI_qacc.cuh"
#include <random>
#include <chrono>
#include <iomanip>

#include "cudacommon.h"
#include <fstream>
#include <iostream>
__device__ clock_t last_print_time = 0;

using namespace std;

MPPI_qacc::MPPI_qacc()
{
    cuda_memory(); // initialize cuda memory and data
	cuda_Initialize(); // RModel로 선언된 로봇에서 mass 및 link 정보를 업데이트, param.yaml을 읽어 Robot.config.h의 value에 할당
}
MPPI_qacc::~MPPI_qacc()
{
    cuda_data_free();
}

__device__ float sgn(float x) {

    float return_value = 0.0;
    if (x < 0) {
        return -1.0;
    } else if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}
__device__ float Clipping(float x, float min, float max){

    if(x > max){
        return max;
    }
    else if(x < min){
        return min;
    }
    else{
        return x;
    }
}
__device__ void Multiply3x3(float* Mat1, float* Mat2, float* Mat3)
{
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8

    Mat3[0] = Mat1[0] * Mat2[0] + Mat1[1] * Mat2[3] + Mat1[2] * Mat2[6];
    Mat3[1] = Mat1[0] * Mat2[1] + Mat1[1] * Mat2[4] + Mat1[2] * Mat2[7];
    Mat3[2] = Mat1[0] * Mat2[2] + Mat1[1] * Mat2[5] + Mat1[2] * Mat2[8];

    Mat3[3] = Mat1[3] * Mat2[0] + Mat1[4] * Mat2[3] + Mat1[5] * Mat2[6];
    Mat3[4] = Mat1[3] * Mat2[1] + Mat1[4] * Mat2[4] + Mat1[5] * Mat2[7];
    Mat3[5] = Mat1[3] * Mat2[2] + Mat1[4] * Mat2[5] + Mat1[5] * Mat2[8];

    Mat3[6] = Mat1[6] * Mat2[0] + Mat1[7] * Mat2[3] + Mat1[8] * Mat2[6];
    Mat3[7] = Mat1[6] * Mat2[1] + Mat1[7] * Mat2[4] + Mat1[8] * Mat2[7];
    Mat3[8] = Mat1[6] * Mat2[2] + Mat1[7] * Mat2[5] + Mat1[8] * Mat2[8];
}


__device__ __forceinline__
void MatVecMul(const float* __restrict__ A,
               const float* __restrict__ x,
               float* __restrict__ y,
               int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        float s = 0.0f;
        const float* Ar = A + r * cols;
        for (int c = 0; c < cols; ++c) {
            s += Ar[c] * x[c];
        }
        y[r] = s;
    }
}

__device__ __forceinline__
void MatMul(const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int m, int k, int n)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.f;
            #pragma unroll
            for (int t = 0; t < k; ++t) {
                s += A[i*k + t] * B[t*n + j];
            }
            C[i*n + j] = s;
        }
    }
}

__device__ __forceinline__
void MatTVecMul(const float* __restrict__ A,
                const float* __restrict__ x,
                float* __restrict__ y,
                int rows, int cols)
{
    // y[j] = sum_i A[i,j] * x[i]
    for (int c = 0; c < cols; ++c) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) {
            s += A[r * cols + c] * x[r];
        }
        y[c] = s;
    }
}

__device__ __forceinline__
float QuadraticPosCost3(const float* __restrict__ des,
                        const float* __restrict__ cur,
                        float Q)
{
    float dx = des[0] - cur[0];
    float dy = des[1] - cur[1];
    float dz = des[2] - cur[2];
    return Q * (dx*dx + dy*dy + dz*dz);
}
__device__ __forceinline__
float QuadraticCost(const float* __restrict__ e, int dim, float Q)
{
    float acc = 0.0f;
    #pragma unroll
    for(int i=0;i<dim;i++){
        acc += e[i] * e[i];
    }
    return Q * acc;
}

__device__ void Multiply3x3_T(float* Mat1, float* Mat2, float* Mat3)
{
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8

    // Rotmat^T : 0, 3, 6
    //            1, 4, 7
    //            2, 5, 8


    Mat3[0] = Mat1[0] * Mat2[0] + Mat1[1] * Mat2[1] + Mat1[2] * Mat2[2];
    Mat3[1] = Mat1[0] * Mat2[3] + Mat1[1] * Mat2[4] + Mat1[2] * Mat2[5];
    Mat3[2] = Mat1[0] * Mat2[6] + Mat1[1] * Mat2[7] + Mat1[2] * Mat2[8];

    Mat3[3] = Mat1[3] * Mat2[0] + Mat1[4] * Mat2[1] + Mat1[5] * Mat2[2];
    Mat3[4] = Mat1[3] * Mat2[3] + Mat1[4] * Mat2[4] + Mat1[5] * Mat2[5];
    Mat3[5] = Mat1[3] * Mat2[6] + Mat1[4] * Mat2[7] + Mat1[5] * Mat2[8];

    Mat3[6] = Mat1[6] * Mat2[0] + Mat1[7] * Mat2[1] + Mat1[8] * Mat2[2];
    Mat3[7] = Mat1[6] * Mat2[3] + Mat1[7] * Mat2[4] + Mat1[8] * Mat2[5];
    Mat3[8] = Mat1[6] * Mat2[6] + Mat1[7] * Mat2[7] + Mat1[8] * Mat2[8];
}

__device__ void Multiply3x3_3N(float* Mat1, float* Mat2, float* Mat3) {
    // Mat1: 3x3 행렬
    // Mat2: 3xN 행렬
    // Mat3: 3xN 결과 행렬 (Mat1 * Mat2)

    for (int i = 0; i < 3; ++i) {  // 행 인덱스 (Mat1)
        for (int j = 0; j < J_NUM; ++j) {  // 열 인덱스 (Mat2, Mat3)
            Mat3[i * J_NUM + j] = 0.0f;  // 초기화
            for (int k = 0; k < 3; ++k) {  // 곱셈 수행
                Mat3[i * J_NUM + j] += Mat1[i * 3 + k] * Mat2[k * J_NUM + j];
            }
        }
    }
}

__device__ void MatchVectorN_num(float* Vec, float Num, float* Vec_out) {
    for (int i = 0; i < J_NUM; ++i) {
        Vec_out[i] = Vec[i] * Num;
    }
}

__device__ void MatchVectorN_num_idx(float* Vec, int idx, float Num, float* Vec_out) {
    for (int i = 0; i < J_NUM; ++i) {
        Vec_out[i] = Vec[idx + i] * Num;
    }
}

__device__ void cross3x1(float Vec1_0, float Vec1_1, float Vec1_2, float Vec2_0, float Vec2_1, float Vec2_2, float* Vec3)
{
    Vec3[0] = Vec1_1*Vec2_2 - Vec1_2*Vec2_1;
    Vec3[1] = Vec1_2*Vec2_0 - Vec1_0*Vec2_2;
    Vec3[2] = Vec1_0*Vec2_1 - Vec1_1*Vec2_0;
}
__device__ void Transpose33(float* Mat1, float* Mat1_T) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Mat1_T[j * 3 + i] = Mat1[i * 3 + j];
        }
    }
}
__device__ void Multiply3x3_T_3x1(float* Mat1, float Vec1_0, float Vec1_1, float Vec1_2, float* Vec2)
{
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8
    float Mat1_T[ROT];
    Transpose33(Mat1, Mat1_T);

    Vec2[0] = Mat1_T[0] * Vec1_0 + Mat1_T[1] * Vec1_1 + Mat1_T[2] * Vec1_2;
    Vec2[1] = Mat1_T[3] * Vec1_0 + Mat1_T[4] * Vec1_1 + Mat1_T[5] * Vec1_2;
    Vec2[2] = Mat1_T[6] * Vec1_0 + Mat1_T[7] * Vec1_1 + Mat1_T[8] * Vec1_2;
}
__device__ void MatchMatrices33(float* array, int index, float* Mat2)
{
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8
    Mat2[0] = array[index + 0];
    Mat2[1] = array[index + 1];
    Mat2[2] = array[index + 2];

    Mat2[3] = array[index + 3];
    Mat2[4] = array[index + 4];
    Mat2[5] = array[index + 5];

    Mat2[6] = array[index + 6];
    Mat2[7] = array[index + 7];
    Mat2[8] = array[index + 8];

}
__device__ void MatchMatrices3N(float* array, int index, float* Mat2)
{
    for (int i=0; i<3*J_NUM; i++)
    {
        Mat2[i] = array[index + i];
    }
}
__device__ void MatchMatrices32(float* array, int index, float* Mat2)
{
    for (int i=0; i<3*J_NUM; i++)
    {
        Mat2[i] = array[index + i];
    }
}
__device__ void Multiply3x2_2x1(float* Mat1, float Vec1_0, float Vec1_1, float* Vec2)
{
    // Rotmat : 0,   1,  2,  3,  4,  5,  6,
    //          7,   8,  9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20.

    Vec2[0] = Mat1[0]  * Vec1_0 + Mat1[1]  * Vec1_1;
    Vec2[1] = Mat1[2]  * Vec1_0 + Mat1[3]  * Vec1_1;
    Vec2[2] = Mat1[4] *  Vec1_0 + Mat1[5]  * Vec1_1;
}
__device__ void Multiply3xN_Nx1(float* Mat, float* Vec, int N, float* Result) {
    for (int i = 0; i < 3; ++i) {
        Result[i] = 0.0f;
        for (int j = 0; j < N; ++j) {
            Result[i] += Mat[i * N + j] * Vec[j];
        }
    }
}
__device__ void cross3x1_and_add(float Vec1_0, float Vec1_1, float Vec1_2, float Vec2_0, float Vec2_1, float Vec2_2,
                                float Vec3_0, float Vec3_1, float Vec3_2, float Vec4_0, float Vec4_1, float Vec4_2, float* Vec5)
{
    float tempVec1[3], tempVec2[3];

    cross3x1(Vec1_0, Vec1_1, Vec1_2, Vec2_0, Vec2_1, Vec2_2, tempVec1);
    cross3x1(Vec3_0, Vec3_1, Vec3_2, Vec4_0, Vec4_1, Vec4_2, tempVec2);

    Vec5[0] = tempVec1[0] + tempVec2[0];
    Vec5[1] = tempVec1[1] + tempVec2[1];
    Vec5[2] = tempVec1[2] + tempVec2[2];

}
__device__ void skew(float Vec1_0, float Vec1_1, float Vec1_2, float* Vec2)
{
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8
    Vec2[0] = 0.0f;
    Vec2[1] = -Vec1_2;
    Vec2[2] = Vec1_1;

    Vec2[3] = Vec1_2;
    Vec2[4] = 0.0f;
    Vec2[5] = -Vec1_0;

    Vec2[6] = -Vec1_1;
    Vec2[7] = Vec1_0;
    Vec2[8] = 0.0f;
}
__device__ void Transpose32(float* Mat1, float* Mat1_T) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat1_T[j * 3 + i] = Mat1[i * 2 + j];
        }
    }
}
__device__ void Transpose3N(float* Mat1, float* Mat1_T) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat1_T[j * 3 + i] = Mat1[i * J_NUM + j];
        }
    }
}
__device__ void Multiply32_Tx33(float* Mat1, float* Mat2, float* Mat3)
{
    // Mat1   : 0, 1, 2, 3, 4, 5, 6,
    //          7, 8, 9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20
    // Mat1^T   : 0, 7, 14              // mat2 : 0, 1, 2
    //            1, 8, 15              //        3, 4, 5
    //            2, 9, 16              //        6, 7, 8
    //            3, 10, 17
    //            4, 11, 18
    //            5, 12, 19
    //            6, 13, 20

    float Mat1_T[3*2];
    Transpose32(Mat1, Mat1_T);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            Mat3[i * 3 + j] = 0;  // Mat3의 원소 초기화
            for (int k = 0; k < 3; k++) {
                Mat3[i * 3 + j] += Mat1_T[i * 3 + k] * Mat2[k * 3 + j];
            }
        }
    }

}
__device__ void MultiplyN3_T_3N(float* Mat1, float* Mat2, float* Mat3)
{
    // Mat1   : 0, 1, 2, 3, 4, 5, 6,
    //          7, 8, 9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20
    // Mat1^T   : 0, 7, 14              // mat2 : 0, 1, 2
    //            1, 8, 15              //        3, 4, 5
    //            2, 9, 16              //        6, 7, 8
    //            3, 10, 17
    //            4, 11, 18
    //            5, 12, 19
    //            6, 13, 20

    float Mat1_T[3*J_NUM];
    Transpose3N(Mat1, Mat1_T);

    for (int i = 0; i < J_NUM; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat3[i * J_NUM + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * J_NUM + j] += Mat1_T[i * 3 + k] * Mat2[k * J_NUM + j];
            }
        }
    }
}

__device__ void Multiply3N_Tx33(float* Mat1, float* Mat2, float* Mat3)
{
    // Mat1   : 0, 1, 2, 3, 4, 5, 6,
    //          7, 8, 9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20
    // Mat1^T   : 0, 7, 14              // mat2 : 0, 1, 2
    //            1, 8, 15              //        3, 4, 5
    //            2, 9, 16              //        6, 7, 8
    //            3, 10, 17
    //            4, 11, 18
    //            5, 12, 19
    //            6, 13, 20

    float Mat1_T[3*J_NUM];
    // Transpose32(Mat1, Mat1_T);
    Transpose3N(Mat1, Mat1_T);

    for (int i = 0; i < J_NUM; i++) {
        for (int j = 0; j < 3; j++) {
            Mat3[i * 3 + j] = 0;  // Mat3 초기화
            for (int k = 0; k < 3; k++) {
                Mat3[i * 3 + j] += Mat1_T[i * 3 + k] * Mat2[k * 3 + j];

            }
        }
    }

}

__device__ void Multiply3N_Tx33_with_i( float* Mat1, float* Mat2, float* Mat3)
{
    // Mat1   : 0, 1, 2, 3, 4, 5, 6,
    //          7, 8, 9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20
    // Mat1^T   : 0, 7, 14              // mat2 : 0, 1, 2
    //            1, 8, 15              //        3, 4, 5
    //            2, 9, 16              //        6, 7, 8
    //            3, 10, 17
    //            4, 11, 18
    //            5, 12, 19
    //            6, 13, 20

    float Mat1_T[3*J_NUM];
    // Transpose32(Mat1, Mat1_T);
    Transpose3N(Mat1, Mat1_T);

    for (int i = 0; i < J_NUM; i++) {
        for (int j = 0; j < 3; j++) {
            Mat3[i * 3 + j] = 0;  // Mat3 초기화
            for (int k = 0; k < 3; k++) {
                Mat3[i * 3 + j] += Mat1_T[i * 3 + k] * Mat2[k * 3 + j];
            }
        }
    }

    for (int k = 0 ; k < 3*J_NUM; k ++)
    {
        // if (isnan(Mat3[k]))
        // {
        //     printf("Mat3 isnana! \n");
        // }
    }
}

__device__ void Multiply23_32(float* Mat1, float* Mat2, float* Mat3)
{
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3[i * 2 + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * 2 + j] += Mat1[i * 3 + k] * Mat2[k * 2 + j];
            }
        }
    }
}
__device__ void Multiply23_32_num(float* Mat1, float*Mat2, float Num, float* Mat3)
{
    float Mat1_T[3*2];
    Transpose32(Mat1, Mat1_T);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3[i * 2 + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * 2 + j] += Mat1_T[i * 3 + k] * Mat2[k * 2 + j];
            }
            Mat3[i * 2 + j] *= Num; // number를 곱함
        }
    }
}

__device__ void MultiplyN3_3N(float* Mat1, float* Mat2, float* Mat3)
{
    for (int i = 0; i < J_NUM; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat3[i * J_NUM + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * J_NUM + j] += Mat1[i * 3 + k] * Mat2[k * J_NUM + j];
            }
        }
    }
}
__device__ void MultiplyN3_3N_num(float* Mat1, float*Mat2, float Num, float* Mat3)
{
    float Mat1_T[3*J_NUM];
    Transpose3N(Mat1, Mat1_T);

    for (int i = 0; i < J_NUM; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat3[i * J_NUM + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * J_NUM + j] += Mat1_T[i * 3 + k] * Mat2[k * J_NUM + j];
            }
            Mat3[i * J_NUM + j] *= Num; // number를 곱함
        }
    }
}

__device__ void AddNN(float* Mat1, float*Mat2, float* Mat3)
{
    for (int i = 0; i < J_NUM; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat3[i * J_NUM + j] = Mat1[i * J_NUM + j] + Mat2[i * J_NUM + j];
        }
    }
}

__device__ void Add3N(float* Mat1, float* Mat2, float* Mat3) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < J_NUM; ++j) {
            Mat3[i * J_NUM + j] = Mat1[i * J_NUM + j] + Mat2[i * J_NUM + j];
        }
    }
}

__device__ void ADD_Total(float M_[J_NUM][J_NUM * J_NUM], float*Mat2)
{
    for (int j = 0; j < J_NUM; ++j) {
        for (int i = 0; i < J_NUM * J_NUM; ++i) {
            Mat2[i] += M_[j][i];
        }
    }
}

__device__ void Add22(float* Mat1, float*Mat2, float* Mat3)
{
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3[i * 2 + j] = Mat1[i * 2 + j] + Mat2[i * 2 + j];
        }
    }
}
__device__ void Multiply3x3_3x2(float* Mat1, float* Mat2, float* Result)
{
    // Mat1 : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8
    // Mat2 : 0,   1,  2,  3,  4,  5,  6,
    //          7,   8,  9, 10, 11, 12, 13,
    //          14, 15, 16, 17, 18, 19, 20.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            Result[i * 2 + j] = 0;  // 결과 행렬의 원소 초기화
            for (int k = 0; k < 3; ++k) {
                Result[i * 2 + j] += Mat1[i * 3 + k] * Mat2[k * 2 + j];
            }
        }
    }
}
__device__ void Add32(float* Mat1, float*Mat2, float* Mat3)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3[i * 2 + j] = Mat1[i * 2 + j] + Mat2[i * 2 + j];
        }
    }
}
__device__ void Multiply23_T_32(float* Mat1, float* Mat2, float* Mat3)
{
    float Mat1_T[3*2];
    Transpose32(Mat1, Mat1_T);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            Mat3[i * 2 + j] = 0;
            for (int k = 0; k < 3; ++k) {
                Mat3[i * 2 + j] += Mat1_T[i * 3 + k] * Mat2[k * 2 + j];
            }
        }
    }
}
__device__ void MatchVector2_num(float* Mat, float Num, float* Vec)
{
    for (int j = 0; j < 2; ++j) {
        Vec[j] = Mat[2 * 2 + j] * Num;
    }
}


__device__ float clamp(float val, float min, float max){
     return fmaxf(min, fminf(val, max));
}

__device__ void subvec3x1(float *vec1, float *vec2, float *output){
    output[0] = vec1[0] - vec2[0];
    output[1] = vec1[1] - vec2[1];
    output[2] = vec1[2] - vec2[2];
}
__device__ float dotvec3x1(float *vec1, float *vec2) {
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]; 
}

__device__ void scaleAddvec3x1(const float *vec, float scalar, float *output) {
    output[0] = vec[0] * scalar;
    output[1] = vec[1] * scalar;
    output[2] = vec[2] * scalar;
}

__device__ void addvec3x1(float *vec1, float *vec2, float *output){
    output[0] = vec1[0] + vec2[0];
    output[1] = vec1[1] + vec2[1];
    output[2] = vec1[2] + vec2[2];
}

__device__ __forceinline__
float MinimumDistance(float *start1, float *end1, float *start2, float *end2){
    float d1[3], d2[3], r[3];

    float tmp[3] = {0.0};
    float tmp2[3] = {0.0};
    float tmp3[3] = {0.0};
    float tmp4[3] = {0.0};

    float t =0.0; 
    float u = 0.0;

    subvec3x1(end1, start1, d1);
    subvec3x1(end2, start2, d2);
    subvec3x1(start2, start1, r);

    float D1 = dotvec3x1(d1, d1);
    float D2 = dotvec3x1(d2, d2);
    float R  = dotvec3x1(d1, d2);
    float S1 = dotvec3x1(d1, r);
    float S2 = dotvec3x1(d2, r);
    float denominator =  D1 * D2 - R * R;
    
    // Step 1 IF parallel ?
    if (D1 == 0 && D2 > 0) { // AB degenerates into a point
        t = 0;
        u = clamp(-S2 / D2, 0.0, 1.0);
        scaleAddvec3x1(d2, u, tmp);
        addvec3x1(start2, tmp, tmp2);
        subvec3x1(start1, tmp2, tmp3);
        return dotvec3x1(tmp3, tmp3);
    }
    if (D2 == 0 && D1 > 0) { // CD degenerates into a point
        u = 0;
        t = clamp(S1 / D1, 0.0, 1.0);

        scaleAddvec3x1(d1, t, tmp); // t*d1
        addvec3x1(start1, tmp, tmp2); // start1 + t*d1
        subvec3x1(start2, tmp2, tmp3); // start2 - start1 + t*d1
        return dotvec3x1(tmp3, tmp3);
    }
    if (D1 == 0 && D2 == 0) { // Both segments degenerate into points
        t = 0;
        u = 0;
        return dotvec3x1(r,r);
    }
    if (denominator == 0) { // Segments are parallel
        t = 0;
        u = -S2 / D2;
        if (u < 0 || u > 1) {
            u = clamp(u, 0.0, 1.0); // Clamp u
            t = clamp((u * R + S1) / D1, 0.0, 1.0); // Recompute t (Step 4)
        }

        scaleAddvec3x1(d2, u, tmp); // d2 *u
        addvec3x1(start2, tmp, tmp2); // start2  + d2*u

        scaleAddvec3x1(d1, t, tmp); // d1 *t
        addvec3x1(start1, tmp, tmp3); // start1  + d1*t

        subvec3x1(tmp3, tmp2, tmp4); // start1  + d1*t - (start2  + d2*u)

        return dotvec3x1(tmp4,tmp4);
    }
    // Step 2 eq. (11) --> t
    t = (S1 * D2 - S2 * R) / denominator;
    t = clamp(t, 0.0, 1.0);
    // Step 3 eq. (10) --> u
    u = (t*R - S2) / D2;
    u = clamp(u, 0.0, 1.0);
    // Step 4 eq. (10) --> t
    t = (u*R + S1)/D1;
    t = clamp(t, 0.0, 1.0);

    // Step 5: Compute the actual minimum distance
    scaleAddvec3x1(d1, t, tmp); // d1 *t
    addvec3x1(start1, tmp, tmp2); // start1  + d1*t

    scaleAddvec3x1(d2, u, tmp); // d2 *u
    addvec3x1(start2, tmp, tmp3); // start2  + d2*u

    subvec3x1(tmp2, tmp3, tmp4); // start1  + d1*t - (start2  + d2*u)
    return dotvec3x1(tmp4,tmp4);

}

__device__ __forceinline__
float CollisionCheck(float *link3, float *link4, float *link5, float *link7, float *linke7)
{
    float link0[3] = {0.0, 0.0, 0.375};
    float link2[3] = {0.0, 0.0, 0.708}; 
    float link_dist[3];
    // Tube 1 - Tube 3
    link_dist[0] = sqrt(MinimumDistance(link0, link2, link4, link5));
    // Tube 1 - Tube 4
    link_dist[1] = sqrt(MinimumDistance(link0, link2, link7, linke7));
    // Tube 2 - Tube 4
    link_dist[2] = sqrt(MinimumDistance(link2, link3, link7, linke7));

    float distance_threshold[3] = {0.20, 0.20, 0.20};
    float cost_col = 0.0f;
    for (int i = 0; i < 3; i++){
        // If predicted link dist is less than threshold, add collision cost
        if (link_dist[i] < distance_threshold[i]){
            cost_col += 1000000.0f; // collision
        }
    }
    return cost_col;
}

__device__ __forceinline__
void Rt_mul_R(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ R_out){
    // A^T * B = (3x3) * (3x3)
    // row-major: idx = r*3 + c
    #pragma unroll
    for(int r=0;r<3;r++){
        #pragma unroll
        for(int c=0;c<3;c++){
            float acc = 0.f;
            // A^T has rows = cols of A
            acc += A[0*3 + r]*B[0*3 + c];
            acc += A[1*3 + r]*B[1*3 + c];
            acc += A[2*3 + r]*B[2*3 + c];
            R_out[r*3 + c] = acc;
        }
    }
}

__device__ __forceinline__
float trace3(const float* __restrict__ R){
    return R[0*3+0] + R[1*3+1] + R[2*3+2];
}

// vee(Skew) : 3x3 skew-symmetric -> R^3 (row-major)
__device__ __forceinline__
void vee3(const float* __restrict__ S, float* __restrict__ v){
    // v = [S32, S13, S21] with 1/2 scaling outside if needed
    v[0] = S[2*3 + 1]; // S(3,2)
    v[1] = S[0*3 + 2]; // S(1,3)
    v[2] = S[1*3 + 0]; // S(2,1)
}

// so3 log: w = log(R_err)^\vee  (row-major, numerically robust)
__device__ __forceinline__
void so3_log(const float* __restrict__ R_err, float* __restrict__ w){
    // cos(theta) = (tr(R)-1)/2
    float tr = trace3(R_err);
    float c = fmaxf(-1.0f, fminf(1.0f, 0.5f*(tr - 1.0f)));
    float theta = acosf(c);

    // near zero: use first-order approx: log(R)^\vee ≈ 0.5 * vee(R - R^T)
    const float eps_small = 1e-5f;
    if(theta < eps_small){
        float S[9];
        // S = 0.5*(R - R^T)
        #pragma unroll
        for(int r=0;r<3;r++){
            #pragma unroll
            for(int c=0;c<3;c++){
                S[r*3+c] = 0.5f*(R_err[r*3+c] - R_err[c*3+r]);
            }
        }
        // vee(S)
        vee3(S, w);
        return;
    }

    // near pi: use robust axis extraction
    const float eps_pi = 1e-3f; // ~0.057 deg
    if(fabsf(M_PI - theta) < eps_pi){
        // From diag elements
        float xx = (R_err[0]-c)/ (1.0f - c); // safer than (1+R[0,0])
        float yy = (R_err[4]-c)/ (1.0f - c);
        float zz = (R_err[8]-c)/ (1.0f - c);
        // pick the largest for stability
        int idx = 0; float m = xx;
        if(yy > m){ m=yy; idx=1; }
        if(zz > m){ m=zz; idx=2; }

        float axis[3] = {0.f,0.f,0.f};
        if(idx==0){
            axis[0] = sqrtf(fmaxf(0.f, xx));
            axis[1] = (R_err[1] + R_err[3]) / (2.0f*axis[0] + 1e-8f);
            axis[2] = (R_err[2] + R_err[6]) / (2.0f*axis[0] + 1e-8f);
        }else if(idx==1){
            axis[1] = sqrtf(fmaxf(0.f, yy));
            axis[0] = (R_err[1] + R_err[3]) / (2.0f*axis[1] + 1e-8f);
            axis[2] = (R_err[5] + R_err[7]) / (2.0f*axis[1] + 1e-8f);
        }else{
            axis[2] = sqrtf(fmaxf(0.f, zz));
            axis[0] = (R_err[2] + R_err[6]) / (2.0f*axis[2] + 1e-8f);
            axis[1] = (R_err[5] + R_err[7]) / (2.0f*axis[2] + 1e-8f);
        }
        // normalize just in case
        float n = sqrtf(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]) + 1e-12f;
        w[0] = theta * axis[0]/n;
        w[1] = theta * axis[1]/n;
        w[2] = theta * axis[2]/n;
        return;
    }

    // general case: w = (theta/(2 sin theta)) * vee(R - R^T)
    float s = sinf(theta);
    float alpha = 0.5f*theta/(s + 1e-12f);
    float S[9];
    #pragma unroll
    for(int r=0;r<3;r++){
        #pragma unroll
        for(int c=0;c<3;c++){
            S[r*3+c] = alpha*(R_err[r*3+c] - R_err[c*3+r]);
        }
    }
    vee3(S, w);
}

__device__ void GetBodyRotationAngle(float* RotMat, float* BodyAngle)
{
    double roll, pitch, yaw;
    double threshold = 0.001;
    // Rotmat : 0, 1, 2
    //          3, 4, 5
    //          6, 7, 8
    pitch = -asin(RotMat[6]);
    if(RotMat[6] > 1.0 - threshold && RotMat[6] < 1.0 + threshold)
    {   //Gimbal lock, pitch = -90deg == cos(pitch) = 0 : singularity is occured
        roll = atan2(-RotMat[1], -RotMat[2]);
        yaw = 0.0;
    }
    else if( RotMat[6] < -1.0 + threshold && RotMat[6] > -1.0 + threshold)
    {   //Gimbal lock, pitch = 90deg
        roll = atan2(RotMat[1], RotMat[2]);
        yaw = 0.0;
    }
    else{  //General solution
        roll = atan2(RotMat[7], RotMat[8]);
        yaw = atan2(RotMat[3], RotMat[0]);
    }
    BodyAngle[0] = roll;
    BodyAngle[1] = pitch;
    BodyAngle[2] = yaw;
}
__device__ void normalizeVector3(const float* axis, float* out) {
    double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (norm == 0) {
        out[0] = out[1] = out[2] = 0.0;  // 예외 처리
        return;
    }
    out[0] = axis[0] / norm;
    out[1] = axis[1] / norm;
    out[2] = axis[2] / norm;
}
__device__ void ComputeTransformationMatrix(
    const float* link_pos,   // 입력: 링크 위치 (길이 3)
    const float* Rotmat,     // 입력: 링크 기준 회전 (길이 9, row-major)
    float        theta,      // 입력: 관절각
    const float* _rho,       // 입력: CoM offset (길이 3)
    const float* axis,       // 입력: 회전축 (길이 3, 정규화 전)
    // 출력
    float* B2Link_pos,       // 출력: base→link 위치 (길이 3)
    float* B2Link_Rot,       // 출력: base→link 회전 (길이 9)
    float* Pos_CoM           // 출력: CoM 위치 (길이 3)
)
{
    float s = sinf(theta);
    float c = cosf(theta);

    // Rodrigues' rotation formula
    float norm_axis[3] = {0.0f, 0.0f, 0.0f};
    normalizeVector3(axis, norm_axis);
    float ax = norm_axis[0];
    float ay = norm_axis[1];
    float az = norm_axis[2];

    float v = 1.0f - c;

    float tmp_R[9];
    // row 0
    tmp_R[0] = c + ax * ax * v;
    tmp_R[1] = ax * ay * v - az * s;
    tmp_R[2] = ax * az * v + ay * s;

    // row 1
    tmp_R[3] = ay * ax * v + az * s;
    tmp_R[4] = c + ay * ay * v;
    tmp_R[5] = ay * az * v - ax * s;

    // row 2
    tmp_R[6] = az * ax * v - ay * s;
    tmp_R[7] = az * ay * v + ax * s;
    tmp_R[8] = c + az * az * v;

    // Rotmat * tmp_R → B2Link_Rot
    MatMul(Rotmat, tmp_R, B2Link_Rot, 3, 3, 3);

    // 위치 그대로 복사
    B2Link_pos[0] = link_pos[0];
    B2Link_pos[1] = link_pos[1];
    B2Link_pos[2] = link_pos[2];

    // CoM 위치
    Pos_CoM[0] = B2Link_pos[0] + B2Link_Rot[0] * _rho[0] + B2Link_Rot[1] * _rho[1] + B2Link_Rot[2] * _rho[2];
    Pos_CoM[1] = B2Link_pos[1] + B2Link_Rot[3] * _rho[0] + B2Link_Rot[4] * _rho[1] + B2Link_Rot[5] * _rho[2];
    Pos_CoM[2] = B2Link_pos[2] + B2Link_Rot[6] * _rho[0] + B2Link_Rot[7] * _rho[1] + B2Link_Rot[8] * _rho[2];
}



__device__ void ComputeEachLinkPose(float* Rot_B2Lnk_parent, float* rpos_lnk_parent,                            // Input
                                    float* Pos_Link_child , float* Rot_Link_child, float* Pos_CoM_child,        // Input
                                    float* Rot_B2Lnk_child, float* rpos_lnk_child, float* rpos_lnk_CoM_child)   // output
{
    Multiply3x3(Rot_B2Lnk_parent, Rot_Link_child, Rot_B2Lnk_child);

    rpos_lnk_child[0] = rpos_lnk_parent[0] + Rot_B2Lnk_parent[0] * Pos_Link_child[0] + Rot_B2Lnk_parent[1] * Pos_Link_child[1] + Rot_B2Lnk_parent[2] * Pos_Link_child[2];
    rpos_lnk_child[1] = rpos_lnk_parent[1] + Rot_B2Lnk_parent[3] * Pos_Link_child[0] + Rot_B2Lnk_parent[4] * Pos_Link_child[1] + Rot_B2Lnk_parent[5] * Pos_Link_child[2];
    rpos_lnk_child[2] = rpos_lnk_parent[2] + Rot_B2Lnk_parent[6] * Pos_Link_child[0] + Rot_B2Lnk_parent[7] * Pos_Link_child[1] + Rot_B2Lnk_parent[8] * Pos_Link_child[2];

    rpos_lnk_CoM_child[0] = rpos_lnk_parent[0] + Rot_B2Lnk_parent[0] * Pos_CoM_child[0] + Rot_B2Lnk_parent[1] * Pos_CoM_child[1] + Rot_B2Lnk_parent[2] * Pos_CoM_child[2];
    rpos_lnk_CoM_child[1] = rpos_lnk_parent[1] + Rot_B2Lnk_parent[3] * Pos_CoM_child[0] + Rot_B2Lnk_parent[4] * Pos_CoM_child[1] + Rot_B2Lnk_parent[5] * Pos_CoM_child[2];
    rpos_lnk_CoM_child[2] = rpos_lnk_parent[2] + Rot_B2Lnk_parent[6] * Pos_CoM_child[0] + Rot_B2Lnk_parent[7] * Pos_CoM_child[1] + Rot_B2Lnk_parent[8] * Pos_CoM_child[2];

}
__device__ void GetPose(const float* B2Link_pos,
                        const float* B2Link_Rot,
                        const float* pos_offset,
                        const float* rot_offset,
                        float* EE_pos,
                        float* EE_rot,
                        float* link3, float* link4, float* link5, float* link7, float* link7e)   
{
    // 1) EE 위치:  r_A = B2Link_pos + R * pos_offset
    float offset_world[3];
    MatVecMul(B2Link_Rot, pos_offset, offset_world, 3, 3);

    EE_pos[0] = B2Link_pos[0] + offset_world[0];
    EE_pos[1] = B2Link_pos[1] + offset_world[1];
    EE_pos[2] = B2Link_pos[2] + offset_world[2];

    // 2) EE 회전: R_bA = B2Link_Rot * rot_offset
    // If rot_offset is identity, EE_rot = B2Link_Rot
    MatMul(B2Link_Rot, rot_offset, EE_rot, 3, 3, 3);
}
__device__ void GetLinkPos(const float* B2Link_pos,
                        float* linkpos)   
{
    // 1) EE 위치:  r_A = B2Link_pos + R * pos_offset
    linkpos[0] = B2Link_pos[0];
    linkpos[1] = B2Link_pos[1];
    linkpos[2] = B2Link_pos[2];
}
__device__ void GetLinkPos_offset(const float* B2Link_pos,
                                const float* B2Link_Rot,
                                const float* pos_offset,
                                float* linkpos)   
{
    // 1) EE 위치:  r_A = B2Link_pos + R * pos_offset
    float offset_link[3];
    MatVecMul(B2Link_Rot, pos_offset, offset_link, 3, 3);

    linkpos[0] = B2Link_pos[0] + offset_link[0];
    linkpos[1] = B2Link_pos[1] + offset_link[1];
    linkpos[2] = B2Link_pos[2] + offset_link[2];
}

// __device__ void GetPose(float* B2Link_pos, float* B2Link_Rot,
//      float* pos_offset, float* rot_offset, float* EE_pos, float* EE_rot){

//     float r_A[3] = {0.0};
//     float R_bA[9] = {0.0};

//     // float pos_offset[3];        float rot_offset[9];
//     // pos_offset[0] = 0.2;    pos_offset[1] = 0.0;    pos_offset[2] = 0.0;    //  Position

//     // rot_offset[0] = 1.0;    rot_offset[1] = 0.0;    rot_offset[2] = 0.0;    //  Rotation Matrix
//     // rot_offset[3] = 0.0;    rot_offset[4] = 1.0;    rot_offset[5] = 0.0;
//     // rot_offset[6] = 0.0;    rot_offset[7] = 0.0;    rot_offset[8] = 1.0;

//     r_A[0] = B2Link_pos[0];
//     r_A[1] = B2Link_pos[1];
//     r_A[2] = B2Link_pos[2];


//     r_A[0] += (B2Link_Rot[0] * pos_offset[0] + B2Link_Rot[1] * pos_offset[1] + B2Link_Rot[2] * pos_offset[2]);
//     r_A[1] += (B2Link_Rot[3] * pos_offset[0] + B2Link_Rot[4] * pos_offset[1] + B2Link_Rot[5] * pos_offset[2]);
//     r_A[2] += (B2Link_Rot[6] * pos_offset[0] + B2Link_Rot[7] * pos_offset[1] + B2Link_Rot[8] * pos_offset[2]);


//     R_bA[0] = B2Link_Rot[0] * rot_offset[0] + B2Link_Rot[1] * rot_offset[3] + B2Link_Rot[2] * rot_offset[6];
//     R_bA[1] = B2Link_Rot[0] * rot_offset[1] + B2Link_Rot[1] * rot_offset[4] + B2Link_Rot[2] * rot_offset[7];
//     R_bA[2] = B2Link_Rot[0] * rot_offset[2] + B2Link_Rot[1] * rot_offset[5] + B2Link_Rot[2] * rot_offset[8];

//     R_bA[3] = B2Link_Rot[3] * rot_offset[0] + B2Link_Rot[4] * rot_offset[3] + B2Link_Rot[5] * rot_offset[6];
//     R_bA[4] = B2Link_Rot[3] * rot_offset[1] + B2Link_Rot[4] * rot_offset[4] + B2Link_Rot[5] * rot_offset[7];
//     R_bA[5] = B2Link_Rot[3] * rot_offset[2] + B2Link_Rot[4] * rot_offset[5] + B2Link_Rot[5] * rot_offset[8];

//     R_bA[6] = B2Link_Rot[6] * rot_offset[0] + B2Link_Rot[7] * rot_offset[3] + B2Link_Rot[8] * rot_offset[6];
//     R_bA[7] = B2Link_Rot[6] * rot_offset[1] + B2Link_Rot[7] * rot_offset[4] + B2Link_Rot[8] * rot_offset[7];
//     R_bA[8] = B2Link_Rot[6] * rot_offset[2] + B2Link_Rot[7] * rot_offset[5] + B2Link_Rot[8] * rot_offset[8];

//     EE_pos[0] = 0.0;
//     EE_pos[1] = 0.0;
//     EE_pos[2] = 0.0;

//     for(int i=0; i<9; i++){
//         EE_rot[i] = 0.0;
//     }

//     EE_pos[0] += 1.0 * r_A[0] + 0.0 * r_A[1] + 0.0 * r_A[2];
//     EE_pos[1] += 0.0 * r_A[0] + 1.0 * r_A[1] + 0.0 * r_A[2];
//     EE_pos[2] += 0.0 * r_A[0] + 0.0 * r_A[1] + 1.0 * r_A[2];

//     EE_rot[0] += 1.0 * R_bA[0] + 0.0 * R_bA[3] + 0.0 * R_bA[6];
//     EE_rot[1] += 1.0 * R_bA[1] + 0.0 * R_bA[4] + 0.0 * R_bA[7];
//     EE_rot[2] += 1.0 * R_bA[2] + 0.0 * R_bA[5] + 0.0 * R_bA[8];

//     EE_rot[3] += 0.0 * R_bA[0] + 1.0 * R_bA[3] + 0.0 * R_bA[6];
//     EE_rot[4] += 0.0 * R_bA[1] + 1.0 * R_bA[4] + 0.0 * R_bA[7];
//     EE_rot[5] += 0.0 * R_bA[2] + 1.0 * R_bA[5] + 0.0 * R_bA[8];

//     EE_rot[6] += 0.0 * R_bA[0] + 0.0 * R_bA[3] + 1.0 * R_bA[6];
//     EE_rot[7] += 0.0 * R_bA[1] + 0.0 * R_bA[4] + 1.0 * R_bA[7];
//     EE_rot[8] += 0.0 * R_bA[2] + 0.0 * R_bA[5] + 1.0 * R_bA[8];
// }


__device__ void Get_Jacob(float* B2Link_pos, float* B2Link_Rot, float* B2Link_COM, float* Jp_link_stacked, float* Jr_link_stacked,
                            float* joint_axis,
                            float* EE_pos, float* _Jr, float* _Jp, float* mass, float* Tot_mass,
                            float* J_COM_stacked, float* J_COM){
    // Eigen::Vector3d* _poslink[] = {&_poslink1, &_poslink2, &_poslink3, &_poslink4, &_poslink5, &_poslink6, &_poslink7};
    int d_JDOF = J_NUM;

    float _Jr_link_B1[3*J_NUM] = {0.0};       float _Jp_link_B1[3*J_NUM] = {0.0};
    float _Jr_link_B2[3*J_NUM] = {0.0};       float _Jp_link_B2[3*J_NUM] = {0.0};
    float _Jr_link_B3[3*J_NUM] = {0.0};       float _Jp_link_B3[3*J_NUM] = {0.0};
    float _Jr_link_B4[3*J_NUM] = {0.0};       float _Jp_link_B4[3*J_NUM] = {0.0};
    float _Jr_link_B5[3*J_NUM] = {0.0};       float _Jp_link_B5[3*J_NUM] = {0.0};
    float _Jr_link_B6[3*J_NUM] = {0.0};       float _Jp_link_B6[3*J_NUM] = {0.0};
    float _Jr_link_B7[3*J_NUM] = {0.0};       float _Jp_link_B7[3*J_NUM] = {0.0};

    float tempVec3_imsi[3] = {0.0};

    float tempVec3_1[3] = {0.0};
    float tempVec3_2[3] = {0.0};
    float tempVec3_3[3] = {0.0};
    float tempVec3_4[3] = {0.0};
    float tempVec3_5[3] = {0.0};
    float tempVec3_6[3] = {0.0};
    float tempVec3_7[3] = {0.0};

    float* _Jr_link_B[] = {_Jr_link_B1, _Jr_link_B2, _Jr_link_B3, _Jr_link_B4, _Jr_link_B5, _Jr_link_B6, _Jr_link_B7  };
    float* _Jp_link_B[] = {_Jp_link_B1, _Jp_link_B2, _Jp_link_B3, _Jp_link_B4, _Jp_link_B5, _Jp_link_B6, _Jp_link_B7  };

    float* tempVec3[] = {tempVec3_1, tempVec3_2, tempVec3_3, tempVec3_4, tempVec3_5, tempVec3_6, tempVec3_7};


    for (int i = 0; i < d_JDOF; ++i) {
        joint_axis[3*i + 0] = B2Link_Rot[9*i + 2];
        joint_axis[3*i + 1] = B2Link_Rot[9*i + 5];
        joint_axis[3*i + 2] = B2Link_Rot[9*i + 8];

        // printf("Joint axis [%i] %f ,%f ,%f \n", i+1, joint_axis[3*i + 0], joint_axis[3*i + 1], joint_axis[3*i + 2]);
    }


    // original
    for (int i = 0; i < d_JDOF; ++i) {  // 열 인덱스 0과 1
        for (int k = 0; k <=i; k ++) {
            for (int j = 0; j < 3; ++j) // 행 인덱스 0, 1, 2
            {
                _Jr_link_B[i][j * J_NUM + k] = joint_axis[j + 3 * k];
            }
        }

    }

    for (int j = 0; j < d_JDOF; j ++) // link number
    {
        for (int i = 0; i <= j; i ++ )
        {
            // j = 0, i = 0
            cross3x1(_Jr_link_B[j][J_NUM * 0 + i], _Jr_link_B[j][J_NUM * 1 + i], _Jr_link_B[j][J_NUM * 2 + i],
                B2Link_pos[3*j + 0] - B2Link_pos[3*i + 0],
                B2Link_pos[3*j + 1] - B2Link_pos[3*i + 1],
                B2Link_pos[3*j + 2] - B2Link_pos[3*i + 2], tempVec3[j]);
            _Jp_link_B[j][J_NUM * 0 + i] = tempVec3[j][0];
            _Jp_link_B[j][J_NUM * 1 + i] = tempVec3[j][1];
            _Jp_link_B[j][J_NUM * 2 + i] = tempVec3[j][2];
        }
        // printf("_Jp_link_B [%i] %f ,%f ,%f,%f ,%f ,%f ,%f \n", j+1, _Jp_link_B[j][0* J_NUM + 0], _Jp_link_B[j][0* J_NUM + 1], _Jp_link_B[j][0* J_NUM + 2], _Jp_link_B[j][0* J_NUM + 3], _Jp_link_B[j][0*J_NUM + 4], _Jp_link_B[j][0*J_NUM + 5], _Jp_link_B[j][0*J_NUM + 6]);
        // printf("                %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _Jp_link_B[j][1* J_NUM + 0], _Jp_link_B[j][1* J_NUM + 1], _Jp_link_B[j][1* J_NUM + 2], _Jp_link_B[j][1* J_NUM + 3], _Jp_link_B[j][1*J_NUM + 4], _Jp_link_B[j][1*J_NUM + 5], _Jp_link_B[j][1*J_NUM + 6]);
        // printf("                %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _Jp_link_B[j][2* J_NUM + 0], _Jp_link_B[j][2* J_NUM + 1], _Jp_link_B[j][2* J_NUM + 2], _Jp_link_B[j][2* J_NUM + 3], _Jp_link_B[j][2*J_NUM + 4], _Jp_link_B[j][2*J_NUM + 5], _Jp_link_B[j][2*J_NUM + 6]);
    }

    for (int j = 0; j < d_JDOF; j ++)
    {
        for(int i=0; i<3*d_JDOF; i++){
            Jp_link_stacked[3*J_NUM*j + i] = _Jp_link_B[j][i];
            Jr_link_stacked[3*J_NUM*j + i] = _Jr_link_B[j][i];
        }
    }   //checked


    ///////////////////////////     compute Linear & Angular jacobian     ///////////////////////////////
    float rpos_A[3] = {0.0};
    float tempMat[3*J_NUM] = {0.0};

    float p_B[3] ={0.0};
    float R_B[9] ={0.0};
    R_B[0] = 1.0;   // setIdentity
    R_B[4] = 1.0;
    R_B[8] = 1.0;

    // Set rpos_A
    Multiply3x3_T_3x1(R_B, EE_pos[0] - p_B[0], EE_pos[1] - p_B[1], EE_pos[2] - p_B[2], rpos_A);


    for (int i = 0; i < d_JDOF; ++i) {
        for (int j = 0; j < 3; ++j) {
            _Jr[j * J_NUM + i] = joint_axis[i * 3 + j];
        }
    } //checked
    // printf("_Jr %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jr[0* J_NUM + 0], _Jr[0* J_NUM + 1], _Jr[0* J_NUM + 2], _Jr[0* J_NUM + 3], _Jr[0* J_NUM + 4], _Jr[0* J_NUM + 5], _Jr[0* J_NUM + 6]);
    // printf("    %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jr[1* J_NUM + 0], _Jr[1* J_NUM + 1], _Jr[1* J_NUM + 2], _Jr[1* J_NUM + 3], _Jr[1* J_NUM + 4], _Jr[1* J_NUM + 5], _Jr[1* J_NUM + 6]);
    // printf("    %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jr[2* J_NUM + 0], _Jr[2* J_NUM + 1], _Jr[2* J_NUM + 2], _Jr[2* J_NUM + 3], _Jr[2* J_NUM + 4], _Jr[2* J_NUM + 5], _Jr[2* J_NUM + 6]);

    for (int j = 0; j < d_JDOF; j ++)
    {
        for (int i = 0; i <= j ; i ++)
        {
            cross3x1(_Jr[J_NUM * 0 + j], _Jr[J_NUM * 1 + j], _Jr[J_NUM * 2 + j],
                rpos_A[0] - B2Link_pos[3*j + 0],
                rpos_A[1] - B2Link_pos[3*j + 1],
                rpos_A[2] - B2Link_pos[3*j + 2], tempVec3[j]);
            _Jp[J_NUM * 0 + j] = tempVec3[j][0];
            _Jp[J_NUM * 1 + j] = tempVec3[j][1];
            _Jp[J_NUM * 2 + j] = tempVec3[j][2];
        }
    } //checked
    // printf("_Jp %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jp[0* J_NUM + 0], _Jp[0* J_NUM + 1], _Jp[0* J_NUM + 2], _Jp[0* J_NUM + 3], _Jp[0* J_NUM + 4], _Jp[0* J_NUM + 5], _Jp[0* J_NUM + 6]);
    // printf("    %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jp[1* J_NUM + 0], _Jp[1* J_NUM + 1], _Jp[1* J_NUM + 2], _Jp[1* J_NUM + 3], _Jp[1* J_NUM + 4], _Jp[1* J_NUM + 5], _Jp[1* J_NUM + 6]);
    // printf("    %f ,%f ,%f,%f ,%f ,%f ,%f \n", _Jp[2* J_NUM + 0], _Jp[2* J_NUM + 1], _Jp[2* J_NUM + 2], _Jp[2* J_NUM + 3], _Jp[2* J_NUM + 4], _Jp[2* J_NUM + 5], _Jp[2* J_NUM + 6]);


    float _J_link_CoM1[3*J_NUM] = {0.0};      float _J_CoM1[3*J_NUM] = {0.0};
    float _J_link_CoM2[3*J_NUM] = {0.0};      float _J_CoM2[3*J_NUM] = {0.0};
    float _J_link_CoM3[3*J_NUM] = {0.0};      float _J_CoM3[3*J_NUM] = {0.0};
    float _J_link_CoM4[3*J_NUM] = {0.0};      float _J_CoM4[3*J_NUM] = {0.0};
    float _J_link_CoM5[3*J_NUM] = {0.0};      float _J_CoM5[3*J_NUM] = {0.0};
    float _J_link_CoM6[3*J_NUM] = {0.0};      float _J_CoM6[3*J_NUM] = {0.0};
    float _J_link_CoM7[3*J_NUM] = {0.0};      float _J_CoM7[3*J_NUM] = {0.0};

    float* _J_link_CoM[] = {_J_link_CoM1, _J_link_CoM2, _J_link_CoM3, _J_link_CoM4, _J_link_CoM5, _J_link_CoM6, _J_link_CoM7};
    float* _J_CoM[] = {_J_CoM1, _J_CoM2, _J_CoM3, _J_CoM4, _J_CoM5, _J_CoM6, _J_CoM7};
    float _J_CoMimsi[3*J_NUM] = {0.0};

    //여기부터
    for (int j = 0; j < d_JDOF; j ++)
    {
        for (int i = 0; i <= j; i ++ )
        {
            cross3x1(_Jr_link_B[j][J_NUM * 0 + i], _Jr_link_B[j][J_NUM * 1 + i], _Jr_link_B[j][J_NUM * 2 + i],
                B2Link_COM[3*j + 0] - B2Link_pos[3*i + 0],
                B2Link_COM[3*j + 1] - B2Link_pos[3*i + 1],
                B2Link_COM[3*j + 2] - B2Link_pos[3*i + 2], tempVec3[j]);

            _J_link_CoM[j][J_NUM * 0 + i] = tempVec3[j][0];
            _J_link_CoM[j][J_NUM * 1 + i] = tempVec3[j][1];
            _J_link_CoM[j][J_NUM * 2 + i] = tempVec3[j][2];
        }
        // printf("_J_link_CoM [%i] %f ,%f ,%f,%f ,%f ,%f ,%f \n", j+1, _J_link_CoM[j][0* J_NUM + 0], _J_link_CoM[j][0* J_NUM + 1], _J_link_CoM[j][0* J_NUM + 2], _J_link_CoM[j][0* J_NUM + 3], _J_link_CoM[j][0*J_NUM + 4], _J_link_CoM[j][0*J_NUM + 5], _J_link_CoM[j][0*J_NUM + 6]);
        // printf("                 %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _J_link_CoM[j][1* J_NUM + 0], _J_link_CoM[j][1* J_NUM + 1], _J_link_CoM[j][1* J_NUM + 2], _J_link_CoM[j][1* J_NUM + 3], _J_link_CoM[j][1*J_NUM + 4], _J_link_CoM[j][1*J_NUM + 5], _J_link_CoM[j][1*J_NUM + 6]);
        // printf("                 %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _J_link_CoM[j][2* J_NUM + 0], _J_link_CoM[j][2* J_NUM + 1], _J_link_CoM[j][2* J_NUM + 2], _J_link_CoM[j][2* J_NUM + 3], _J_link_CoM[j][2*J_NUM + 4], _J_link_CoM[j][2*J_NUM + 5], _J_link_CoM[j][2*J_NUM + 6]);
    }


    for(int i=0; i<3*d_JDOF; i++)
    {
        for (int j = 0; j < d_JDOF; j ++)
        {
            J_COM_stacked[3*J_NUM*j + i] = _J_link_CoM[j][i];
            _J_CoM[j][i] = mass[j] * _J_link_CoM[j][i];
        }
    }   //checked
    // for(int j=0; j<7; j++){
    //     printf("_J_CoM [%i] %f ,%f ,%f,%f ,%f ,%f ,%f \n", j+1, _J_CoM[j][0* J_NUM + 0], _J_CoM[j][0* J_NUM + 1], _J_CoM[j][0* J_NUM + 2], _J_CoM[j][0* J_NUM + 3], _J_CoM[j][0*J_NUM + 4], _J_CoM[j][0*J_NUM + 5], _J_CoM[j][0*J_NUM + 6]);
    //     printf("            %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _J_CoM[j][1* J_NUM + 0], _J_CoM[j][1* J_NUM + 1], _J_CoM[j][1* J_NUM + 2], _J_CoM[j][1* J_NUM + 3], _J_CoM[j][1*J_NUM + 4], _J_CoM[j][1*J_NUM + 5], _J_CoM[j][1*J_NUM + 6]);
    //     printf("            %f ,%f ,%f,%f ,%f ,%f ,%f \n",      _J_CoM[j][2* J_NUM + 0], _J_CoM[j][2* J_NUM + 1], _J_CoM[j][2* J_NUM + 2], _J_CoM[j][2* J_NUM + 3], _J_CoM[j][2*J_NUM + 4], _J_CoM[j][2*J_NUM + 5], _J_CoM[j][2*J_NUM + 6]);
    // }
    for (int j=0; j<d_JDOF;j++)
    {
        for(int i=0; i<3*d_JDOF; i++)
        {
            _J_CoMimsi[i] +=_J_CoM[j][i];
        }
    }
    // printf("_J_CoMimsi  %f ,%f ,%f, %f ,%f ,%f ,%f \n", _J_CoMimsi[0* J_NUM + 0], _J_CoMimsi[0* J_NUM + 1], _J_CoMimsi[0* J_NUM + 2], _J_CoMimsi[0* J_NUM + 3], _J_CoMimsi[0*J_NUM + 4], _J_CoMimsi[0*J_NUM + 5], _J_CoMimsi[0*J_NUM + 6]);
    // printf("            %f ,%f ,%f, %f ,%f ,%f ,%f \n", _J_CoMimsi[1* J_NUM + 0], _J_CoMimsi[1* J_NUM + 1], _J_CoMimsi[1* J_NUM + 2], _J_CoMimsi[1* J_NUM + 3], _J_CoMimsi[1*J_NUM + 4], _J_CoMimsi[1*J_NUM + 5], _J_CoMimsi[1*J_NUM + 6]);
    // printf("            %f ,%f ,%f, %f ,%f ,%f ,%f \n", _J_CoMimsi[2* J_NUM + 0], _J_CoMimsi[2* J_NUM + 1], _J_CoMimsi[2* J_NUM + 2], _J_CoMimsi[2* J_NUM + 3], _J_CoMimsi[2*J_NUM + 4], _J_CoMimsi[2*J_NUM + 5], _J_CoMimsi[2*J_NUM + 6]);

    // printf("tot mass : %f \n", Tot_mass[0]);
    for(int i=0; i<3*d_JDOF; i++)
    {
        // _J_CoMimsi[i] /= Tot_mass[0];
        J_COM[i] = _J_CoMimsi[i] /Tot_mass[0];
    }

    // printf("_J_CoM  %f ,%f ,%f, %f ,%f ,%f ,%f \n", J_COM[0* J_NUM + 0], J_COM[0* J_NUM + 1], J_COM[0* J_NUM + 2], J_COM[0* J_NUM + 3], J_COM[0*J_NUM + 4], J_COM[0*J_NUM + 5], J_COM[0*J_NUM + 6]);
    // printf("        %f ,%f ,%f, %f ,%f ,%f ,%f \n", J_COM[1* J_NUM + 0], J_COM[1* J_NUM + 1], J_COM[1* J_NUM + 2], J_COM[1* J_NUM + 3], J_COM[1*J_NUM + 4], J_COM[1*J_NUM + 5], J_COM[1*J_NUM + 6]);
    // printf("        %f ,%f ,%f, %f ,%f ,%f ,%f \n", J_COM[2* J_NUM + 0], J_COM[2* J_NUM + 1], J_COM[2* J_NUM + 2], J_COM[2* J_NUM + 3], J_COM[2*J_NUM + 4], J_COM[2*J_NUM + 5], J_COM[2*J_NUM + 6]);
}

__device__ void Main_loop(
    const float* PosLink,      // J_NUM * POS
    const float* RotLink,      // J_NUM * ROT
    const float* rho,          // J_NUM * POS
    const float* axis,         // J_NUM * POS
    const float* I_G,          // J_NUM * ROT
    const float* mass,         // J_NUM
    const float* Tot_mass,     // 1
    const float* id_parents,   // J_NUM (float or int)
    const float* q,            // next_q_input (J_NUM)
    const float* qdot,         // next_qdot_input (J_NUM)
    float* EE_pos,             // 길이 3
    float* EE_rot,              // 길이 9
    float* link3, float* link4, float* link5, float* link7, float* link7e // 길이 3 for collision check
){

    int d_JDOF = J_NUM;

    float _Pos_CoM[J_NUM][3] = { {0.0f} };
    float _Body_pos[J_NUM][3] = { {0.0f} };
    float _Body_Rot[J_NUM][9] = { {0.0f} };
    float _B2Link_pos[J_NUM][3] = { {0.0f} };
    float _B2Link_Rot[J_NUM][9] = { {0.0f} };
    float _B2Link_CoM[J_NUM][3] = { {0.0f} };

    float B2Link_pos_stacked[3 * J_NUM] = {0.0};
    float B2Link_Rot_stacked[9 * J_NUM] = {0.0};
    float B2Link_CoM_stacked[3 * J_NUM] = {0.0};

     // 링크별로 접근할 때 인덱스 계산
    for (int i = 0; i < d_JDOF; ++i) {
        // link i의 pos, rho, axis 시작 포인터
        const float* pos_i  = &PosLink[i * POS];   // 길이 3
        const float* rho_i  = &rho[i * POS];       // 길이 3
        const float* axis_i = &axis[i * POS];      // 길이 3

        // link i의 회전행렬(3x3) 시작 포인터
        const float* R_i    = &RotLink[i * ROT];   // 길이 9
        const float* IG_i   = &I_G[i * ROT];       // 길이 9 (나중에 사용)

        float qi    = q[i];
        float qdoti = qdot[i];

        ComputeTransformationMatrix(
            pos_i,         // float[3]
            R_i,           // float[9]
            qi,            // float[J_NUM]
            rho_i,         // float[3]
            axis_i,        // float[3]
            _Body_pos[i],  // output 3
            _Body_Rot[i],  // output 9
            _Pos_CoM[i]    // output 3
        );
    }

    // printf("_q << %f, %f, %f, %f, %f, %f, %f; \n",q[0], q[1], q[2], q[3], q[4], q[5],
    //     q[6]);

    // printf("_Body_pos: %f, %f %f \n", _Body_pos[0][0],  _Body_pos[0][1], _Body_pos[0][2]);
    // printf("         : %f, %f %f \n", _Body_pos[1][0],  _Body_pos[1][1], _Body_pos[1][2]);
    // printf("         : %f, %f %f \n", _Body_pos[2][0],  _Body_pos[2][1], _Body_pos[2][2]);
    // printf("         : %f, %f %f \n", _Body_pos[3][0],  _Body_pos[3][1], _Body_pos[3][2]);
    // printf("         : %f, %f %f \n", _Body_pos[4][0],  _Body_pos[4][1], _Body_pos[4][2]);
    // printf("         : %f, %f %f \n", _Body_pos[5][0],  _Body_pos[5][1], _Body_pos[5][2]);
    // printf("         : %f, %f %f \n", _Body_pos[6][0],  _Body_pos[6][1], _Body_pos[6][2]);


    for (int i = 0; i < d_JDOF; ++i)
    {
        int parent = id_parents[i];

        if (parent < 0) {
            // base link인 경우: 자기 자신의 변환이 누적 변환
            for (int k = 0; k < 9; ++k) _B2Link_Rot[i][k] = _Body_Rot[i][k];
            for (int k = 0; k < 3; ++k) {
                _B2Link_pos[i][k] = _Body_pos[i][k];
                _B2Link_CoM[i][k] = _Pos_CoM[i][k];
            }
        } else {
            // 부모가 있는 경우: 누적 변환 적용
            ComputeEachLinkPose(
                _B2Link_Rot[parent], _B2Link_pos[parent],   // 부모의 누적 변환
                _Body_pos[i], _Body_Rot[i], _Pos_CoM[i],    // 현재 링크의 상대 pose
                _B2Link_Rot[i], _B2Link_pos[i], _B2Link_CoM[i]  // 누적된 결과
            );
        }
    }
    // printf("_B2Link_p: %f, %f %f \n", _B2Link_pos[0][0],  _Body_pos[0][1], _Body_pos[0][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[1][0],  _B2Link_pos[1][1], _B2Link_pos[1][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[2][0],  _B2Link_pos[2][1], _B2Link_pos[2][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[3][0],  _B2Link_pos[3][1], _B2Link_pos[3][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[4][0],  _B2Link_pos[4][1], _B2Link_pos[4][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[5][0],  _B2Link_pos[5][1], _B2Link_pos[5][2]);
    // printf("         : %f, %f %f \n", _B2Link_pos[6][0],  _B2Link_pos[6][1], _B2Link_pos[6][2]);



    // // j = 0 ~ J_NUM
    // for (int j = 0; j < d_JDOF; j ++)
    // {
    //     for(int i=0; i<3; i++){
    //         B2Link_pos_stacked[3*(j)+i] = _B2Link_pos[j][i];
    //         B2Link_CoM_stacked[3*(j)+i] = _B2Link_CoM[j][i];
    //     }
    //     for(int i=0; i<9 ; i++){
    //         B2Link_Rot_stacked[9*(j)+i] = _B2Link_Rot[j][i];
    //     }
    // }


    float pos_offset_EE[3];    float rot_offset_EE[9];

    pos_offset_EE[0] = 0.0;    pos_offset_EE[1] = 0.0;    pos_offset_EE[2] = 0.0;

    rot_offset_EE[0] = 1.0;    rot_offset_EE[1] = 0.0;    rot_offset_EE[2] = 0.0;    //  Rotation Matrix
    rot_offset_EE[3] = 0.0;    rot_offset_EE[4] = 1.0;    rot_offset_EE[5] = 0.0;
    rot_offset_EE[6] = 0.0;    rot_offset_EE[7] = 0.0;    rot_offset_EE[8] = 1.0;

    // body id   right    //   left
    int ee_id = 6;     

    GetPose(_B2Link_pos[ee_id], _B2Link_Rot[ee_id],
        pos_offset_EE, rot_offset_EE,
        EE_pos, EE_rot,
        link3, link4, link5, link7, link7e);
    
    GetLinkPos(_B2Link_pos[2], link3);
    GetLinkPos(_B2Link_pos[3], link4);
    GetLinkPos(_B2Link_pos[4], link5);
    GetLinkPos(_B2Link_pos[6], link7);

    float e7_offset[3] ={0.0, 0.0, 0.15}; 
    GetLinkPos_offset(_B2Link_pos[6], _B2Link_Rot[6], e7_offset, link7e);
    // // check

    // printf("EE_pos : %f %f %f \n", EE_pos[0], EE_pos[1], EE_pos[2]);
    // printf("EE_rot   %f %f %f, \n ", EE_rot[0], EE_rot[1], EE_rot[2]);
    // printf("         %f %f %f, \n ", EE_rot[3], EE_rot[4], EE_rot[5]);
    // printf("         %f %f %f, \n ", EE_rot[6], EE_rot[7], EE_rot[8]);

    // printf("EE_pos_r : %f %f %f \n", EE_pos_r[0], EE_pos_r[1], EE_pos_r[2]);
    // printf("EE_rot_r   %f %f %f, \n ", EE_rot_r[0], EE_rot_r[1], EE_rot_r[2]);
    // printf("           %f %f %f, \n ", EE_rot_r[3], EE_rot_r[4], EE_rot_r[5]);
    // printf("           %f %f %f, \n ", EE_rot_r[6], EE_rot_r[7], EE_rot_r[8]);
}


__device__ float powN_cuda(int powN, float x) {
    float y;
    y = x;
    if(powN == 0)
    {
        return 1;
    }
    else if(powN == 1)
    {
        return x;
    }
    else
    {
        for(int i = 0; i<powN-1; i++)
        {
            x = x*y;
        }
        return x;
    }
}

__device__ void inverse22(float* A, float* A_inv)
{
    float det = A[0]*A[3] - A[1]*A[2];

    A_inv[0] =  A[3] / det ;
    A_inv[1] = -A[1] / det ;
    A_inv[2] = -A[2] / det ;
    A_inv[3] =  A[0] / det ;

}

__device__ void massMatrixInverseLDLT(float* A, float* A_inv, int n)
{
    float L[7 * 7] = {0};  // Maximum expected size
    float D[7] = {0};

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * n + k] * D[k] * L[j * n + k];
            }
            if (i == j) {
                D[j] = sum;
                L[j * n + j] = 1.0f;
            } else {
                L[i * n + j] = sum / D[j];
            }
        }
    }

    float Linv[7 * 7] = {0};
    for (int i = 0; i < n; ++i) {
        Linv[i * n + i] = 1.0f;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = Linv[i * n + j];
            for (int k = j + 1; k <= i; ++k) {
                sum -= L[i * n + k] * Linv[k * n + j];
            }
            Linv[i * n + j] = sum / L[i * n + i];
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = (i == j) ? 1.0f / D[i] : 0.0f;
            for (int k = i + 1; k < n; ++k) {
                sum -= Linv[k * n + i] * Linv[k * n + j] / D[k];
            }
            A_inv[i * n + j] = A_inv[j * n + i] = sum;
        }
    }
}

__global__ void MPPI(float* next_qacc_all,
                    float* dev_id_parents,
                    const float* dev_PosLink,                   // size: J_NUM * POS
                    const float* dev_RotLink,                   // size: J_NUM * ROT
                    const float* dev_rho,                       // size: J_NUM * POS
                    const float* dev_axis,                      // size: J_NUM * POS
                    const float* dev_I_G,                       // size: J_NUM * ROT
                    const float* dev_mass,                      // size: J_NUM
                    const float* dev_Tot_mass,                  // size: 1
                    float *des_pos_l, float *des_EE_rot_l,      // size : 3 // 9
                    float* cuda_dt2,                            // size : 1
                    float *q, float *qdot,                      // size : 7 // 7
                    float *noise,                               // size : J_NUM * N * T
                    float *next_q_all, float *next_qdot_all,    // size : J_NUM * N * T // J_NUM * N * T
                    float *cost_pos_l, float *cost_rpy_l,       // size : N * T // N * T
                    float* final_cost_all,                      // size : J_NUM * N 
                    float *u_vector_all                         // size : J_NUM * T
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // N*T 개

    // ====== Cost function weight ====== //
    float Q_p = 1000000.0f;     // Q: weight for desired ee pos
    float Q_o = 50000.0f;       // Q2: weight for desired ee ori

    float penalty = 10000000;
    ////////////////////////////////////////

    float EE_l_pos[3] = {0.0};                          float EE_l_rot[9] = {0.0};

    float next_q_input[J_NUM], next_qdot_input[J_NUM];

    // get initial data
    float dev_q[J_NUM];                  float dev_qdot[J_NUM];
    for(int i=0; i<J_NUM; i++){
        dev_q[i] = q[i];
        dev_qdot[i] = qdot[i];
    }
    
    float d_q_limit_min[7] = {-2.6, -1.65, -2.8, -2.9, -2.7, 0.5, -2.9};
    float d_q_limit_max[7] = { 2.6,  1.65,  2.8, -0.15, 2.75, 4.4, 2.9};
    float d_qdot_limit[7];
    for(int i=0; i<J_NUM; i++){
        d_qdot_limit[i] = 0.8;
    }

    float q_mid[J_NUM];
    for (int j = 0; j < J_NUM; j++) {
        q_mid[j] = 0.5f * (d_q_limit_min[j] + d_q_limit_max[j]);  
    }

    float link3[3], link4[3], link5[3], link7[3], link7e[3];

    if(tid < MainIndex){ // MainIndex = cuda_K : mean horizon
        int baseU   = tid * cuda_T * J_NUM;

        #pragma unroll
        for (int j = 0; j < J_NUM; ++j) {
            int idx_final = IDX_FINAL_COST(j, tid);
            final_cost_all[idx_final] = 0.0f;
        }

        #pragma unroll
        for(int i = 0; i<cuda_T; i++)
        {   
            int idx_common = tid * cuda_T + i;
            // Loop over each joint and then over the time horizon.
            #pragma unroll
            for (int j = 0; j < J_NUM; j++) {
                int offset = j * cuda_T;

                int idx_q = IDX_Q(j, tid, i);
                // qacc update -->  qacc{t+1} = del_qacc{t} + samples(u)
                // tmp_values => q_min : -10, q_max : 10
                // next_qacc_all[idx_q] = prev_qacc[j][0] + u[baseU + offset + i];
                //
                next_qacc_all[idx_q] = u_vector_all[IDX_VEC(j, 0)] + noise[baseU + offset + i];
            }

            float dt = (i < cuda_T - 10) ? cuda_dt : cuda_dt2[0];  // cuda_dt2는 예: 0.25

            #pragma unroll
            for (int k = 0; k < J_NUM; k ++)
            {
                int idx_q = IDX_Q(k, tid, i);

                next_qdot_all[idx_q] = dev_qdot[k] + next_qacc_all[idx_q] *dt;
                next_q_all[idx_q] = dev_q[k] + dev_qdot[k]*dt;

                next_qdot_input[k] = next_qdot_all[idx_q];
                next_q_input[k] = next_q_all[idx_q];
            }
        
            // 1) dynamics + EE pose
            Main_loop(
                dev_PosLink, dev_RotLink, dev_rho, dev_axis, dev_I_G,
                dev_mass, dev_Tot_mass, dev_id_parents,
                next_q_input, next_qdot_input,
                EE_l_pos, EE_l_rot,
                link3, link4, link5, link7, link7e
            );

            // // 2) EE tracking cost (pos + ori)
            float pos_cost = QuadraticPosCost3(des_pos_l, EE_l_pos, Q_p);
            float R_err[9];
            Rt_mul_R(des_EE_rot_l, EE_l_rot, R_err);

            float w[3];
            so3_log(R_err, w);

            float ori_cost = QuadraticCost(w, 3, Q_o);
            
            cost_pos_l[idx_common] = pos_cost;
            cost_rpy_l[idx_common] = ori_cost;

            // cost_rpy_l[tid*cuda_T+i] = 0.0;
            // for(int l=0; l<9; l++){
            //     cost_rpy_l[tid*cuda_T+i] += Q_o*powf(des_EE_rot_l[l] - EE_l_rot[l],  2.0f);
            // }

            float cost_col = CollisionCheck(link3, link4, link5, link7, link7e);

            // if(tid == 0 && i < 5){
            //     // printf("link3[%d, %d]: %f, %f, %f \n", tid, i, link3[0], link3[1], link3[2]);
            //     // printf("link4[%d, %d]: %f, %f, %f \n", tid, i, link4[0], link4[1], link4[2]);
            //     // printf("link5[%d, %d]: %f, %f, %f \n", tid, i, link5[0], link5[1], link5[2]);
            //     // printf("link7[%d, %d]: %f, %f, %f \n", tid, i, link7[0], link7[1], link7[2]);
            //     // printf("link7e[%d, %d]: %f, %f, %f \n", tid, i, link7e[0], link7e[1], link7e[2]);
            //     printf("cost_col: %f \n", cost_col);
            // }
            // Compute the index for the current time step.
            int idx = tid * cuda_T + i;

            float cost_null[J_NUM] = {0.0};
            float cost_q[J_NUM] = {0.0};
            float cost_qdot[J_NUM] = {0.0};
            // Loop over all 7 joints.
            #pragma unroll
            for (int j = 0; j < J_NUM; j++) {
                float q    = next_q_input[j];
                float qdot = next_qdot_input[j];

                float qmin = d_q_limit_min[j];
                float qmax = d_q_limit_max[j];
                float qdot_lim = d_qdot_limit[j];

                // 위치 제한 위반 여부 (0 또는 1로 변환)
                float pos_violation =
                    (q < qmin || q > qmax) ? 1.0f : 0.0f;

                // 속도 제한 위반 여부
                float vel_violation =
                    (qdot < -qdot_lim || qdot > qdot_lim) ? 1.0f : 0.0f;

                float qdot_cost_base = 500.0f * qdot * qdot;

                cost_q[j]    = pos_violation * penalty;
                cost_qdot[j] = qdot_cost_base + vel_violation * penalty;

                float e_q = q_mid[j] - q;
                cost_null[j] = 1000.0f * (e_q * e_q);
            }

            // 인덱스 계산 (공통 인덱스)
            
            // 1. dev_q 및 dev_qdot 업데이트 (각 관절의 현재 상태)
            // next_qdot[]와 next_q[]를 바로 사용합니다.
            // #pragma unroll
            for (int j = 0; j < J_NUM; j++) {
                dev_qdot[j] = next_qdot_input[j];  // 다음 루프에서 사용할 기반값
                dev_q[j] = next_q_input[j];  // 값을 읽어서 다음 루프에 사용
            }

            float cost_sum = 0.0;
            #pragma unroll
            for (int j = 0; j < J_NUM; j++) {
                int idx_final = IDX_FINAL_COST(j, tid);  // flat index for final_cost_all

                cost_sum =
                cost_pos_l[idx_common]
                + cost_rpy_l[idx_common]
                // + cost_u[idx_joint]
                + cost_qdot[j] + cost_q[j]
                + cost_null[j]
                + cost_col
                ;

                // S += S_0 ~ S_T-1
                final_cost_all[idx_final] += cost_sum;
            }

        }
        // // Add terminal cost
        int idx_T = tid * cuda_T + cuda_T - 1;  // 마지막 타임스텝 인덱스
        float terminal_cost = 10.0f * (cost_pos_l[idx_T] + cost_rpy_l[idx_T]);

        #pragma unroll
        for (int j = 0; j < J_NUM; j++) {
            int idx_final_T = IDX_FINAL_COST(j, tid);  // j*cuda_K + tid
            final_cost_all[idx_final_T] += terminal_cost;
        }

    }
}

//---------------------------------------------------------------------------------
__global__ void init_rng(curandState *state, unsigned long seed, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_random_numbers(const float * __restrict__ sigma,
                                        curandState *state,
                                        float *u,
                                        float mean,
                                        int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int joint_num = tid % J_NUM;

    curandState localState = state[tid];
    float noise = curand_normal(&localState);
    state[tid] = localState;               // 업데이트된 state 다시 저장

    u[tid] = noise * sigma[joint_num] + mean;
}

__global__ void set_input(float* single_u, float* multi_u)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int d_J_DOF = J_NUM;
    int d_TimeWinodw = cuda_T;

    int index3 = tid % (d_TimeWinodw*d_J_DOF);
    int joint_num = index3 / d_TimeWinodw;
    int win_num = index3 % d_TimeWinodw;
    int sam_num = tid / (d_TimeWinodw*d_J_DOF); //해당 joint의 몇 번째 sample인지

    multi_u[tid] = single_u[sam_num*d_J_DOF+joint_num];
}


void MPPI_qacc::mppi_host(float *state1, float *state2,
                float* des_EE_pos_l, float* des_EE_rot_l,
                float *prev_position_l, float* prev_rotation_l, float *returnArray)
{
    seed_count++;

    CUDA_CHECK_ERROR();
    // to do exteding one arm -> two arms
    cudaMemcpy(d_des_pos_l, des_EE_pos_l,  3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_des_EE_rot_l, des_EE_rot_l,  9 * sizeof(float), cudaMemcpyHostToDevice);

    cout << "left goal:   "<< des_EE_pos_l[0]<<", "<<des_EE_pos_l[1]<<", "<< des_EE_pos_l[2] << endl;
    cout << "left actual: "<< prev_position_l[0]<<", "<<prev_position_l[1]<<", "<< prev_position_l[2] << endl;

    float tttt = 0;

    _gap[0] = 0.0;

    for(int i=0; i<3; i++)
    {
        _gap[0] += abs(des_EE_pos_l[i] - prev_position_l[i]);
    }
    for(int i=0; i<9; i++)
    {
        _gap[0] += abs(des_EE_rot_l[i] - prev_rotation_l[i]) * 0.1;
    }
    cout << RED << " GAP : " << _gap[0] << RESET << endl<< endl;

    reset_check = 0;
    for(int i=0; i<J_NUM; i++)
    {
        if(abs(des_state1[i] - state1[i]) >= 0.1)
        {
            reset_check = 1;
            break;
        }
    }
    if(reset_check == 1)
    {
        for(int i=0; i<J_NUM; i++)
        {
            des_state1[i] = state1[i];
        }
    }
    // for(int i=0; i<J_NUM; i++)
    // {
    //     des_state1[i] = state1[i];
    //     des_state2[i] = state2[i];
    // }
    float _variance[J_NUM];
    for (int j = 0; j < J_NUM; j ++)
    {
        _variance[j] = 1.0;
    }

    cudaMemcpy(dev_sigma, _variance,  J_NUM * sizeof(float), cudaMemcpyHostToDevice);

    init_rng<<<Sampling_block, ThreadNum>>>(d_states, seed_count, total_K);
    generate_random_numbers<<<Sampling_block, ThreadNum>>>(
        dev_sigma,     // float* sigma  (J_NUM 크기)
        d_states,      // curandState* (재사용)
        d_u_single,    // output: K x J
        0.0f,          // mean
        total_K   // K * J_NUM
    );
    set_input<<<Expanding_block, ThreadNum>>>(d_u_single, d_u_all);

    // cudaMemcpy(dev_q, state1,  J_NUM * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_qdot, state2, J_NUM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_q, des_state1,  J_NUM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_qdot, des_state2, J_NUM * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_u_vector_all, u_vector_all,  total_T * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(u_all, d_u_all,  cuda_K*cuda_T *J_NUM* sizeof(float), cudaMemcpyDeviceToHost);


    _sol_cost = (_gap[0]+_gap[1]) * 100;   //scale up
    float _dt_sec1 = 10;
    float _dt_sec2 = 100;

    float ddtt = 0.001;
    if(_sol_cost > _dt_sec2)
    {
        _cuda_dt2[0] = _dt2_max;    //목적지가 매우 멀면 최대한 길게 보는 dynamic dt
    }
    else if(_sol_cost <= _dt_sec2 && _sol_cost > _dt_sec1)
    {
        _cuda_dt2[0] = -1 * ((_dt2_max - ddtt) / pow(_dt_sec1 - _dt_sec2,2)) * pow(_sol_cost - _dt_sec2,2) + _dt2_max;    //중간은 2차 보간
    }
    else
    {
        _cuda_dt2[0] = ddtt;    //목적지가 매우 가까우면 가장 가까이 보는 dynamic dt
    }

    cudaMemcpyAsync(dev_cuda_dt2, _cuda_dt2, sizeof(float), cudaMemcpyHostToDevice);


    MPPI<<<Main_block, ThreadNum>>>(dev_next_tau_all, dev_id_parents, dev_PosLink, dev_RotLink, dev_rho, dev_axis, dev_I_G, dev_mass, dev_Tot_mass,

                                    d_des_pos_l, dev_des_EE_rot_l,
            
                                    dev_cuda_dt2,

                                    dev_q, dev_qdot,
                                    d_u_all,
                                    dev_next_q_all, dev_next_qdot_all,

                                    d_cost_pos_l, dev_cost_rpy_l,
                                    dev_final_cost_all,

                                    dev_u_vector_all
    );

    float* final_cost_all = new float[J_NUM * cuda_K];
    cudaMemcpy(final_cost_all, dev_final_cost_all, J_NUM * cuda_K * sizeof(float), cudaMemcpyDeviceToHost);

    float min_cost[J_NUM];
    for (int j = 0; j < J_NUM; ++j) {
        float* cost_ptr = final_cost_all + j * cuda_K;
        min_cost[j] = *std::min_element(cost_ptr, cost_ptr + cuda_K);
    }

    float weights_sum[J_NUM] = {0.0f};
    float weights[J_NUM][cuda_K];
    float w_epsilon[J_NUM][cuda_T] = {0.0f};

    for (int j = 0; j < J_NUM; ++j) {
        for (int k = 0; k < cuda_K; ++k) {
            float cost = final_cost_all[IDX_FINAL_COST(j, k)];
            float w = expf(-(cost - min_cost[j]) / _lambda);
            weights[j][k] = w;
            weights_sum[j] += w;
        }
    }

    for(int t=0; t<cuda_T; t++)
    {
        for(int k=0; k<cuda_K; k++)
            {
            for (int j = 0; j < J_NUM ; j ++) {
                w_epsilon[j][t] += (weights[j][k]/weights_sum[j]) * u_all[k*J_NUM*(cuda_T)+cuda_T*j+t];

                }
            }
    }


    for (int j = 0; j < J_NUM; j++) {
        for (int t = 0; t < cuda_T; t++) {
            u_vector_all[IDX_VEC(j, t)] += w_epsilon[j][t];
        }
    }

    for (int j = 0; j < J_NUM; j++) {
        _u_star[j] = u_vector_all[IDX_VEC(j, 0)];
    }

    double conv_thr = 0.0001;
    for (int j = 0; j < J_NUM; j++)
    {
        bool use_zero = false;

        use_zero = (_gap[0] < conv_thr);

        if (use_zero)
        {
            returnArray[j] = 0.0;
            returnArray[J_NUM + j] = des_state1[j];

            des_state1[j] = returnArray[J_NUM + j];
            des_state2[j] = 0.0;
        }
        else
        {
            // returnArray[j] = _u_star[j];
            returnArray[j] = des_state2[j] + _u_star[j] * cuda_dt;
            returnArray[J_NUM + j] = des_state1[j] + returnArray[j] * cuda_dt;

            des_state1[j] = returnArray[J_NUM + j];
            des_state2[j] = returnArray[j];
        }
    }
}


int MPPI_qacc::calculateBlocks(int totalThreads, int threadsPerBlock) {
    return (totalThreads + threadsPerBlock - 1) / threadsPerBlock;  // 올림 계산
}

void MPPI_qacc::cuda_Initialize()
{   
    // 1) mass, id_parents, Tot_mass
    for (int k = 0; k < J_NUM; ++k) {
        h_mass[k]      = RModel._body_mass[k];
        h_id_parents[k] = static_cast<float>(RModel.id_parents[k]); // dev_id_parents를 float로 쓰고 있으면 이렇게
        // 만약 int로 쓰고 싶으면 host/dev 둘 다 int로 타입 변경 추천
    }
    h_Tot_mass[0] = RModel._Tot_mass;

    // 2) pos, rho, axis (크기: J_NUM x POS)
    for (int k = 0; k < J_NUM; ++k) {
        for (int i = 0; i < POS; ++i) {
            int idx = k * POS + i;

            h_PosLink[idx] = RModel._poslinks[k](i);
            h_rho[idx]     = RModel._rhos[k](i);
            h_axis[idx]    = RModel._joint_axiss[k](i);
        }
    }

    // 3) RotLink, RotCoM, I_G (크기: J_NUM x ROT, ROT = 9)
    for (int k = 0; k < J_NUM; ++k) {
        for (int i = 0; i < POS; ++i) {
            for (int j = 0; j < POS; ++j) {
                int idx = k * ROT + (3 * i + j);  // 3x3 row-major flatten

                h_RotLink[idx] = RModel._Rot_links[k](i, j);
                h_RotCoM[idx]  = RModel._Rot_CoMs[k](i, j);
                h_I_G[idx]     = RModel._I_G[k](i, j);
            }
        }
    }

    ////////////////////////////////////////////////
    cudaMemcpy(dev_PosLink, h_PosLink, J_NUM * POS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_RotLink, h_RotLink, J_NUM * ROT * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_rho, h_rho, J_NUM * POS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_RotCoM, h_RotCoM, J_NUM * ROT * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_axis, h_axis, J_NUM * POS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_I_G, h_I_G, J_NUM * ROT * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_mass,      h_mass, J_NUM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Tot_mass,  h_Tot_mass, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_id_parents, h_id_parents, J_NUM * sizeof(float), cudaMemcpyHostToDevice);    
}


void MPPI_qacc::cuda_memory(){
    cudaMalloc((void**)&dev_PosLink,  J_NUM * POS * sizeof(float));
    cudaMalloc((void**)&dev_RotLink,  J_NUM * ROT * sizeof(float));

    cudaMalloc((void**)&dev_rho,      J_NUM * POS * sizeof(float));
    cudaMalloc((void**)&dev_RotCoM,   J_NUM * ROT * sizeof(float));

    cudaMalloc((void**)&dev_axis,     J_NUM * POS * sizeof(float));
    cudaMalloc((void**)&dev_I_G,      J_NUM * ROT * sizeof(float));

    cudaMalloc((void**)&dev_mass,     J_NUM * sizeof(float));
    cudaMalloc((void**)&dev_Tot_mass, 1 * sizeof(float));
    cudaMalloc((void**)&dev_id_parents, J_NUM * sizeof(float));

    cudaMalloc((void**)&dev_des_EE_rot_l, 9 * sizeof(float));
}


void MPPI_qacc::cuda_data_free(){
    // ================= PRBDL data =================== //
    cudaFree(dev_PosLink);
    cudaFree(dev_RotLink);

    cudaFree(dev_rho);
    cudaFree(dev_RotCoM);

    cudaFree(dev_axis);
    cudaFree(dev_I_G);
    
    cudaFree(dev_mass);
    cudaFree(dev_Tot_mass);
    cudaFree(dev_id_parents);
    // ================= PRBDL data =================== //

    cudaFree(dev_des_EE_rot_l);
    cudaFree(dev_cost_rpy_l);
    cudaFree(d_des_pos_l);

    cudaFree(dev_cuda_dt2);
    cudaFree(dev_next_tau_all);
    cudaFree(dev_next_del_tau_all);

    cudaFree(dev_next_q_all);
    cudaFree(dev_next_qdot_all);

    cudaFree(dev_u_vector_all);
    cudaFree(dev_final_cost_all);
    cudaFree(dev_sigma);

    cudaFree(d_states);

}


void MPPI_qacc::state_init()
{

    Main_block = calculateBlocks(MainIndex, ThreadNum);
    Sampling_block = calculateBlocks(Samplingndex, ThreadNum);
    Expanding_block = calculateBlocks(ExpandingIndex, ThreadNum);
    /**
     * device 변수들 메모리 할당
    */
    cudaMalloc(&d_states, total_K * sizeof(curandState));

    cudaMalloc(&d_u_single,(cuda_K*J_NUM) * sizeof(float));
    cudaMalloc(&d_u_all, (cuda_K*cuda_T *J_NUM)* sizeof(float));


    cudaMalloc(&d_des_pos_l, 3 * sizeof(float));

    cudaMalloc(&dev_q, J_NUM * sizeof(float));
    cudaMalloc(&dev_qdot, J_NUM * sizeof(float));

    cudaMalloc(&d_cost_pos_l,cuda_K*cuda_T* sizeof(float));


    cudaMalloc(&dev_cost_rpy_l,cuda_K*cuda_T* sizeof(float));



    cudaMalloc(&d_cost_u, (cuda_K*cuda_T* J_NUM + J_NUM * cuda_T)*sizeof(float));


    cudaMalloc(&dev_cuda_dt2, 1 * sizeof(float));


    cudaError_t err;



    cudaMalloc(&dev_sigma, J_NUM * sizeof(float));


    // /**
    //  * device 변수들 host 변수로 메모리 copy
    // */
    cudaMemset(d_u_single, 0.0,  (cuda_K *J_NUM)* sizeof(float));
    cudaMemset(d_u_all, 0.0,  cuda_K*cuda_T*J_NUM* sizeof(float));

    //////////////////////////////////////////////////////////////////
    ///// all memory set to 0.0
    //////////////////////////////////////////////////////////////////
    ///
    cudaMalloc(&dev_next_tau_all, total_KT * sizeof(float));
    cudaMalloc(&dev_next_del_tau_all, total_KT * sizeof(float));

    cudaMalloc(&dev_next_q_all, total_KT * sizeof(float));
    cudaMalloc(&dev_next_qdot_all, total_KT * sizeof(float));

    cudaMalloc(&dev_u_vector_all, total_T * sizeof(float));
    cudaMalloc(&dev_final_cost_all, total_K * sizeof(float));

    cudaMemset(dev_next_tau_all, 0, total_KT * sizeof(float));
    cudaMemset(dev_next_del_tau_all, 0, total_KT * sizeof(float));

    cudaMemset(dev_next_q_all, 0, total_KT * sizeof(float));
    cudaMemset(dev_next_qdot_all, 0, total_KT * sizeof(float));

    cudaMemset(dev_u_vector_all, 0, total_T * sizeof(float));
    cudaMemset(dev_final_cost_all, 0, total_K * sizeof(float));


}