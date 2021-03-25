#include <assert.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>



using namespace std;
#define USING_SIMD

struct vec4
{
    union {
        float x, r;
    };

    union {
        float y, g;
    };

    union {
        float z, b;
    };

    union {
        float w, a;
    };

    inline float& operator[](int index)
    {
        assert(0 <= index && index < 4);
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            break;
        }
        // shouldn't be here;
        assert(false);
    }

    inline const float& operator[](int index) const
    {
        assert(0 <= index && index < 4);
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            break;
        }
        // shouldn't be here;
        assert(false);
    }
    /*
    float operator*=(const vec4& v)
    {
        *this = *this * v;
        return (*this);
    }
    */
};

float operator*(const vec4& v1, const vec4& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

ostream& operator<<(ostream& os, const vec4& v)
{
    for (int i = 0; i < 4; i++)
    {
        os << v[i] << " ";
    }
    return os;
}


struct mat4
{
    vec4 data[4];

    inline vec4& operator[](int index)
    {
        assert(0 <= index && index < 4);
        return data[index];
    }

    inline const vec4& operator[](int index) const
    {
        assert(0 <= index && index < 4);
        return data[index];
    }
};



// 1.   4x4 matrix multiplication is 64 multiplication and 48 additions
// 2.   SSE can be reduced to 16 muptlications and 12 additions
mat4 operator*(const mat4& m1, const mat4& m2)
{
    vec4 a0 = m1[0];
    vec4 a1 = m1[1];
    vec4 a2 = m1[2];
    vec4 a3 = m1[3];

    vec4 b0 = { m2[0][0], m2[1][0], m2[2][0], m2[3][0] };
    vec4 b1 = { m2[0][1], m2[1][1], m2[2][1], m2[3][1] };
    vec4 b2 = { m2[0][2], m2[1][2], m2[2][2], m2[3][2] };
    vec4 b3 = { m2[0][3], m2[1][3], m2[2][3], m2[3][3] };

    mat4 m;
    m[0] = { a0 * b0, a0 * b1, a0 * b2, a0 * b3 };
    m[1] = { a1 * b0, a1 * b1, a1 * b2, a1 * b3 };
    m[2] = { a2 * b0, a2 * b1, a2 * b2, a2 * b3 };
    m[3] = { a3 * b0, a3 * b1, a3 * b2, a3 * b3 };
    return m;
}



/*
doing it row by row. the parrallel operations we want to do is:

so for the first row of the output matrix, we would have something like:

    m1[0][0] * m2[0][0]     m1[0][0] * m2[1][0]     m1[0][0] * m2[2][0]     m1[0][0] * m2[3][0]         |
             +                       +                       +                       +                  |
                                                                                                        |
    m1[0][1] * m2[0][1]     m1[0][1] * m2[1][1]     m1[0][1] * m2[2][1]     m1[0][1] * m2[3][1]         |
             +                       +                       +                       +                  |
                                                                                                        |
    m1[0][2] * m2[0][2]     m1[0][2] * m2[1][2]     m1[0][2] * m2[2][2]     m1[0][2] * m2[3][2]         |
             +                       +                       +                       +                  |
                                                                                                        V
    m1[0][3] * m2[0][3]     m1[0][3] * m2[1][3]     m1[0][3] * m2[2][3]     m1[0][3] * m2[3][3]


as you can see, so you would need

    m1[0][0] * _______      m1[0][0] * _______      m1[0][0] * _______      m1[0][0] * _______          |
             +                       +                       +                       +                  |
                                                                                                        |
    m1[0][1] * _______      m1[0][1] * _______      m1[0][1] * _______      m1[0][1] * _______          |
             +                       +                       +                       +                  |
                                                                                                        |
    m1[0][2] * _______      m1[0][2] * _______      m1[0][2] * _______      m1[0][2] * _______          |
             +                       +                       +                       +                  |
                                                                                                        V
    m1[0][3] * _______      m1[0][3] * _______      m1[0][3] * _______      m1[0][3] * _______


which boils down to

    _128m (m1[0][0]) * _______
                     +

    _128m (m1[0][1]) * _______
                     +

    _128m (m1[0][2]) * _______
                     +

    _128m (m1[0][3]) * _______
*/
mat4 SIMDMultiply(const mat4& m1, const mat4& m2)
{
#ifdef USING_SIMD

    __m128 b0 = _mm_load_ps(&m2[0][0]);
    __m128 b1 = _mm_load_ps(&m2[1][0]);
    __m128 b2 = _mm_load_ps(&m2[2][0]);
    __m128 b3 = _mm_load_ps(&m2[3][0]);

    mat4 m;

    for (int i = 0; i < 4; i++)
    {
        __m128 m1_0 = _mm_set1_ps(m1[i][0]);
        __m128 m1_1 = _mm_set1_ps(m1[i][1]);
        __m128 m1_2 = _mm_set1_ps(m1[i][2]);
        __m128 m1_3 = _mm_set1_ps(m1[i][3]);

        __m128 r0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m1_0, b0), _mm_mul_ps(m1_1, b1)),
            _mm_add_ps(_mm_mul_ps(m1_2, b2), _mm_mul_ps(m1_3, b3)));

        _mm_store_ps((float*)&m[i], r0);
    }

    /*
    __m128 m1_00 = _mm_set1_ps(m1[0][0]);
    __m128 m1_01 = _mm_set1_ps(m1[0][1]);
    __m128 m1_02 = _mm_set1_ps(m1[0][2]);
    __m128 m1_03 = _mm_set1_ps(m1[0][3]);

    __m128 m1_10 = _mm_set1_ps(m1[1][0]);
    __m128 m1_11 = _mm_set1_ps(m1[1][1]);
    __m128 m1_12 = _mm_set1_ps(m1[1][2]);
    __m128 m1_13 = _mm_set1_ps(m1[1][3]);

    __m128 m1_20 = _mm_set1_ps(m1[2][0]);
    __m128 m1_21 = _mm_set1_ps(m1[2][1]);
    __m128 m1_22 = _mm_set1_ps(m1[2][2]);
    __m128 m1_23 = _mm_set1_ps(m1[2][3]);

    __m128 m1_30 = _mm_set1_ps(m1[3][0]);
    __m128 m1_31 = _mm_set1_ps(m1[3][1]);
    __m128 m1_32 = _mm_set1_ps(m1[3][2]);
    __m128 m1_33 = _mm_set1_ps(m1[3][3]);

    __m128 r0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m1_00, b0), _mm_mul_ps(m1_01, b1)),
        _mm_add_ps(_mm_mul_ps(m1_02, b2), _mm_mul_ps(m1_03, b3)));
    __m128 r1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m1_10, b0), _mm_mul_ps(m1_11, b1)),
        _mm_add_ps(_mm_mul_ps(m1_12, b2), _mm_mul_ps(m1_13, b3)));
    __m128 r2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m1_20, b0), _mm_mul_ps(m1_21, b1)),
        _mm_add_ps(_mm_mul_ps(m1_22, b2), _mm_mul_ps(m1_23, b3)));
    __m128 r3 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m1_30, b0), _mm_mul_ps(m1_31, b1)),
        _mm_add_ps(_mm_mul_ps(m1_32, b2), _mm_mul_ps(m1_33, b3)));

    _mm_store_ps((float*)&m[0], r0);
    _mm_store_ps((float*)&m[1], r1);
    _mm_store_ps((float*)&m[2], r2);
    _mm_store_ps((float*)&m[3], r3);
    */
    return m;
#endif
}

ostream& operator<<(ostream& os, const mat4& m)
{
    for (int i = 0; i < 4; i++)
    {
        os << m[i] << "\n";
    }
    return os;
}

/*
http://www.cs.uu.nl/docs/vakken/magr/2017-2018/files/SIMD%20Tutorial.pdf

Fast 4x4 Matrix Inverse with SSE SIMD, Explained
https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

Streaming SIMD Extensions - Inverse of 4x4 Matrix
https://peertje.daanberg.net/drivers/intel/download.intel.com/design/PentiumIII/sml/24504301.pdf

Efficient 4x4 matrix multiplication (C vs assembly)
https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
*/

void MatrixMultiplication(int n)
{
    for (int i = 0; i < n; i++)
    {
        mat4 m1 = { 1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12,
                    13, 14, 15, 16 };

        mat4 m2 = { 1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12,
                    13, 14, 15, 16 };

        mat4 m3 = m1 * m2;
    }
}

void SIMDMatrixMultiplication(int n)
{
    for (int i = 0; i < n; i++)
    {
        mat4 m1 = { 1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16 };

        mat4 m2 = { 1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12,
                    13, 14, 15, 16 };

        mat4 m3 = SIMDMultiply(m1, m2);
    }
}

int main()
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop;
    std::chrono::microseconds duration;

    int iterations = 100000;
    {
        cout << "SIMDMatrixMultiplication" << endl;

        start = std::chrono::high_resolution_clock::now();
        SIMDMatrixMultiplication(iterations);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << "time: " << duration.count() / 1000000.0 << " seconds" << endl << endl;
    }

    {
        cout << "MatrixMultiplication" << endl;

        start = std::chrono::high_resolution_clock::now();
        MatrixMultiplication(iterations);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << "time: " << duration.count() / 1000000.0 << " seconds" << endl << endl;
    }
}









