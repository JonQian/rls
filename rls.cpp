#include <iostream>
#include <stdlib.h>
#include <string.h>
extern"C" {
#include <cblas.h>
}

void rls(int n, float *a, float *p, float *x, float b, float lbd = 1.0f)
{
    float *q = new float[n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n, n, 1.0f, x, n, p, n, 0, q, n);
    float qt = lbd;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, n, 1.0f, q, n, x, 1, 1.0f, &qt, 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, n, 1.0f / qt, p, n, x, 1, 0.0f, q, 1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, n, -1.0f, x, n, a, 1, 1.0f, &b, 1);
    cblas_saxpy(n, b, q, 1, a, 1);
    float *eye = new float[n * n];
    memset(eye, 0, sizeof(float)*n * n);
    for (int i = 0; i < n; i++) {
        eye[i * n + i] = 1.0f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, 1, -1.0f, q, 1, x, n, 1.0f, eye, n);
    float *cp = new float[n * n];
    memcpy(cp, p, sizeof(float)*n * n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f / lbd, eye, n, cp, n, 0.0f, p, n);
    delete [] cp;
    delete [] eye;
    delete [] q;
}

int main(int argc, char **argv)
{
    int n = 3;//自变量个数
    float a[] = {1, 2, 3};
    float p0 = 1000000.0f;
    float lbd = 1.0f;

    std::cout << "correct a: [";
    for (int i = 0; i < n - 1; i++) {
        std::cout << a[i] << ",";
    }
    std::cout << a[n - 1] << "]" << std::endl;

    //生成测试数据
    int ns = 50;//训练样本数
    float *x = new float[ns * n];
    float *b = new float[ns];
    for (int i = 0; i < ns; i++) {
        float *xi = x + i * n;
        b[i] = 0;
        for (int j = 0; j < n; j++) {
            xi[j] = 10.0f / 32767 * rand();
            b[i] += a[j] * xi[j];
        }
        b[i] += 0.005f / 32767 * rand();
    }

    float *ai = new float[n];
    for (int i = 0; i < n; i++) {
        ai[i] = 0;
    }
    float *pi = new float[n * n];
    memset(pi, 0, sizeof(float)*n * n);
    for (int i = 0; i < n; i++) {
        pi[i * n + i] = p0;
    }
    for (int i = 0; i < ns; i++) {
        float bi = b[i];
        float *xi = x + i * n;
        rls(n, ai, pi, xi, bi, lbd);
    }
    std::cout << "rls a: [";
    for (int i = 0; i < n - 1; i++) {
        std::cout << ai[i] << ",";
    }
    std::cout << ai[n - 1] << "]" << std::endl;

    delete [] pi;
    delete [] ai;
    delete [] b;
    delete [] x;
    return 0;
}
