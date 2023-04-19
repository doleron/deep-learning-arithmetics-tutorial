#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <thread>

using namespace Eigen;

const int size = 512;

float foo(const MatrixXd &A, const MatrixXd &B, MatrixXd &C) {
    float result = 0.0;
    for (int i = 0; i < 100; ++i)
    {
        C.noalias() = A * B;

        int x = 0;
        int y = 0;

        result += C(x, y);
    }
    return result;
}

void worker(const std::string & id) {

    std::chrono::high_resolution_clock::time_point begin_time_ref;
    std::chrono::high_resolution_clock::time_point end_time_ref;

    const MatrixXd A = 10 * MatrixXd::Random(size, size);
    const MatrixXd B = 10 * MatrixXd::Random(size, size);

    MatrixXd C;
    double test = 0;

    const int max = 30;
    for (int step = 0; step < max; ++step) {
        begin_time_ref = std::chrono::high_resolution_clock::now();

        test += foo(A, B, C);

        end_time_ref = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ref - begin_time_ref);
        auto duration = ms.count();
        float fps = 100 * 1000.0 / duration;
        std::cout << id << "\t" << step << "\t" << fps << " fps\t" << duration << " ms\n";
       
    }

    std::cout << "test value is:" << test << "\n";

}

int main(int argc, char ** argv)
{

    Eigen::initParallel();

    std::cout << Eigen::nbThreads() << " eigen threads\n";

    std::thread t0(worker, "t-0");
    //std::thread t1(worker, "t-1");
    //std::thread t2(worker, "t-2");

    t0.join();
    //t1.join();
    //t2.join();

    return 0;
}
