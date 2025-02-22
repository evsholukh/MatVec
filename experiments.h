#pragma once

// #include "measured.h"
// #include "vector.h"
// // #include "cl.h"


// class VectorAddRuntime : public TimeMeasured {
// protected:
//     Vector x, y;
// public:
//     VectorAddRuntime(std::vector<float> x, std::vector<float> y): x(x), y(y) {}
//     VectorAddRuntime(const size_t size) : x(Vector::random(size)), y(Vector::random(size)) {};

//     virtual void perform() override {
//         this->x.add(y);
//     }
// };

// class VectorSumRuntime : public TimeMeasured {
// protected:
//     Vector vec;
// public:
//     VectorSumRuntime(std::vector<float> vec): vec(vec) {}
//     VectorSumRuntime(const size_t size) : vec(Vector::random(size)) {};

//     virtual void perform() override {
//         this->vec.sum();
//     }
// };

// class VectorDotRuntime : public VectorAddRuntime {
// public:
//     VectorDotRuntime(std::vector<float> x, std::vector<float> y): VectorAddRuntime(x, y) {}
//     VectorDotRuntime(size_t size) : VectorAddRuntime(size) {}

//     virtual void perform() override {
//         this->x.dot(this->y);
//     }
// };

// class VectorAddOpenCL : public VectorAddRuntime {
// protected:
//     OpenCLHelper helper;

// public:
//     VectorAddOpenCL(std::vector<float> x, std::vector<float> y): VectorAddRuntime(x, y) {}
//     VectorAddOpenCL(size_t size) : VectorAddRuntime(size) {}

//     virtual void perform() override {
//         helper.vector_add(this->x.vec(), this->y.vec());
//     }
// };

// class VectorSumOpenCL : public VectorSumRuntime {
// protected:
//     OpenCLHelper helper;

// public:
//     VectorSumOpenCL(std::vector<float> vec) : VectorSumRuntime(vec), helper(OpenCLHelper()) {}
//     VectorSumOpenCL(const size_t size) : VectorSumRuntime(size), helper(OpenCLHelper()) {}

//     virtual void perform() override {
//         this->helper.vector_sum(this->vec.vec());
//     };
// };

// class VectorDotOpenCL : public VectorAddOpenCL {
// public:
//     VectorDotOpenCL(std::vector<float> x, std::vector<float> y): VectorAddOpenCL(x, y) {}
//     VectorDotOpenCL(size_t size) : VectorAddOpenCL(size) {}

//     virtual void perform() override {
//         helper.vector_dot(this->x.vec(), this->y.vec());
//     }
// };

