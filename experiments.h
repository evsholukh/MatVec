
#include "measured.h"
#include "vector.h"
#include "cl.h"


class VectorAdd : public TimeMeasured {
private:
    Vector x;
    Vector y;
    OpenCLHelper helper;

public:
    VectorAdd(const size_t size) : x(Vector::generate(size)),
                                   y(Vector::generate(size)),
                                   helper(OpenCLHelper()) {}
    ~VectorAdd() {};
    virtual void perform();
};

inline void VectorAdd::perform() {
    helper.vector_add(this->x.data(), this->y.data());
}

class VectorSum : public TimeMeasured {
    private:
        Vector x;
        Vector y;
        OpenCLHelper helper;
    
    public:
        VectorSum(const size_t size) : x(Vector::generate(size)),
                                       y(Vector::generate(size)),
                                       helper(OpenCLHelper()) {}
        ~VectorSum() {};
        virtual void perform();
};
    
inline void VectorSum::perform() {
    helper.vector_sum(this->x.data());
}
