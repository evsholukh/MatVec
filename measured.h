
#include <chrono>

class TimeMeasured {
private:
public:
    TimeMeasured();
    ~TimeMeasured();

    float measure();
    virtual void perform() = 0;
};

TimeMeasured::TimeMeasured() { }

TimeMeasured::~TimeMeasured() { }

inline float TimeMeasured::measure() {
    auto start_time = std::chrono::high_resolution_clock::now();
    this->perform();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end_time - start_time;

    return elapsed.count();
}
