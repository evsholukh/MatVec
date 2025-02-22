#pragma once

#include <vector>
#include <random>


class Vector {
private:
    std::vector<float> _data;
public:
    Vector(std::vector<float> data): _data(data) {}
    ~Vector() {}

    static Vector generate(size_t size);

    float size_mb();
    float sum();
    void print();

    std::vector<float>& data();
};

inline Vector Vector::generate(size_t size) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(generator);
    }
    return Vector(data);
}

float Vector::size_mb() {
    return (sizeof(float) * this->_data.size()) / (1024 * 1024);
}

inline float Vector::sum() {
    float res = 0.0f;
    for (size_t i = 0; i < this->_data.size(); i++) {
        res += this->_data.at(i);
    }
    return res;
}

inline void Vector::print() {
    for (size_t i = 0; i < this->_data.size(); i++) {
        std::cout << this->_data.at(i);
        if (i < this->_data.size() - 2)  {
            std::cout << ",";
        } else {
            std::cout << std::endl;
        }
    }
}

inline std::vector<float>& Vector::data() {
    return this->_data;
}

