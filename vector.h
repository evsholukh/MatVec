#pragma once

#include <iostream>

/**
 * @class Vector
 *
 * @brief This class represents a vector with data of type T and size _size. It has methods for sum, addition, multiplication, dot product and printing the vector.
 *
 * The template parameter T is used to specify the type of the elements in the vector. The size of the vector is stored in _size.
 *
 * @tparam T - the type of data stored in the vector.
 */
template <typename T>
class Vector
{

public:
    /**
     * @brief Constructor for the Vector class.
     *
     * It takes two parameters, a pointer to an array of elements of type T and size_t that specifies its size.
     *
     * @param data - a pointer to an array of elements of type T.
     * @param size - the size of the vector.
     */
    Vector(T *data, size_t size) : _data(data), _size(size) {}

    /**
     * @brief Accessor method for the private member variable '_data'.
     *
     * @return T* - a pointer to the vector's elements.
     */
    T *data() const { return _data; }

    /**
     * @brief Accessor method for the private member variable '_size'.
     *
     * @return size_t - the size of the vector.
     */
    size_t size() const { return _size; }

    /**
     * @brief Returns the sum of all elements in the vector.
     *
     * This method iterates over all the elements in the vector and returns their sum.
     *
     * @return T - the sum of all elements in the vector.
     */
    virtual T sum() const
    {
        T val = T(0);
        for (size_t i = 0; i < _size; i++)
        {
            val += _data[i];
        }
        return val;
    }

    /**
     * @brief Adds another vector to this one element-wise.
     *
     * This method iterates over all the elements in both vectors and adds corresponding elements from the second vector to the first.
     *
     * @param o - the Vector to add to this one.
     */
    virtual void add(const Vector<T> &o)
    {
        for (size_t i = 0; i < _size; i++)
        {
            _data[i] += o._data[i];
        }
    }

    /**
     * @brief Multiplies this vector with another one element-wise.
     *
     * This method iterates over all the elements in both vectors and multiplies corresponding elements from the second vector to the first.
     *
     * @param o - the Vector to multiply this one by.
     */
    virtual void mul(const Vector<T> &o)
    {
        for (size_t i = 0; i < _size; i++)
        {
            _data[i] *= o._data[i];
        }
    }

    /**
     * @brief Computes the dot product of this vector with another one.
     *
     * This method iterates over all the elements in both vectors and computes their dot product.
     *
     * @param o - the Vector to compute the dot product with.
     *
     * @return T - the result of the dot product.
     */
    virtual T dot(const Vector<T> &o) const
    {
        T val = T(0);

        for (size_t i = 0; i < _size; i++)
        {
            val += _data[i] * o._data[i];
        }

        return val;
    }

    /**
     * @brief Prints the vector to the standard output.
     *
     * This method prints all elements of the vector to the standard output, separated by commas and enclosed in square brackets.
     */
    virtual void print() const
    {
        std::cout << "[";

        for (size_t i = 0; i < _size; i++)
        {
            std::cout << _data[i];

            if (i != _size - 1)
            {
                std::cout << ",";
            }
        }

        std::cout << "]";
    }

    /**
     * @brief Returns the size of this vector in megabytes.
     *
     * This method returns the size of this vector in megabytes, calculated by dividing its total size (in bytes) by 1024*1024.
     *
     * @return size_t - the size of this vector in megabytes.
     */
    virtual size_t size_mb() const
    {
        return (sizeof(T) * _size) / (1024 * 1024);
    }

protected:
    T *_data;
    size_t _size;
};