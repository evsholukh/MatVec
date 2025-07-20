#pragma once

#include <chrono>

class Utils
{

public:
    /**
     * @brief Measures the execution time of a function.
     *
     * This method accepts a callable (function or lambda) object and returns the duration in seconds.
     * It measures the start time before calling the function, and end time after it's done, then calculates the difference.
     *
     * @tparam F The type of the function/lambda to be measured.
     * @param func The callable (function or lambda) object to measure.
     * @return float The duration in seconds.
     */
    template <typename F>
    static float measure(F func)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        func();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end_time - start_time;

        return elapsed.count();
    }

    /**
     * @brief Creates a new array of the given size, filled with zeros or a specified value.
     *
     * This function creates an array of the given 'size' and fills it with either zeros (default) or a user-specified value.
     * If the 'group_size' argument is provided, the array will be padded at the end to ensure that it's a multiple of 'group_size'.
     * The function returns a pointer to the newly created array.
     *
     * @tparam T The type of the elements in the array.
     * @param size The desired size of the new array.
     * @param group_size The size of the groups for padding purposes (default is 1).
     * @param val The value to fill the array with (default is 0).
     * @return T* A pointer to the newly created and filled array.
     */
    template <typename T>
    static T *create_array(const size_t size, const size_t group_size = 1, const T val = T(0))
    {
        const int groups_count = (size + group_size - 1) / group_size;
        const int global_size = group_size * groups_count;

        auto data = new T[global_size];

        fill_array(data, global_size, T(0));
        fill_array(data, size, val);

        return data;
    }

    /**
     * @brief Randomizes the elements in an array.
     *
     * This function takes a pointer to an array and its 'size', then fills the array with random values from a uniform distribution between -1 and 1.
     *
     * @tparam T The type of the elements in the array.
     * @param data A pointer to the array to be randomized.
     * @param size The size of the array.
     */
    template <typename T>
    static void randomize_array(T *data, const size_t size)
    {
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> dist(-1, 1);

        for (size_t i = 0; i < size; i++)
        {
            data[i] *= dist(generator);
        }
    }

    /**
     * @brief Fills an array with the specified value.
     *
     * This function takes a pointer to an array, its 'size', and a value to fill the array with.
     * It fills the array with the provided value.
     *
     * @tparam T The type of the elements in the array.
     * @param data A pointer to the array to be filled.
     * @param size The size of the array.
     * @param val The value to fill the array with (default is 0).
     */
    template <typename T>
    static void fill_array(T *data, const size_t size, const T val = T(0))
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = val;
        }
    }
};
