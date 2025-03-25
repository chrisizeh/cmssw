#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdint>

template <size_t ALIGNMENT = 128>
struct MLSoA {

    // Helper function to compute aligned size
    constexpr inline size_t alignSize(size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    }

    static constexpr int alignment = ALIGNMENT;
    MLSoA(int elements, std::byte* mem) : elements_(elements), 
                                          intCol_(mem),
                                          floatCol_(mem + alignSize(elements_ * sizeof(int), alignment)),
                                          doubleCol_(floatCol_ + alignSize(elements_ * sizeof(float), alignment)),
                                          eigenCol_(doubleCol_ + alignSize(elements_ * sizeof(double), alignment)),
                                          scalarCol_(eigenCol_ + alignSize(elements_ * sizeof(Eigen::Vector3d::Scalar), alignment) * 3) {}

    enum class SoAColumnType {
        scalar = 0,
        column = 1,
        eigen = 2
    };

    private:
    
    int elements_ = 0;
    int* intCol_ = nullptr;
    float* floatCol_ = nullptr;
    double* doubleCol_ = nullptr;
    Eigen::Vector3d::Scalar* eigenCol_ = nullptr;
    int* scalarCol_ = nullptr;
};