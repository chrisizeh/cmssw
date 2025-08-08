#include <cppunit/extensions/HelperMacros.h>
#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "PhysicsTools/PyTorch/test/NvtxScopedRange.h"
#include "PhysicsTools/PyTorch/test/testTorchBase.h"
#include "PhysicsTools/PyTorch/interface/Converter.h"

namespace torchtest {

  using namespace cms::torch;

  class TestSOADataTypes : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestSOADataTypes);
    CPPUNIT_TEST(testInterfaceVerbose);
    CPPUNIT_TEST(testMultiOutput);
    CPPUNIT_TEST(testSingleElement);
    CPPUNIT_TEST(testNoElement);
    CPPUNIT_TEST(testEmptyMetadata);
    CPPUNIT_TEST_SUITE_END();

  public:
    void testInterfaceVerbose();
    void testIncorrectMetadata();
    void testMultiOutput();
    void testSingleElement();
    void testNoElement();
    void testEmptyMetadata();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestSOADataTypes);

  GENERATE_SOA_LAYOUT(SoATemplate,
                      SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
                      SOA_EIGEN_COLUMN(Eigen::Vector3d, b),

                      SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),

                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),

                      SOA_SCALAR(float, type),
                      SOA_SCALAR(int, someNumber),

                      SOA_COLUMN(double, v),
                      SOA_COLUMN(double, w));

  using SoA = SoATemplate<>;
  using SoAView = SoA::View;
  using SoAMetaRecords = SoA::View::Metarecords;

  constexpr auto tol = 1.0e-5;

  void fill(SoAView &view) {
    view.type() = 4;
    view.someNumber() = 5;

    for (int i = 0; i < view.metadata().size(); i++) {
      view[i].a()(0) = 1 + i;
      view[i].a()(1) = 2 + i;
      view[i].a()(2) = 3 + i;

      view[i].b()(0) = 4 + i;
      view[i].b()(1) = 5 + i;
      view[i].b()(2) = 6 + i;

      view[i].c()(0, 0) = 4 + i;
      view[i].c()(0, 1) = 6 + i;
      view[i].c()(1, 0) = 8 + i;
      view[i].c()(1, 1) = 10 + i;

      view.x()[i] = 12 + i;
      view.y()[i] = 1 + 2.5 * i;
      view.z()[i] = 36 * i;
    }
  };

  void check(SoAView &view,
             ::torch::Tensor tensor_vector,
             ::torch::Tensor tensor_matrix,
             ::torch::Tensor tensor_column,
             ::torch::Tensor tensor_scalar) {
    for (int i = 0; i < view.metadata().size(); i++) {
      std::cout << view[i].a()(0) - tensor_vector[i][0][0] << std::endl;
      /*
        CPPUNIT_ASSERT((view[i].a()(0) - tensor_vector[i][0][0]) < tol);
        CPPUNIT_ASSERT(view[i].a()(1) - tensor_vector[i][0][1] < tol);
        CPPUNIT_ASSERT(view[i].a()(2) - tensor_vector[i][0][2] < tol);
        CPPUNIT_ASSERT(view[i].a()(0) - tensor_vector[i][0][0] > -tol);
        CPPUNIT_ASSERT(view[i].a()(1) - tensor_vector[i][0][1] > -tol);
        CPPUNIT_ASSERT(view[i].a()(2) - tensor_vector[i][0][2] > -tol);

        CPPUNIT_ASSERT(view[i].b()(0) - tensor_vector[i][1][0] < tol);
        CPPUNIT_ASSERT(view[i].b()(1) - tensor_vector[i][1][1] < tol);
        CPPUNIT_ASSERT(view[i].b()(2) - tensor_vector[i][1][2] < tol);
        CPPUNIT_ASSERT(view[i].b()(0) - tensor_vector[i][1][0] > -tol);
        CPPUNIT_ASSERT(view[i].b()(1) - tensor_vector[i][1][1] > -tol);
        CPPUNIT_ASSERT(view[i].b()(2) - tensor_vector[i][1][2] > -tol);

        CPPUNIT_ASSERT(view[i].c()(0, 0) - tensor_matrix[i][0][0][0] < tol);
        CPPUNIT_ASSERT(view[i].c()(0, 0) - tensor_matrix[i][0][0][0] > -tol);
        CPPUNIT_ASSERT(view[i].c()(0, 1) - tensor_matrix[i][0][0][1] < tol);
        CPPUNIT_ASSERT(view[i].c()(0, 1) - tensor_matrix[i][0][0][1] > -tol);
        CPPUNIT_ASSERT(view[i].c()(1, 0) - tensor_matrix[i][0][1][0] < tol);
        CPPUNIT_ASSERT(view[i].c()(1, 0) - tensor_matrix[i][0][1][0] > -tol);
        CPPUNIT_ASSERT(view[i].c()(1, 1) - tensor_matrix[i][0][1][1] < tol);
        CPPUNIT_ASSERT(view[i].c()(1, 1) - tensor_matrix[i][0][1][1] > -tol);

        CPPUNIT_ASSERT(view.x()[i] - tensor_column[i][0] < tol);
        CPPUNIT_ASSERT(view.x()[i] - tensor_column[i][0] > -tol);

        CPPUNIT_ASSERT(view.y()[i] - tensor_column[i][1] < tol);
        CPPUNIT_ASSERT(view.y()[i] - tensor_column[i][1] > -tol);

        CPPUNIT_ASSERT(view.z()[i] - tensor_column[i][2] < tol);
        CPPUNIT_ASSERT(view.z()[i] - tensor_column[i][2] > -tol);

        CPPUNIT_ASSERT(view.type() - tensor_scalar[i][0] < tol);
        CPPUNIT_ASSERT(view.type() - tensor_scalar[i][0] > -tol);
	*/
    };
  };

  void check_output(SoAView &view) {
    for (int i = 0; i < view.metadata().size(); i++) {
      CPPUNIT_ASSERT(view.x()[i] - view.v()[i] < tol);
      CPPUNIT_ASSERT(view.x()[i] - view.v()[i] > -tol);

      CPPUNIT_ASSERT(view.y()[i] - view.w()[i] < tol);
      CPPUNIT_ASSERT(view.y()[i] - view.w()[i] > -tol);
    }
  };

  void TestSOADataTypes::testInterfaceVerbose() {
    ::torch::Device torchDevice(::torch::kCPU);

    // Large batch size, so multiple bunches needed
    const std::size_t batch_size = 325;

    // Create and fill SoA layout
    const std::size_t slBufferSize = SoA::computeDataSize(batch_size);
    std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
    SoA sl{slBuffer.get(), batch_size};
    SoAView view{sl};
    fill(view);

    SoAMetaRecords records = view.records();
    SoAMetadata<SoA> input(batch_size);
    input.append_block("vector", records.a(), records.b());
    input.append_block("matrix", records.c());
    input.append_block("column", records.x(), records.y(), records.z());
    input.append_block("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector"});

    SoAMetadata<SoA> output(batch_size);
    output.append_block("result", records.v());
    ModelMetadata metadata(input, output);

    std::vector<::torch::IValue> tensors = Converter::convert_input(metadata, torchDevice);

    // Check if tensor list built correctly
    check(view, tensors[3].toTensor(), tensors[2].toTensor(), tensors[0].toTensor(), tensors[1].toTensor());
  };

  void TestSOADataTypes::testMultiOutput() {
    ::torch::Device torchDevice(::torch::kCPU);

    // Large batch size, so multiple bunches needed
    const std::size_t batch_size = 325;

    // Create and fill SoA layout
    const std::size_t slBufferSize = SoA::computeDataSize(batch_size);
    std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
    SoA sl{slBuffer.get(), batch_size};
    SoAView view{sl};
    fill(view);

    SoAMetaRecords records = view.records();
    SoAMetadata<SoA> input(batch_size);
    input.append_block("x", records.x());
    input.append_block("y", records.y());

    SoAMetadata<SoA> output(batch_size);
    output.append_block("v", records.v());
    output.append_block("w", records.w());
    ModelMetadata metadata(input, output);

    std::vector<::torch::IValue> tensors = Converter::convert_input(metadata, torchDevice);
    Converter::convert_output(tensors, metadata, torchDevice);

    // Check if tensor list built correctly
    check_output(view);
  };

  void TestSOADataTypes::testSingleElement() {
    ::torch::Device torchDevice(::torch::kCPU);

    // Create and fill SoA layout
    const std::size_t batch_size = 1;
    const std::size_t slBufferSize = SoA::computeDataSize(batch_size);
    std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
    SoA sl{slBuffer.get(), batch_size};
    SoAView view{sl};
    fill(view);

    // Run Converter for single tensor
    SoAMetaRecords records = view.records();
    SoAMetadata<SoA> input(batch_size);
    input.append_block("vector", records.a(), records.b());
    input.append_block("matrix", records.c());
    input.append_block("column", records.x(), records.y(), records.z());
    input.append_block("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector"});

    SoAMetadata<SoA> output(batch_size);
    output.append_block("result", records.v());
    ModelMetadata metadata(input, output);

    std::vector<::torch::IValue> tensors = Converter::convert_input(metadata, torchDevice);

    // Check if tensor list built correctly
    check(view, tensors[3].toTensor(), tensors[2].toTensor(), tensors[0].toTensor(), tensors[1].toTensor());
  };

  void TestSOADataTypes::testNoElement() {
    ::torch::Device torchDevice(::torch::kCPU);
    //Create empty SoA Layout
    const std::size_t batch_size = 0;
    const std::size_t slBufferSize = SoA::computeDataSize(batch_size);
    std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
    SoA sl{slBuffer.get(), batch_size};
    SoAView view{sl};
    SoAMetaRecords records = view.records();

    // Run Converter
    SoAMetadata<SoA> input(batch_size);
    input.append_block("vector", records.a(), records.b());
    input.append_block("matrix", records.c());
    input.append_block("column", records.x(), records.y(), records.z());
    input.append_block("scalar", records.type());
    input.change_order({"column", "scalar", "matrix", "vector"});

    SoAMetadata<SoA> output(batch_size);
    output.append_block("result", records.v());
    ModelMetadata metadata(input, output);

    std::vector<::torch::IValue> tensors = Converter::convert_input(metadata, torchDevice);

    // Check if tensor list has empty tensors
    CPPUNIT_ASSERT(tensors[0].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[1].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[2].toTensor().size(0) == 0);
    CPPUNIT_ASSERT(tensors[3].toTensor().size(0) == 0);
  };

  void TestSOADataTypes::testEmptyMetadata() {
    ::torch::Device torchDevice(::torch::kCPU);

    // Create and fill SoA layout 
    const std::size_t batch_size = 12;
    const std::size_t slBufferSize = SoA::computeDataSize(batch_size);
    std::unique_ptr<std::byte, decltype(std::free) *> slBuffer{
        reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, slBufferSize)), std::free};
    SoA sl{slBuffer.get(), batch_size};
    SoAView view{sl};
    fill(view);

    // Run Converter for empty metadata
    SoAMetadata<SoA> input(batch_size);
    SoAMetadata<SoA> output(batch_size);
    ModelMetadata metadata(input, output);

    std::vector<::torch::IValue> tensors = Converter::convert_input(metadata, torchDevice);

    // Check if tensor list is empty
    CPPUNIT_ASSERT(tensors.size() == 0);
  };

}  // namespace torchtest
