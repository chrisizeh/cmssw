# PyTorch Wrapper for C++ and Alpaka

The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data. This can be used directly with a PyTorch model. The result can also be dynamically placed into a SoA buffer.

## Metadata

The structual information of the input and output SoA are stored in an `SoAMetadata`. These two objects are then combined to a `ModelMetadata`, to be used by the `Converter`.

### Defining Metadata

Metadata can be defined using either an automatic or explicit approach. The automatic approach deduces types from the provided pointers, while the explicit approach requires manually specifying types and structures.

#### Example SOA Template for Model Input:
```cpp
GENERATE_SOA_LAYOUT(SoATemplate,
    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
    SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_SCALAR(float, type),
    SOA_SCALAR(int, someNumber));
```

#### Example SOA Template for Model Output:
```cpp
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster));
```

#### Metadata Definition (Automatic Approach):
```cpp
PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(batch_size, queue);
fill(queue, deviceCollection);
auto view = deviceCollection.view();
auto result_view = deviceResultCollection.view();

SoAMetadata<SoA> input(batch_size);
input.append_block("vector", 2, view[0].a());
input.append_block("matrix", 1, view[0].c());
input.append_block("normal", 3, view.x());
input.append_block("scalar", view.type());
input.change_order({"normal", "scalar", "matrix", "vector"});

SoAMetadata<SoA> output(batch_size);
output.append_block("result", 1, result_view.cluster());
ModelMetadata metadata(input, output);
```

#### Metadata Definition (Explicit Approach):
```cpp
InputMetadata input({Double, Float, Double, Float, Int}, 
                    {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0}, 
                    {3, 2, 0, 1, -1});
OutputMetadata output(Int, 1);
ModelMetadata metadata(batch_size, input, output);
```

* The first vector `{Double, Float, Double, Float, Int}` defines the data types of the input blocks.
* The second vector specifies the structure of each block:
    * `{2, 3}` represents a block with two columns of eigen vectors with 3 values.    
    * `{1, 2, 2}` represents a block with one column of a 2x2 eigen matrix.
    * `3` represents a block with three columns in the tensor.
    * `0, 0` indicate two scalar values.
* The third vector is optional, defining the ordering, desribed below.

### Ordering of Blocks

The function `change_order()` in the allows specifying the order in which the blocks should be processed. The order should match the expected input configuration of the PyTorch model.

In the explicit approach, the order can be changed by providing a vector with the position of each block in the final tensor list. \
To mask a block, the value in the ordering vector must be set to -1.\
e.g. {2, -1, 0, 1}, results in the following order of the blocks:
Block 3, Block 4, Block 1. Block 2 is masked.