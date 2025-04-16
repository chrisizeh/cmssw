# PyTorch Wrapper for C++ and Alpaka

The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data. This can be used directly with a PyTorch model. The result can also be dynamically placed into a SoA buffer.

## Metadata

The structual information of the input and output SoA are stored in an `SoAMetadata`. These two objects are then combined to a `ModelMetadata`, to be used by the `Converter`.

### Defining Metadata

The `SoAMetadata' can be defined by first initialising the object and then adding blocks to the metadata. Each block is transformed into a tensor whose size and type are derived from the columns provided.

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
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();

SoAMetadata<SoA> input(batch_size);
input.append_block("eigen_vector", records.a(), records.b());
input.append_block("eigen_matrix", records.c());
input.append_block("column", records.x(), records.y(), records.z());
input.append_block("scalar", view.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

SoAMetadata<SoA> output(batch_size);
output.append_block("result", result_view.cluster());
ModelMetadata metadata(input, output);
```

### Ordering of Blocks

The function `change_order()` in the allows specifying the order in which the blocks should be processed. The order should match the expected input configuration of the PyTorch model.