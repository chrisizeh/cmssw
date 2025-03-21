# PyTorch Wrapper for C++ and Alpaka

The efficiently use PyTorch models on the GPU with SoA input, a producer is provided, which automatically converts an input SoA into multiple tensors, runs the model and stores the output in a predefined SoA. This is done by providing metadata of the input and output SoA to the producer. Then, using the raw byte buffer of the SoA, it is wrapped by a tensor object, without copying the data.

## Metadata

There are three types of Metadata, `InputMetadata`, `OutputMetadata` and `ModelMetadata`. The `ModelMetadata` consists of the `OutputMetadata` and `InputMetadata`, and the number of elements that are present. This number of elements are the number of rows in the SoA, which must be similar for Input and Output SoA.

To create a single tensor, the columns have to be of the same type. If the SoA contains columns of different types, it can be partitioned into blocks, with each block getting converted to a torch tensor. This is done, by passing to the constructor of the InputMetadata vectors (see example). 
For constructing of the metadata, the following info is needed for each block:

- Type: The torch type, which is associated to a C++ type. The following types are support by torch:
    - `Byte = torch::kByte`
    - `Char = torch::kChar`
    - `Short = torch::kShort`
    - `Int = torch::kInt`
    - `Long = torch::kLong`
    - `UInt16 = torch::kUInt16`
    - `UInt32 = torch::kUInt32`
    - `UInt64 = torch::kUInt64`
    - `Half = torch::kHalf`
    - `Float = torch::kFloat`
    - `Double = torch::kDouble`
- Columns: The number of columns associated to the block.
    - If the block is a scalar, columns have to be `0`, defining that it is not a column with `n` elements.
    - If the block is an eigen object, columns must be a vector of the number of columns and dimension of the object.\
    e.g. a block of two columns of `eigen::Matrix3d` has the columns value `{2, 3, 3}`.
- Optional: Ordering. If the blocks should be returned in a different order then in the SoA, or some block should be masked.
    - The value in the ordering defines the position in the resulting vector the corresponding tensor will be at.
    - To mask a block, the position in the ordering vector must be a `-1`.
    - e.g. {2, -1, 0, 1}, results in the following vector:
        - Block 3
        - Block 4
        - Block 1

## Example

SOA Template for Model Input:
```
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

SOA Template for Model Output:
```
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster))
```

Metadata Definition for converting to tensor:
```
InputMetadata input({Double, Float, Double, Float, Int}, {{{2, 3}}, {{1, 2, 2}}, 3, 0, 0});
OutputMetadata output(Int, 1);
ModelMetadata metadata(batch_size, input, output);
```