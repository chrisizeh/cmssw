# PyTorch Wrapper for C++ and Alpaka

The efficiently use PyTorch models on the GPU with SOA input, a producer is provided, which automatically converts an input SOA into multiple tensors, runs the model and stores the output in a predefined SOA. This is done by providing metadata of the input and output SOA to the producer. Then, using the raw byte buffer of the SOA, it is wrapped by a tensor object, without copying the data.

## Metadata

There a three types of Metadata, `InputMetadata`, `OutputMetadata` and `ModelMetadata`. The `ModelMetadata` consists of the `OutputMetadata` and `InputMetadata`, and the number of elements that are present. This number of elements are the number of rows in the SOA, which must be similar for Input and Output SOA.


## Example

```
GENERATE_SOA_LAYOUT(SoAInputTemplate,
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_COLUMN(int, t),
    SOA_COLUMN(float, phi),
    SOA_COLUMN(float, psi),
    SOA_COLUMN(float, theta)
    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, b))
```


```
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, type))
```


```
InputMetadata input({{torch::kDouble, torch::kInt, torch::kFloat, torch::kFloat}}, {{3, 1, 3, {2, 3}}}, {{1, -1, 2, 0}});
OutputMetadata output({torch::kInt, 1});
ModelMetadata metadata(batch_size, input, output);
```