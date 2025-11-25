namespace AlterEgo.Models.Neural.Interfaces;

using TorchSharp;
using static TorchSharp.torch;

/// <summary>
/// Interface for graph neural network layers, enabling swappable GNN implementations.
/// </summary>
public interface IGraphLayer : IDisposable
{
    /// <summary>
    /// Forward pass through the graph layer.
    /// </summary>
    /// <param name="nodeFeatures">Node feature matrix (N x D_in).</param>
    /// <param name="adjacencyMatrix">Normalized adjacency matrix (N x N).</param>
    /// <returns>Updated node features (N x D_out).</returns>
    Tensor Forward(Tensor nodeFeatures, Tensor adjacencyMatrix);

    /// <summary>
    /// Input feature dimension.
    /// </summary>
    int InputDim { get; }

    /// <summary>
    /// Output feature dimension.
    /// </summary>
    int OutputDim { get; }

    /// <summary>
    /// Move the layer to the specified device.
    /// </summary>
    void ToDevice(Device device);
}
