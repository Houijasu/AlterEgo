namespace AlterEgo.Models.Neural.Interfaces;

using TorchSharp;
using static TorchSharp.torch;

/// <summary>
/// Interface for fusion layers that combine CNN and GNN outputs.
/// </summary>
public interface IFusionLayer : IDisposable
{
    /// <summary>
    /// Fuse two feature tensors.
    /// </summary>
    /// <param name="cnnFeatures">Features from CNN branch (batch x D_cnn).</param>
    /// <param name="gnnFeatures">Features from GNN branch (batch x D_gnn).</param>
    /// <returns>Fused features (batch x D_out).</returns>
    Tensor Forward(Tensor cnnFeatures, Tensor gnnFeatures);

    /// <summary>
    /// CNN input dimension.
    /// </summary>
    int CnnDim { get; }

    /// <summary>
    /// GNN input dimension.
    /// </summary>
    int GnnDim { get; }

    /// <summary>
    /// Output dimension after fusion.
    /// </summary>
    int OutputDim { get; }

    /// <summary>
    /// Move the layer to the specified device.
    /// </summary>
    void ToDevice(Device device);
}
