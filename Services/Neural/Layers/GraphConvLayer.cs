namespace AlterEgo.Services.Neural.Layers;

using AlterEgo.Models.Neural.Interfaces;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Graph Convolutional Network layer implementing the GCN propagation rule:
/// H' = sigma(D^(-1/2) * A * D^(-1/2) * H * W)
/// where A includes self-loops.
/// </summary>
public sealed class GraphConvLayer : Module<Tensor, Tensor, Tensor>, IGraphLayer
{
    private readonly Linear _linear;
    private readonly BatchNorm1d? _batchNorm;
    private readonly Dropout _dropout;
    private readonly bool _useBatchNorm;
    private readonly bool _useActivation;

    public int InputDim { get; }
    public int OutputDim { get; }

    public GraphConvLayer(
        int inputDim,
        int outputDim,
        bool useBias = true,
        bool useBatchNorm = true,
        bool useActivation = true,
        float dropout = 0.0f,
        string name = "GraphConvLayer") : base(name)
    {
        InputDim = inputDim;
        OutputDim = outputDim;
        _useBatchNorm = useBatchNorm;
        _useActivation = useActivation;

        _linear = Linear(inputDim, outputDim, hasBias: useBias);
        InitializeWeights();

        if (useBatchNorm)
        {
            _batchNorm = BatchNorm1d(outputDim, momentum: 0.1, eps: 1e-5);
        }

        _dropout = Dropout(dropout);

        RegisterComponents();
    }

    private void InitializeWeights()
    {
        // Xavier/Glorot uniform initialization for training stability
        init.xavier_uniform_(_linear.weight);
        if (_linear.bias is not null)
        {
            init.zeros_(_linear.bias);
        }
    }

    /// <summary>
    /// Forward pass: H' = Activation(BatchNorm(A * X * W))
    /// </summary>
    /// <param name="nodeFeatures">Node features (N x InputDim).</param>
    /// <param name="adjacencyMatrix">Normalized adjacency matrix (N x N).</param>
    public override Tensor forward(Tensor nodeFeatures, Tensor adjacencyMatrix)
    {
        // Graph convolution: A * X
        var aggregated = adjacencyMatrix.mm(nodeFeatures);

        // Linear transformation: (A * X) * W + b
        var output = _linear.forward(aggregated);

        // Batch normalization (if enabled)
        if (_useBatchNorm && _batchNorm is not null)
        {
            output = _batchNorm.forward(output);
        }

        // Activation
        if (_useActivation)
        {
            output = functional.relu(output);
        }

        // Dropout
        output = _dropout.forward(output);

        return output;
    }

    /// <summary>
    /// Interface implementation for forward pass.
    /// </summary>
    Tensor IGraphLayer.Forward(Tensor nodeFeatures, Tensor adjacencyMatrix)
        => forward(nodeFeatures, adjacencyMatrix);

    /// <summary>
    /// Move layer to specified device.
    /// </summary>
    public void ToDevice(Device device)
    {
        this.to(device);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _linear.Dispose();
            _batchNorm?.Dispose();
            _dropout.Dispose();
        }
        base.Dispose(disposing);
    }
}
