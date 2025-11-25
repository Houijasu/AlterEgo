namespace AlterEgo.Services.Neural.Layers;

using AlterEgo.Models.Neural.Interfaces;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Attention-based fusion layer that combines CNN and GNN outputs
/// using multi-head cross-attention mechanism.
/// </summary>
public sealed class AttentionFusion : Module<Tensor, Tensor, Tensor>, IFusionLayer
{
    private readonly int _numHeads;
    private readonly int _headDim;

    private readonly Linear _cnnProj;
    private readonly Linear _gnnProj;
    private readonly Linear _queryProj;
    private readonly Linear _keyProj;
    private readonly Linear _valueProj;
    private readonly Linear _outputProj;
    private readonly LayerNorm _layerNorm;
    private readonly Dropout _dropout;

    public int CnnDim { get; }
    public int GnnDim { get; }
    public int OutputDim { get; }

    public AttentionFusion(
        int cnnDim,
        int gnnDim,
        int outputDim,
        int numHeads = 4,
        float dropout = 0.1f,
        string name = "AttentionFusion") : base(name)
    {
        if (outputDim % numHeads != 0)
        {
            throw new ArgumentException(
                $"Output dimension ({outputDim}) must be divisible by number of heads ({numHeads})");
        }

        CnnDim = cnnDim;
        GnnDim = gnnDim;
        OutputDim = outputDim;
        _numHeads = numHeads;
        _headDim = outputDim / numHeads;

        // Project CNN and GNN to same dimension
        _cnnProj = Linear(cnnDim, outputDim);
        _gnnProj = Linear(gnnDim, outputDim);

        // Multi-head attention projections
        _queryProj = Linear(outputDim, outputDim);
        _keyProj = Linear(outputDim, outputDim);
        _valueProj = Linear(outputDim, outputDim);
        _outputProj = Linear(outputDim, outputDim);

        _layerNorm = LayerNorm(outputDim);
        _dropout = Dropout(dropout);

        InitializeWeights();
        RegisterComponents();
    }

    private void InitializeWeights()
    {
        foreach (var linear in new[] { _cnnProj, _gnnProj, _queryProj, _keyProj, _valueProj, _outputProj })
        {
            init.xavier_uniform_(linear.weight);
            if (linear.bias is not null)
            {
                init.zeros_(linear.bias);
            }
        }
    }

    public override Tensor forward(Tensor cnnFeatures, Tensor gnnFeatures)
    {
        // cnnFeatures: [N, cnnDim]
        // gnnFeatures: [N, gnnDim]

        var batchSize = cnnFeatures.shape[0];

        // Project to common dimension
        var cnnProj = _cnnProj.forward(cnnFeatures); // [N, outputDim]
        var gnnProj = _gnnProj.forward(gnnFeatures); // [N, outputDim]

        // Use CNN as query, GNN as key/value (cross-attention)
        var query = _queryProj.forward(cnnProj); // [N, outputDim]
        var key = _keyProj.forward(gnnProj);     // [N, outputDim]
        var value = _valueProj.forward(gnnProj); // [N, outputDim]

        // Reshape for multi-head attention: [N, numHeads, headDim]
        query = query.view(batchSize, _numHeads, _headDim);
        key = key.view(batchSize, _numHeads, _headDim);
        value = value.view(batchSize, _numHeads, _headDim);

        // Scaled dot-product attention
        var scale = MathF.Sqrt(_headDim);
        var scores = query.bmm(key.transpose(1, 2)) / scale; // [N, numHeads, numHeads]
        var attnWeights = functional.softmax(scores, dim: -1);
        attnWeights = _dropout.forward(attnWeights);

        // Apply attention to values
        var attended = attnWeights.bmm(value); // [N, numHeads, headDim]

        // Reshape back: [N, outputDim]
        attended = attended.view(batchSize, OutputDim);

        // Output projection
        var output = _outputProj.forward(attended);
        output = _dropout.forward(output);

        // Residual connection with CNN projection and layer norm
        output = _layerNorm.forward(output + cnnProj);

        return output;
    }

    /// <summary>
    /// Interface implementation for forward pass.
    /// </summary>
    Tensor IFusionLayer.Forward(Tensor cnnFeatures, Tensor gnnFeatures)
        => forward(cnnFeatures, gnnFeatures);

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
            _cnnProj.Dispose();
            _gnnProj.Dispose();
            _queryProj.Dispose();
            _keyProj.Dispose();
            _valueProj.Dispose();
            _outputProj.Dispose();
            _layerNorm.Dispose();
            _dropout.Dispose();
        }
        base.Dispose(disposing);
    }
}
