namespace AlterEgo.Services.Neural.Layers;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// 1D Convolutional block with BatchNorm, ReLU, and Dropout.
/// Used for processing sequential/tabular features.
/// </summary>
public sealed class Conv1DBlock : Module<Tensor, Tensor>
{
    private readonly Conv1d _conv;
    private readonly BatchNorm1d _batchNorm;
    private readonly Dropout _dropout;

    public int InputChannels { get; }
    public int OutputChannels { get; }

    public Conv1DBlock(
        int inputChannels,
        int outputChannels,
        int kernelSize = 3,
        float dropout = 0.2f,
        string name = "Conv1DBlock") : base(name)
    {
        InputChannels = inputChannels;
        OutputChannels = outputChannels;

        // Same padding to preserve sequence length
        var padding = kernelSize / 2;

        _conv = Conv1d(inputChannels, outputChannels, kernelSize, padding: padding);
        _batchNorm = BatchNorm1d(outputChannels);
        _dropout = Dropout(dropout);

        InitializeWeights();
        RegisterComponents();
    }

    private void InitializeWeights()
    {
        // Kaiming initialization for ReLU activation
        init.kaiming_normal_(_conv.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
        if (_conv.bias is not null)
        {
            init.zeros_(_conv.bias);
        }
    }

    public override Tensor forward(Tensor input)
    {
        // input: (batch, channels, length)
        var x = _conv.forward(input);
        x = _batchNorm.forward(x);
        x = functional.relu(x);
        x = _dropout.forward(x);
        return x;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _conv.Dispose();
            _batchNorm.Dispose();
            _dropout.Dispose();
        }
        base.Dispose(disposing);
    }
}
