namespace AlterEgo.Services.Neural.Layers;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Expands tabular features using RBF kernels, polynomial, and sinusoidal features.
/// Creates rich feature representations for better regression performance.
/// </summary>
public sealed class FeatureExpander : Module<Tensor, Tensor>
{
    private readonly int _outputDim;
    private readonly int _numRbfCenters;
    private readonly int _numSinFrequencies;
    private readonly Parameter _rbfCenters;
    private readonly Parameter _rbfWidths;
    private readonly Parameter _sinFrequencies;  // Learnable frequencies
    private readonly Linear _projection;
    private readonly Linear _projection2;  // Second projection for more capacity
    private readonly LayerNorm _layerNorm;
    private readonly Dropout _dropout;

    public int OutputDim => _outputDim;

    public FeatureExpander(
        int outputDim = 16,
        int rbfCenters = 8,
        int sinFrequencies = 4,
        float dropout = 0.0f,
        string name = "FeatureExpander") : base(name)
    {
        _outputDim = outputDim;
        _numRbfCenters = rbfCenters;
        _numSinFrequencies = sinFrequencies;

        // Learnable RBF centers initialized uniformly across input range
        _rbfCenters = Parameter(linspace(0f, 1f, rbfCenters));
        // Learnable widths initialized to cover the space
        _rbfWidths = Parameter(ones(rbfCenters) * (1f / rbfCenters));

        // Learnable sinusoidal frequencies (initialized to 1, 2, 4, 8 * pi)
        var initFreqs = new float[sinFrequencies];
        for (var i = 0; i < sinFrequencies; i++)
        {
            initFreqs[i] = MathF.Pow(2, i) * MathF.PI;
        }
        _sinFrequencies = Parameter(tensor(initFreqs));

        // Input dimension breakdown:
        // - original: 1
        // - RBF: rbfCenters
        // - polynomial: 5 (x^2, x^3, x^4, sqrt(x), log(x))
        // - sinusoidal: 2 * sinFrequencies (sin + cos for each frequency)
        // - piecewise: 3 (relu variants at different thresholds)
        var expandedDim = 1 + rbfCenters + 5 + (2 * sinFrequencies) + 3;

        // Two-layer projection for more expressiveness
        var hiddenDim = Math.Max(outputDim * 2, expandedDim);
        _projection = Linear(expandedDim, hiddenDim);
        _projection2 = Linear(hiddenDim, outputDim);
        _layerNorm = LayerNorm(outputDim);
        _dropout = Dropout(dropout);

        InitializeWeights();
        RegisterComponents();
    }

    private void InitializeWeights()
    {
        init.xavier_uniform_(_projection.weight);
        init.xavier_uniform_(_projection2.weight, gain: 0.5);  // Smaller for stability
        if (_projection.bias is not null) init.zeros_(_projection.bias);
        if (_projection2.bias is not null) init.zeros_(_projection2.bias);
    }

    public override Tensor forward(Tensor input)
    {
        Tensor x;
        if (input.dim() == 1)
        {
            x = input.unsqueeze(1);
        }
        else if (input.shape[1] > 1)
        {
            x = input[TensorIndex.Colon, TensorIndex.Slice(0, 1)];
        }
        else
        {
            x = input;
        }

        // Normalize input to [0, 1] range
        using var xMin = x.min();
        using var xMax = x.max();
        using var range = xMax - xMin + 1e-8f;
        var xNorm = (x - xMin) / range;

        var featureList = new List<Tensor> { xNorm };

        // 1. RBF features
        using (var xRepeated = xNorm.repeat(1, _numRbfCenters))
        using (var centers = _rbfCenters.unsqueeze(0).expand(xNorm.shape[0], -1))
        using (var diff = xRepeated - centers)
        using (var diffSq = diff.pow(2))
        using (var widths = _rbfWidths.unsqueeze(0).expand(xNorm.shape[0], -1))
        using (var widthsSq = widths.pow(2) * 2 + 1e-8f)
        {
            featureList.Add(exp(-diffSq / widthsSq));
        }

        // 2. Polynomial features
        featureList.Add(xNorm.pow(2));
        featureList.Add(xNorm.pow(3));
        featureList.Add(xNorm.pow(4));
        using (var xClamped = xNorm.clamp(min: 1e-8f))
        {
            featureList.Add(xClamped.sqrt());
            featureList.Add(xClamped.log());
        }

        // 3. Sinusoidal features (Fourier features for capturing periodic patterns)
        for (var i = 0; i < _numSinFrequencies; i++)
        {
            using var freq = _sinFrequencies[i];
            using var scaled = xNorm * freq;
            featureList.Add(sin(scaled));
            featureList.Add(cos(scaled));
        }

        // 4. Piecewise linear features (soft ReLU at different thresholds)
        featureList.Add(functional.softplus(xNorm - 0.25f, beta: 10));
        featureList.Add(functional.softplus(xNorm - 0.50f, beta: 10));
        featureList.Add(functional.softplus(xNorm - 0.75f, beta: 10));

        // Concatenate all features
        var expanded = cat(featureList, dim: 1);

        // Dispose individual feature tensors
        foreach (var feat in featureList)
        {
            feat.Dispose();
        }

        // Two-layer projection with GELU activation
        using var h1 = _projection.forward(expanded);
        expanded.Dispose();
        using var h1Act = functional.gelu(h1);
        using var h1Drop = _dropout.forward(h1Act);
        using var h2 = _projection2.forward(h1Drop);

        // Layer normalization
        var normalized = _layerNorm.forward(h2);

        return normalized;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _rbfCenters.Dispose();
            _rbfWidths.Dispose();
            _sinFrequencies.Dispose();
            _projection.Dispose();
            _projection2.Dispose();
            _layerNorm.Dispose();
            _dropout.Dispose();
        }
        base.Dispose(disposing);
    }
}
