namespace AlterEgo.Services.Neural;

using AlterEgo.Models.Neural;
using AlterEgo.Models.Neural.Interfaces;
using AlterEgo.Services.Neural.Layers;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Hybrid CNN + GNN model for house price prediction.
/// Combines local feature extraction (CNN) with graph-based learning (GCN)
/// through attention-based fusion.
/// </summary>
public sealed class HybridCnnGcnModel : Module<Tensor, Tensor, Tensor>
{
    private readonly HybridConfig _config;

    // Feature expansion
    private readonly FeatureExpander _featureExpander;

    // CNN Branch
    private readonly List<Conv1DBlock> _cnnBlocks = [];
    private readonly AdaptiveAvgPool1d _cnnPool;

    // GNN Branch (GCN)
    private readonly List<Module<Tensor, Tensor, Tensor>> _gnnLayers = [];

    // Fusion
    private readonly AttentionFusion _fusion;

    // Output Head
    private readonly Sequential _outputHead;

    public HybridCnnGcnModel(HybridConfig config, string name = "HybridCnnGcnModel") : base(name)
    {
        _config = config;

        // Feature Expander (RBF + polynomial + sinusoidal)
        _featureExpander = new FeatureExpander(
            outputDim: config.ExpandedFeatures,
            rbfCenters: config.RbfCenters,
            sinFrequencies: config.SinFrequencies,
            dropout: config.Dropout * 0.5f,  // Light dropout in feature expansion
            name: "feature_expander");

        // CNN Branch: series of Conv1D blocks
        var cnnInChannels = 1; // Single channel for 1D conv

        for (var i = 0; i < config.CnnChannels.Length; i++)
        {
            var outChannels = config.CnnChannels[i];
            var block = new Conv1DBlock(
                inputChannels: cnnInChannels,
                outputChannels: outChannels,
                kernelSize: config.CnnKernel,
                dropout: config.Dropout,
                name: $"cnn_block_{i}");
            _cnnBlocks.Add(block);
            register_module($"cnn_block_{i}", block);
            cnnInChannels = outChannels;
        }

        _cnnPool = AdaptiveAvgPool1d(1);

        // GNN Branch: stacked GCN layers
        var gnnInDim = config.GnnInputDim;

        for (var i = 0; i < config.GnnLayers; i++)
        {
            var isLastLayer = i == config.GnnLayers - 1;
            var layer = new GraphConvLayer(
                inputDim: gnnInDim,
                outputDim: config.GnnHiddenDim,
                useBatchNorm: true,
                useActivation: !isLastLayer,
                dropout: config.Dropout,
                name: $"gcn_layer_{i}");
            _gnnLayers.Add(layer);
            register_module($"gnn_layer_{i}", layer);

            gnnInDim = config.GnnHiddenDim;
        }

        // Attention-based fusion
        _fusion = new AttentionFusion(
            cnnDim: config.CnnOutputDim,
            gnnDim: config.GnnHiddenDim,
            outputDim: config.FusionInputDim,
            numHeads: config.GnnHeads,
            dropout: config.Dropout,
            name: "attention_fusion");

        // Output regression head
        var headDim1 = config.FusionInputDim / 2;
        var headDim2 = headDim1 / 2;

        _outputHead = Sequential(
            ("fc1", Linear(config.FusionInputDim, headDim1)),
            ("bn1", BatchNorm1d(headDim1)),
            ("relu1", ReLU()),
            ("dropout1", Dropout(config.Dropout)),
            ("fc2", Linear(headDim1, headDim2)),
            ("relu2", ReLU()),
            ("dropout2", Dropout(config.Dropout / 2)),
            ("fc_out", Linear(headDim2, 1))
        );

        InitializeOutputHead();

        // Register remaining components
        register_module("feature_expander", _featureExpander);
        register_module("cnn_pool", _cnnPool);
        register_module("attention_fusion", _fusion);
        register_module("output_head", _outputHead);
    }

    private void InitializeOutputHead()
    {
        var modules = _outputHead.modules().ToArray();
        var linearCount = modules.Count(m => m is Linear);
        var currentLinear = 0;

        foreach (var module in modules)
        {
            if (module is Linear linear)
            {
                currentLinear++;

                if (currentLinear == linearCount)
                {
                    // Final regression layer: use Xavier with small gain for stable outputs
                    init.xavier_uniform_(linear.weight, gain: 0.1);
                }
                else
                {
                    // Hidden layers: Xavier uniform for balanced gradients
                    init.xavier_uniform_(linear.weight);
                }

                if (linear.bias is not null)
                {
                    init.zeros_(linear.bias);
                }
            }
        }
    }

    /// <summary>
    /// Forward pass through the hybrid model.
    /// </summary>
    /// <param name="nodeFeatures">Node features [N, D].</param>
    /// <param name="adjacencyMatrix">Normalized adjacency [N, N].</param>
    /// <returns>Predictions [N, 1].</returns>
    public override Tensor forward(Tensor nodeFeatures, Tensor adjacencyMatrix)
    {
        // Expand features for CNN and GNN input
        var expandedFeatures = _featureExpander.forward(nodeFeatures);

        // CNN Branch: [N, expanded] -> [N, 1, expanded] -> Conv1D -> [N, channels, 1] -> [N, channels]
        var cnnInput = expandedFeatures.unsqueeze(1); // Add channel dim

        foreach (var block in _cnnBlocks)
        {
            cnnInput = block.forward(cnnInput);
        }

        var cnnOutput = _cnnPool.forward(cnnInput).squeeze(-1); // [N, cnn_out_dim]

        // GNN Branch: [N, expanded] -> Graph layers -> [N, gnn_hidden]
        var gnnOutput = expandedFeatures;
        foreach (var layer in _gnnLayers)
        {
            gnnOutput = layer.forward(gnnOutput, adjacencyMatrix);
        }

        // Final activation for GNN output
        gnnOutput = functional.relu(gnnOutput);

        // Attention Fusion: combine CNN and GNN outputs
        var fused = _fusion.forward(cnnOutput, gnnOutput);

        // Output head: regression
        var predictions = _outputHead.forward(fused);

        return predictions;
    }

    /// <summary>
    /// Get total parameter count for logging.
    /// </summary>
    public long ParameterCount()
    {
        return parameters().Sum(p => p.numel());
    }

    /// <summary>
    /// Get trainable parameter count.
    /// </summary>
    public long TrainableParameterCount()
    {
        return parameters().Where(p => p.requires_grad).Sum(p => p.numel());
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _featureExpander.Dispose();
            foreach (var block in _cnnBlocks) block.Dispose();
            _cnnPool.Dispose();
            foreach (var layer in _gnnLayers) layer.Dispose();
            _fusion.Dispose();
            _outputHead.Dispose();
        }
        base.Dispose(disposing);
    }
}
