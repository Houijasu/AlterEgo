namespace AlterEgo.Services;

using AlterEgo.Models;
using AlterEgo.Models.Neural;
using AlterEgo.Services.Neural;

using Spectre.Console;

using TorchSharp;

using static TorchSharp.torch;

/// <summary>
/// High-level service for training and using the hybrid CNN+GCN model.
/// Supports single model and ensemble training for improved accuracy.
/// </summary>
public sealed class HybridModelService : IDisposable
{
    private readonly HybridConfig _config;
    private readonly Device _device;
    private HybridCnnGcnModel? _model;
    private HybridTrainer? _trainer;
    private bool _disposed;

    // Ensemble support
    private readonly List<HybridCnnGcnModel> _ensembleModels = [];
    private bool _isEnsemble;

    public HybridModelService(HybridConfig? config = null, bool forceCpu = false)
    {
        _config = config ?? new HybridConfig();

        // Auto-detect GPU availability unless CPU is forced
        _device = forceCpu ? CPU : (cuda.is_available() ? CUDA : CPU);

        AnsiConsole.MarkupLine($"[blue]Using device: {_device}[/]");
    }

    /// <summary>
    /// Initialize the model and trainer.
    /// </summary>
    public void Initialize()
    {
        AnsiConsole.MarkupLine("[grey]Creating model...[/]");
        _model = new HybridCnnGcnModel(_config);

        AnsiConsole.MarkupLine("[grey]Moving model to device...[/]");
        _model.to(_device);

        AnsiConsole.MarkupLine("[grey]Creating trainer...[/]");
        _trainer = new HybridTrainer(_model, _config, _device);

        AnsiConsole.MarkupLine("[grey]Initializing optimizer...[/]");
        _trainer.Initialize();

        AnsiConsole.MarkupLine($"[green]Model initialized with {_model.ParameterCount():N0} parameters[/]");
    }

    /// <summary>
    /// Train the model on house data from a CSV file.
    /// </summary>
    public HybridTrainingResult Train(string dataPath, bool verbose = true)
    {
        if (_model is null || _trainer is null)
        {
            throw new InvalidOperationException("Call Initialize() first");
        }

        // Load data
        var (sizes, prices) = LoadData(dataPath);

        if (verbose)
        {
            AnsiConsole.MarkupLine($"[grey]Loaded {sizes.Length} samples from {Path.GetFileName(dataPath)}[/]");
        }

        // Create graph data using configured strategy
        using var graphData = GraphBuilder.CreateGraphDataFromSizes(
            sizes, prices, _config.GnnInputDim, _config.GraphK, _device,
            _config.GraphType, _config.GraphBins);

        if (verbose)
        {
            var graphDesc = _config.GraphType switch
            {
                GraphType.Quantile => $"quantile graph with {_config.GraphBins} bins",
                GraphType.Hybrid => $"hybrid graph (k={_config.GraphK}, bins={_config.GraphBins})",
                _ => $"k-NN graph with k={_config.GraphK}"
            };
            AnsiConsole.MarkupLine($"[grey]Built {graphDesc}[/]");
        }

        // Split indices
        var (trainIndices, valIndices, _) = GraphBuilder.SplitIndices(
            graphData.NumNodes, _config.TrainRatio, _config.ValRatio);

        // Train
        return _trainer.Train(graphData, trainIndices, valIndices, verbose);
    }

    /// <summary>
    /// Train the model on provided arrays.
    /// </summary>
    public HybridTrainingResult Train(float[] sizes, float[] prices, bool verbose = true)
    {
        if (_model is null || _trainer is null)
        {
            throw new InvalidOperationException("Call Initialize() first");
        }

        // Create graph data using configured strategy
        using var graphData = GraphBuilder.CreateGraphDataFromSizes(
            sizes, prices, _config.GnnInputDim, _config.GraphK, _device,
            _config.GraphType, _config.GraphBins);

        // Split indices
        var (trainIndices, valIndices, _) = GraphBuilder.SplitIndices(
            graphData.NumNodes, _config.TrainRatio, _config.ValRatio);

        // Train
        return _trainer.Train(graphData, trainIndices, valIndices, verbose);
    }

    /// <summary>
    /// Train an ensemble of models for improved accuracy.
    /// Each model is trained with a different random seed.
    /// </summary>
    /// <param name="dataPath">Path to training data CSV.</param>
    /// <param name="numModels">Number of models in ensemble (default 5).</param>
    /// <param name="verbose">Show progress.</param>
    /// <returns>Best R² achieved across all models.</returns>
    public double TrainEnsemble(string dataPath, int numModels = 5, bool verbose = true)
    {
        if (verbose)
        {
            AnsiConsole.MarkupLine($"[blue]Training ensemble of {numModels} models...[/]");
        }

        var (sizes, prices) = LoadData(dataPath);
        var bestR2 = double.MinValue;

        for (var i = 0; i < numModels; i++)
        {
            if (verbose)
            {
                AnsiConsole.MarkupLine($"\n[cyan]═══ Ensemble Model {i + 1}/{numModels} ═══[/]");
            }

            // Set different random seed for each model
            manual_seed(42 + i * 1000);
            if (cuda.is_available())
            {
                cuda.manual_seed(42 + i * 1000);
            }

            // Create fresh model
            var model = new HybridCnnGcnModel(_config);
            model.to(_device);

            // Create trainer
            using var trainer = new HybridTrainer(model, _config, _device, $"checkpoints/ensemble_{i}");
            trainer.Initialize();

            // Create graph data
            using var graphData = GraphBuilder.CreateGraphDataFromSizes(
                sizes, prices, _config.GnnInputDim, _config.GraphK, _device,
                _config.GraphType, _config.GraphBins);

            var (trainIndices, valIndices, _) = GraphBuilder.SplitIndices(
                graphData.NumNodes, _config.TrainRatio, _config.ValRatio, seed: 42 + i);

            // Train
            var result = trainer.Train(graphData, trainIndices, valIndices, verbose);

            if (result.BestValR2 > bestR2)
            {
                bestR2 = result.BestValR2;
            }

            // Store model for ensemble prediction
            _ensembleModels.Add(model);
        }

        _isEnsemble = true;

        // Also keep first model as primary for backward compatibility
        _model = _ensembleModels[0];

        if (verbose)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]Ensemble training complete![/]");
            AnsiConsole.MarkupLine($"[blue]Best individual R²: {bestR2:F4}[/]");
            AnsiConsole.MarkupLine($"[cyan]Ensemble size: {_ensembleModels.Count} models[/]");
        }

        return bestR2;
    }

    /// <summary>
    /// Make predictions for new data.
    /// Uses ensemble averaging if trained with TrainEnsemble.
    /// </summary>
    public float[] Predict(float[] sizes, float[] existingSizes, float[] existingPrices)
    {
        var modelsToUse = _isEnsemble ? _ensembleModels : (_model is not null ? [_model] : []);

        if (modelsToUse.Count == 0)
        {
            throw new InvalidOperationException("Model not initialized or trained");
        }

        // Combine new sizes with existing data to build graph
        var allSizes = existingSizes.Concat(sizes).ToArray();
        var allPrices = existingPrices.Concat(new float[sizes.Length]).ToArray();

        using var noGrad = no_grad();
        using var graphData = GraphBuilder.CreateGraphDataFromSizes(
            allSizes, allPrices, _config.GnnInputDim, _config.GraphK, _device,
            _config.GraphType, _config.GraphBins);

        // Ensemble averaging
        var ensemblePreds = new float[sizes.Length];

        foreach (var model in modelsToUse)
        {
            model.eval();
            var predictions = model.forward(graphData.NodeFeatures, graphData.AdjacencyMatrix);

            // Extract predictions for new samples (last N)
            var allPreds = predictions.data<float>().ToArray();
            var rawPreds = allPreds[^sizes.Length..];

            // Accumulate predictions
            for (var i = 0; i < rawPreds.Length; i++)
            {
                ensemblePreds[i] += rawPreds[i];
            }
        }

        // Average and denormalize
        var finalPreds = new float[sizes.Length];
        for (var i = 0; i < sizes.Length; i++)
        {
            var avgPred = ensemblePreds[i] / modelsToUse.Count;
            finalPreds[i] = (avgPred * graphData.TargetStd) + graphData.TargetMean;
        }

        return finalPreds;
    }

    /// <summary>
    /// Make a single prediction.
    /// </summary>
    public float Predict(float size, float[] existingSizes, float[] existingPrices)
    {
        var predictions = Predict([size], existingSizes, existingPrices);
        return predictions[0];
    }

    /// <summary>
    /// Evaluate model performance using ensemble if available.
    /// </summary>
    public HybridMetrics Evaluate(float[] sizes, float[] prices)
    {
        var modelsToUse = _isEnsemble ? _ensembleModels : (_model is not null ? [_model] : []);

        if (modelsToUse.Count == 0)
        {
            throw new InvalidOperationException("Model not initialized");
        }

        using var noGrad = no_grad();
        using var graphData = GraphBuilder.CreateGraphDataFromSizes(
            sizes, prices, _config.GnnInputDim, _config.GraphK, _device,
            _config.GraphType, _config.GraphBins);

        // Ensemble averaging
        var ensemblePreds = new float[sizes.Length];

        foreach (var model in modelsToUse)
        {
            model.eval();
            var predictions = model.forward(graphData.NodeFeatures, graphData.AdjacencyMatrix);
            var predArray = predictions.squeeze().data<float>().ToArray();

            for (var i = 0; i < predArray.Length; i++)
            {
                ensemblePreds[i] += predArray[i];
            }
        }

        // Average and denormalize predictions
        var denormPreds = new float[sizes.Length];
        for (var i = 0; i < sizes.Length; i++)
        {
            var avgPred = ensemblePreds[i] / modelsToUse.Count;
            denormPreds[i] = (avgPred * graphData.TargetStd) + graphData.TargetMean;
        }

        var r2 = ComputeR2(denormPreds, prices);
        var rmse = ComputeRMSE(denormPreds, prices);
        var mae = ComputeMAE(denormPreds, prices);

        return new HybridMetrics(r2, rmse, mae);
    }

    /// <summary>
    /// Save the trained model.
    /// </summary>
    public void SaveModel(string path)
    {
        if (_model is null)
        {
            throw new InvalidOperationException("Model not initialized");
        }

        _model.save(path);
        AnsiConsole.MarkupLine($"[green]Model saved to {Markup.Escape(path)}[/]");
    }

    /// <summary>
    /// Load a model from disk.
    /// </summary>
    public void LoadModel(string path)
    {
        if (_model is null)
        {
            Initialize();
        }

        _model!.load(path);
        AnsiConsole.MarkupLine($"[green]Model loaded from {Markup.Escape(path)}[/]");
    }

    /// <summary>
    /// Export training metrics to CSV.
    /// </summary>
    public void ExportTrainingMetrics(string path)
    {
        _trainer?.ExportMetrics(path);
        AnsiConsole.MarkupLine($"[grey]Metrics exported to {Markup.Escape(path)}[/]");
    }

    internal static (float[] sizes, float[] prices) LoadData(string path)
    {
        var lines = File.ReadAllLines(path).Skip(1).ToArray(); // Skip header
        var sizes = new float[lines.Length];
        var prices = new float[lines.Length];

        for (var i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            sizes[i] = float.Parse(parts[0]);
            prices[i] = float.Parse(parts[1]);
        }

        return (sizes, prices);
    }

    private static double ComputeR2(float[] predictions, float[] targets)
    {
        var meanTarget = targets.Average();
        var ssRes = predictions.Zip(targets, (p, t) => Math.Pow(t - p, 2)).Sum();
        var ssTot = targets.Select(t => Math.Pow(t - meanTarget, 2)).Sum();
        return 1 - ssRes / (ssTot + 1e-8);
    }

    private static double ComputeRMSE(float[] predictions, float[] targets)
    {
        var mse = predictions.Zip(targets, (p, t) => Math.Pow(t - p, 2)).Average();
        return Math.Sqrt(mse);
    }

    private static double ComputeMAE(float[] predictions, float[] targets)
    {
        return predictions.Zip(targets, (p, t) => Math.Abs(t - p)).Average();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _trainer?.Dispose();

            // Dispose ensemble models
            foreach (var model in _ensembleModels)
            {
                model.Dispose();
            }
            _ensembleModels.Clear();

            // Only dispose _model if not in ensemble (ensemble already disposed it)
            if (!_isEnsemble)
            {
                _model?.Dispose();
            }

            _disposed = true;
        }
    }
}
