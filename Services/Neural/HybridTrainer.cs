namespace AlterEgo.Services.Neural;

using AlterEgo.Models.Neural;

using Spectre.Console;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

/// <summary>
/// Training loop with gradient clipping, LR scheduling, early stopping, and checkpointing.
/// </summary>
public sealed class HybridTrainer : IDisposable
{
    private readonly HybridConfig _config;
    private readonly HybridCnnGcnModel _model;
    private readonly Device _device;
    private readonly string _checkpointDir;

    private OptimizerHelper? _optimizer;
    private bool _disposed;

    // Training state
    private double _bestValR2 = double.MinValue;
    private int _bestEpoch;
    private string? _bestCheckpointPath;

    // Metrics history
    private readonly List<EpochMetrics> _history = [];

    // SWA (Stochastic Weight Averaging) state
    private readonly List<Dictionary<string, Tensor>> _swaWeights = [];
    private bool _swaStarted;

    public IReadOnlyList<EpochMetrics> History => _history;

    public HybridTrainer(
        HybridCnnGcnModel model,
        HybridConfig config,
        Device device,
        string checkpointDir = "checkpoints")
    {
        _model = model;
        _config = config;
        _device = device;
        _checkpointDir = checkpointDir;

        if (!Directory.Exists(checkpointDir))
        {
            Directory.CreateDirectory(checkpointDir);
        }
    }

    /// <summary>
    /// Initialize optimizer. Call before training.
    /// </summary>
    public void Initialize()
    {
        // AdamW with weight decay for regularization
        _optimizer = AdamW(
            _model.parameters(),
            lr: _config.LearningRate,
            weight_decay: _config.WeightDecay);
    }

    /// <summary>
    /// Train the model with full robustness features.
    /// </summary>
    public HybridTrainingResult Train(
        GraphData graphData,
        int[] trainIndices,
        int[] valIndices,
        bool verbose = true)
    {
        if (_optimizer is null)
        {
            throw new InvalidOperationException("Call Initialize() before training");
        }

        var startTime = DateTime.UtcNow;
        var patienceCounter = 0;
        var finalTrainLoss = 0.0;
        var finalValR2 = 0.0;

        // Move data to device
        using var dataOnDevice = graphData.To(_device);

        var trainMask = GraphBuilder.CreateMask(graphData.NumNodes, trainIndices, _device);
        var valMask = GraphBuilder.CreateMask(graphData.NumNodes, valIndices, _device);

        if (verbose)
        {
            AnsiConsole.MarkupLine($"[blue]Starting training for {_config.Epochs} epochs...[/]");
            AnsiConsole.MarkupLine($"[grey]Model parameters: {_model.ParameterCount():N0}[/]");
            AnsiConsole.MarkupLine($"[grey]Train samples: {trainIndices.Length}, Validation samples: {valIndices.Length}[/]");
            AnsiConsole.WriteLine();
        }

        var warmupEpochs = _config.WarmupEpochs;
        var actualEpochs = 0;

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            actualEpochs = epoch + 1;

            // Update learning rate (warmup + cosine annealing)
            var lr = GetLearningRate(epoch, warmupEpochs);
            SetLearningRate(lr);

            // Training step
            _model.train();
            var trainLoss = TrainEpoch(dataOnDevice, trainMask);
            finalTrainLoss = trainLoss;

            // Validation step
            _model.eval();
            var (valLoss, valR2) = ValidateEpoch(dataOnDevice, valMask);
            finalValR2 = valR2;

            // Track metrics
            _history.Add(new EpochMetrics(epoch, trainLoss, valLoss, valR2, lr));

            // Early stopping check
            if (valR2 > _bestValR2)
            {
                _bestValR2 = valR2;
                _bestEpoch = epoch;
                patienceCounter = 0;

                // Save best model checkpoint
                SaveCheckpoint(epoch, valR2, valLoss);
            }
            else
            {
                patienceCounter++;
            }

            // SWA: Collect weight snapshots after SwaStartEpoch
            if (_config.UseSwa && epoch >= _config.SwaStartEpoch &&
                (epoch - _config.SwaStartEpoch) % _config.SwaUpdateFreq == 0)
            {
                CollectSwaWeights();
                _swaStarted = true;
            }

            // Log progress periodically
            if (verbose && (epoch % 10 == 0 || epoch == _config.Epochs - 1 || patienceCounter >= _config.EarlyStopPatience))
            {
                var marker = valR2 >= _bestValR2 ? "[green]★[/]" : " ";
                AnsiConsole.MarkupLine(
                    $"{marker} Epoch {epoch + 1,4} | " +
                    $"Loss: [yellow]{trainLoss:F4}[/] | " +
                    $"Val R²: [cyan]{valR2:F4}[/] | " +
                    $"LR: [grey]{lr:E2}[/] | " +
                    $"Best: [green]{_bestValR2:F4}[/]");
            }

            // Early stopping
            if (patienceCounter >= _config.EarlyStopPatience)
            {
                if (verbose)
                {
                    AnsiConsole.MarkupLine(
                        $"[yellow]Early stopping at epoch {epoch + 1} " +
                        $"(no improvement for {_config.EarlyStopPatience} epochs)[/]");
                }
                break;
            }
        }

        // Apply SWA if weights were collected and compare with best checkpoint
        var usedSwa = false;
        if (_swaStarted && _swaWeights.Count > 0)
        {
            // Apply SWA averaged weights
            ApplySwaWeights();

            // Evaluate SWA model
            _model.eval();
            var (_, swaR2) = ValidateEpoch(dataOnDevice, valMask);

            if (verbose)
            {
                AnsiConsole.MarkupLine($"[cyan]SWA R²: {swaR2:F4} (averaged from {_swaWeights.Count} snapshots)[/]");
            }

            // Keep SWA weights if better, otherwise load best checkpoint
            if (swaR2 > _bestValR2)
            {
                _bestValR2 = swaR2;
                usedSwa = true;
                if (verbose)
                {
                    AnsiConsole.MarkupLine("[green]Using SWA weights (better than best checkpoint)[/]");
                }
            }
            else
            {
                LoadBestCheckpoint();
                if (verbose)
                {
                    AnsiConsole.MarkupLine("[yellow]Keeping best checkpoint (better than SWA)[/]");
                }
            }
        }
        else
        {
            // Load best model
            LoadBestCheckpoint();
        }

        var trainingTime = DateTime.UtcNow - startTime;

        if (verbose)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]Training complete![/]");
            AnsiConsole.MarkupLine($"[blue]Best R²: {_bestValR2:F4} at epoch {_bestEpoch + 1}{(usedSwa ? " (SWA)" : "")}[/]");
            AnsiConsole.MarkupLine($"[grey]Training time: {trainingTime:hh\\:mm\\:ss}[/]");
        }

        return new HybridTrainingResult(
            finalTrainLoss,
            finalValR2,
            _bestValR2,
            _bestEpoch,
            actualEpochs,
            trainingTime);
    }

    private double TrainEpoch(GraphData data, Tensor mask)
    {
        _optimizer!.zero_grad();

        // Forward pass
        var predictions = _model.forward(data.NodeFeatures, data.AdjacencyMatrix);

        // Compute loss (only on training nodes)
        var loss = ComputeCombinedLoss(predictions, data.Targets, mask);

        // Backward pass
        loss.backward();

        // Gradient clipping for stability
        nn.utils.clip_grad_norm_(_model.parameters(), _config.GradientClip);

        // Optimizer step
        _optimizer.step();

        return loss.item<float>();
    }

    private (double loss, double r2) ValidateEpoch(GraphData data, Tensor mask)
    {
        using var _ = no_grad();

        var predictions = _model.forward(data.NodeFeatures, data.AdjacencyMatrix);

        var loss = ComputeCombinedLoss(predictions, data.Targets, mask);
        var r2 = ComputeR2(predictions, data.Targets, mask);

        return (loss.item<float>(), r2);
    }

    private Tensor ComputeCombinedLoss(Tensor predictions, Tensor targets, Tensor mask)
    {
        // Apply mask to get only relevant predictions
        var maskIndices = mask.nonzero().squeeze();
        var maskedPred = predictions.index_select(0, maskIndices);
        var maskedTarget = targets.index_select(0, maskIndices);

        // Combined loss: MSE + Huber + R2-based
        var mseLoss = functional.mse_loss(maskedPred, maskedTarget);
        var huberLoss = functional.huber_loss(
            maskedPred, maskedTarget,
            reduction: Reduction.Mean,
            delta: _config.HuberDelta);

        // R2-based loss component (1 - R2)
        var ssRes = (maskedTarget - maskedPred).pow(2).sum();
        var ssTot = (maskedTarget - maskedTarget.mean()).pow(2).sum();
        var r2Loss = ssRes / (ssTot + 1e-8f);

        var combinedLoss =
            _config.MseLossWeight * mseLoss +
            _config.HuberLossWeight * huberLoss +
            _config.R2LossWeight * r2Loss;

        return combinedLoss;
    }

    private double ComputeR2(Tensor predictions, Tensor targets, Tensor mask)
    {
        var maskIndices = mask.nonzero().squeeze();
        var maskedPred = predictions.index_select(0, maskIndices);
        var maskedTarget = targets.index_select(0, maskIndices);

        var ssRes = (maskedTarget - maskedPred).pow(2).sum();
        var ssTot = (maskedTarget - maskedTarget.mean()).pow(2).sum();

        var r2 = 1.0 - (ssRes / (ssTot + 1e-8f)).item<float>();
        return r2;
    }

    private double GetLearningRate(int epoch, int warmupEpochs)
    {
        var baseLr = _config.LearningRate;
        var minLr = baseLr * 0.01;

        if (epoch < warmupEpochs)
        {
            // Linear warmup
            return baseLr * (epoch + 1) / warmupEpochs;
        }

        if (!_config.UseCosineAnnealing)
        {
            return baseLr;
        }

        // Cosine annealing with warm restarts style decay
        var progress = (double)(epoch - warmupEpochs) / (_config.Epochs - warmupEpochs);
        return minLr + 0.5 * (baseLr - minLr) * (1 + Math.Cos(Math.PI * progress));
    }

    private void SetLearningRate(double lr)
    {
        foreach (var paramGroup in _optimizer!.ParamGroups)
        {
            paramGroup.LearningRate = lr;
        }
    }

    private void SaveCheckpoint(int epoch, double r2, double loss)
    {
        // Delete previous best if exists
        if (_bestCheckpointPath is not null && File.Exists(_bestCheckpointPath))
        {
            try
            {
                File.Delete(_bestCheckpointPath);
            }
            catch
            {
                // Ignore deletion errors
            }
        }

        _bestCheckpointPath = Path.Combine(
            _checkpointDir,
            $"hybrid_cnn_gcn_best_epoch{epoch:D4}_r2{r2:F4}.pt");

        _model.save(_bestCheckpointPath);
    }

    private void LoadBestCheckpoint()
    {
        if (_bestCheckpointPath is null || !File.Exists(_bestCheckpointPath))
        {
            return;
        }

        _model.load(_bestCheckpointPath);
    }

    /// <summary>
    /// Collects current model weights for SWA averaging.
    /// </summary>
    private void CollectSwaWeights()
    {
        var snapshot = new Dictionary<string, Tensor>();
        foreach (var (name, param) in _model.named_parameters())
        {
            // Clone the parameter to CPU for storage (detach from graph)
            snapshot[name] = param.detach().clone().cpu();
        }
        _swaWeights.Add(snapshot);
    }

    /// <summary>
    /// Applies averaged SWA weights to the model.
    /// </summary>
    private void ApplySwaWeights()
    {
        if (_swaWeights.Count == 0) return;

        using var noGrad = no_grad();

        // Average all collected weights
        foreach (var (name, param) in _model.named_parameters())
        {
            // Start with zeros on device
            var avgWeight = zeros_like(param);

            // Sum all snapshots
            foreach (var snapshot in _swaWeights)
            {
                using var snapshotOnDevice = snapshot[name].to(_device);
                avgWeight.add_(snapshotOnDevice);
            }

            // Divide by count to get average
            avgWeight.div_(_swaWeights.Count);

            // Copy averaged weights to model parameter
            param.copy_(avgWeight);
            avgWeight.Dispose();
        }
    }

    /// <summary>
    /// Export training metrics to CSV.
    /// </summary>
    public void ExportMetrics(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("Epoch,TrainLoss,ValLoss,ValR2,LearningRate");

        foreach (var m in _history)
        {
            writer.WriteLine($"{m.Epoch},{m.TrainLoss:F6},{m.ValLoss:F6},{m.ValR2:F6},{m.LearningRate:E4}");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _optimizer?.Dispose();

            // Dispose SWA weight snapshots
            foreach (var snapshot in _swaWeights)
            {
                foreach (var tensor in snapshot.Values)
                {
                    tensor.Dispose();
                }
            }
            _swaWeights.Clear();

            _disposed = true;
        }
    }
}

/// <summary>
/// Metrics for a single training epoch.
/// </summary>
public readonly record struct EpochMetrics(
    int Epoch,
    double TrainLoss,
    double ValLoss,
    double ValR2,
    double LearningRate);
