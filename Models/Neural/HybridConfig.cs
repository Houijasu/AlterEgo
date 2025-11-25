namespace AlterEgo.Models.Neural
{
    using AlterEgo.Services.Neural;

    /// <summary>
    /// Type of Graph Neural Network layer to use.
    /// </summary>
    public enum GnnType
    {
        /// <summary>Graph Convolutional Network - simple, fast, uniform neighbor aggregation.</summary>
        GCN
    }

    /// <summary>
    /// Configuration for the CNN + GNN hybrid neural network.
    /// Optimized defaults for small tabular regression tasks.
    /// </summary>
    public record HybridConfig
    {
        // CNN Configuration - minimal for 1D regression
        public int[] CnnChannels { get; init; } = [8, 16];
        public int CnnKernel { get; init; } = 3;

        // GNN Configuration - lightweight
        public GnnType GnnType { get; init; } = GnnType.GCN;
        /// <summary>GNN input dimension - must match ExpandedFeatures since GNN receives expanded features directly.</summary>
        public int GnnInputDim => ExpandedFeatures;  // Computed to always match
        public int GnnHiddenDim { get; init; } = 16;
        public int GnnHeads { get; init; } = 2;
        public int GnnLayers { get; init; } = 1;
        public int GraphK { get; init; } = 10;

        // Graph Construction Strategy - hybrid for best structure
        public GraphType GraphType { get; init; } = GraphType.Hybrid;
        public int GraphBins { get; init; } = 8;

        // Feature Expansion - enhanced for better representation
        public int ExpandedFeatures { get; init; } = 16;
        public int RbfCenters { get; init; } = 8;
        public int SinFrequencies { get; init; } = 4;  // Fourier features

        // Training Configuration - longer training with patience
        public int Epochs { get; init; } = 2000;
        public float LearningRate { get; init; } = 5e-4f;
        public float WeightDecay { get; init; } = 1e-5f;
        public int EarlyStopPatience { get; init; } = 150;
        public float Dropout { get; init; } = 0.05f;
        public float GradientClip { get; init; } = 0.5f;

        // Learning rate scheduling
        public bool UseCosineAnnealing { get; init; } = true;
        public int WarmupEpochs { get; init; } = 20;

        // Stochastic Weight Averaging (SWA) for better generalization
        public bool UseSwa { get; init; } = true;
        public int SwaStartEpoch { get; init; } = 100;  // Start SWA after this epoch
        public int SwaUpdateFreq { get; init; } = 5;    // Average weights every N epochs

        // Data Split - use more data for training
        public float TrainRatio { get; init; } = 0.8f;
        public float ValRatio { get; init; } = 0.1f;

        // Loss weights - focus on MSE for regression
        public float MseLossWeight { get; init; } = 0.7f;
        public float HuberLossWeight { get; init; } = 0.2f;
        public float R2LossWeight { get; init; } = 0.1f;
        public float HuberDelta { get; init; } = 5000f;

        // Computed properties
        public int CnnOutputDim => CnnChannels[^1];
        public int FusionInputDim => CnnOutputDim + GnnHiddenDim;
    }

    /// <summary>
    /// Result from training the hybrid neural network.
    /// </summary>
    public record HybridTrainingResult(
        double FinalTrainLoss,
        double FinalValR2,
        double BestValR2,
        int BestEpoch,
        int TotalEpochs,
        TimeSpan TrainingTime);

    /// <summary>
    /// Metrics for evaluating the hybrid model.
    /// </summary>
    public record HybridMetrics(
        double RSquared,
        double RMSE,
        double MAE);
}
