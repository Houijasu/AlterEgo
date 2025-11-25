namespace AlterEgo.Models.Neural
{
    using TorchSharp;
    using static TorchSharp.torch;

    /// <summary>
    /// Container for graph data used by the GNN.
    /// </summary>
    public class GraphData : IDisposable
    {
        /// <summary>
        /// Node feature matrix (N x D).
        /// </summary>
        public Tensor NodeFeatures { get; set; }

        /// <summary>
        /// Normalized adjacency matrix (N x N).
        /// </summary>
        public Tensor AdjacencyMatrix { get; set; }

        /// <summary>
        /// Edge weights matrix (N x N).
        /// </summary>
        public Tensor EdgeWeights { get; set; }

        /// <summary>
        /// Target values (N x 1), normalized.
        /// </summary>
        public Tensor Targets { get; set; }

        /// <summary>
        /// Mean of original targets for denormalization.
        /// </summary>
        public float TargetMean { get; set; }

        /// <summary>
        /// Standard deviation of original targets for denormalization.
        /// </summary>
        public float TargetStd { get; set; }

        /// <summary>
        /// Number of nodes in the graph.
        /// </summary>
        public int NumNodes => (int)NodeFeatures.shape[0];

        /// <summary>
        /// Feature dimension.
        /// </summary>
        public int FeatureDim => (int)NodeFeatures.shape[1];

        public GraphData(
            Tensor nodeFeatures,
            Tensor adjacencyMatrix,
            Tensor edgeWeights,
            Tensor targets,
            float targetMean = 0f,
            float targetStd = 1f)
        {
            NodeFeatures = nodeFeatures;
            AdjacencyMatrix = adjacencyMatrix;
            EdgeWeights = edgeWeights;
            Targets = targets;
            TargetMean = targetMean;
            TargetStd = targetStd;
        }

        public void Dispose()
        {
            NodeFeatures?.Dispose();
            AdjacencyMatrix?.Dispose();
            EdgeWeights?.Dispose();
            Targets?.Dispose();
        }

        /// <summary>
        /// Move all tensors to the specified device.
        /// </summary>
        public GraphData To(Device device)
        {
            return new GraphData(
                NodeFeatures.to(device),
                AdjacencyMatrix.to(device),
                EdgeWeights.to(device),
                Targets.to(device),
                TargetMean,
                TargetStd);
        }
    }
}
