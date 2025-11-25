namespace AlterEgo.Services.Neural;

using AlterEgo.Models.Neural;

using TorchSharp;

using static TorchSharp.torch;

/// <summary>
/// Type of graph construction strategy.
/// </summary>
public enum GraphType
{
    /// <summary>k-NN based on feature distance (original behavior).</summary>
    KNN,
    /// <summary>Quantile-based: connects houses in same/adjacent size bins.</summary>
    Quantile,
    /// <summary>Hybrid: combines k-NN with quantile structure.</summary>
    Hybrid
}

/// <summary>
/// Utility for constructing graphs from tabular house data.
/// Supports multiple graph construction strategies.
/// </summary>
public static class GraphBuilder
{
    /// <summary>
    /// Builds a k-NN graph from a feature matrix.
    /// </summary>
    /// <param name="features">Feature matrix (N x D).</param>
    /// <param name="k">Number of nearest neighbors.</param>
    /// <param name="device">Target device.</param>
    /// <returns>Normalized adjacency matrix with self-loops.</returns>
    public static Tensor BuildKnnGraph(Tensor features, int k, Device device)
    {
        var n = (int)features.shape[0];
        k = Math.Min(k, n - 1);

        // Compute pairwise distances
        using var distances = ComputePairwiseDistances(features);

        // Get k-nearest neighbors for each node (including self)
        var (_, indices) = distances.topk(k + 1, dim: 1, largest: false);

        // Convert indices to CPU for iteration
        var indicesData = indices.data<long>().ToArray();
        indices.Dispose();

        // Build adjacency matrix
        var adjacencyData = new float[n * n];

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= k; j++)
            {
                var neighborIdx = (int)indicesData[i * (k + 1) + j];
                adjacencyData[i * n + neighborIdx] = 1.0f;
                adjacencyData[neighborIdx * n + i] = 1.0f; // Symmetric
            }
        }

        // Create tensor from array
        var adjacency = tensor(adjacencyData, new long[] { n, n }, dtype: ScalarType.Float32, device: device);

        // Normalize: D^(-1/2) * A * D^(-1/2)
        return NormalizeAdjacency(adjacency);
    }

    /// <summary>
    /// Normalizes adjacency matrix using symmetric normalization.
    /// A_norm = D^(-1/2) * A * D^(-1/2)
    /// </summary>
    public static Tensor NormalizeAdjacency(Tensor adjacency)
    {
        var degree = adjacency.sum(dim: 1);
        var degreeInvSqrt = degree.pow(-0.5f);

        // Handle divide by zero (isolated nodes)
        degreeInvSqrt = degreeInvSqrt.where(
            degreeInvSqrt.isfinite(),
            zeros_like(degreeInvSqrt));

        var degreeMatrix = diag(degreeInvSqrt);
        return degreeMatrix.mm(adjacency).mm(degreeMatrix);
    }

    /// <summary>
    /// Computes pairwise Euclidean distances between all rows.
    /// </summary>
    private static Tensor ComputePairwiseDistances(Tensor features)
    {
        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        var sqNorm = (features * features).sum(dim: 1, keepdim: true);
        var distances = sqNorm + sqNorm.t() - 2 * features.mm(features.t());
        return distances.clamp(min: 0); // Numerical stability
    }

    /// <summary>
    /// Builds a quantile-based graph where houses in same/adjacent size bins are connected.
    /// This creates meaningful market segment clusters for the GNN.
    /// </summary>
    /// <param name="sizes">House sizes (for binning).</param>
    /// <param name="numBins">Number of quantile bins (default 10).</param>
    /// <param name="device">Target device.</param>
    /// <returns>Normalized adjacency matrix.</returns>
    public static Tensor BuildQuantileGraph(float[] sizes, int numBins, Device device)
    {
        var n = sizes.Length;
        numBins = Math.Min(numBins, n); // Can't have more bins than samples

        // Compute quantile boundaries
        var sortedSizes = sizes.OrderBy(s => s).ToArray();
        var binEdges = new float[numBins + 1];
        binEdges[0] = float.MinValue;
        binEdges[numBins] = float.MaxValue;

        for (var i = 1; i < numBins; i++)
        {
            var percentile = (float)i / numBins;
            var idx = (int)(percentile * (n - 1));
            binEdges[i] = sortedSizes[idx];
        }

        // Assign each house to a bin
        var binAssignments = new int[n];
        for (var i = 0; i < n; i++)
        {
            for (var b = 0; b < numBins; b++)
            {
                if (sizes[i] >= binEdges[b] && sizes[i] < binEdges[b + 1])
                {
                    binAssignments[i] = b;
                    break;
                }
            }
            // Handle edge case for maximum value
            if (sizes[i] >= binEdges[numBins - 1])
            {
                binAssignments[i] = numBins - 1;
            }
        }

        // Build adjacency: connect houses in same bin (weight 1.0) and adjacent bins (weight 0.5)
        var adjacencyData = new float[n * n];

        for (var i = 0; i < n; i++)
        {
            for (var j = i; j < n; j++)
            {
                var binDiff = Math.Abs(binAssignments[i] - binAssignments[j]);

                float weight = binDiff switch
                {
                    0 => 1.0f,  // Same bin: strong connection
                    1 => 0.5f,  // Adjacent bin: medium connection
                    _ => 0.0f   // Far bins: no connection
                };

                if (weight > 0 || i == j)
                {
                    adjacencyData[i * n + j] = i == j ? 1.0f : weight; // Self-loop always 1
                    adjacencyData[j * n + i] = i == j ? 1.0f : weight; // Symmetric
                }
            }
        }

        var adjacency = tensor(adjacencyData, new long[] { n, n }, dtype: ScalarType.Float32, device: device);
        return NormalizeAdjacency(adjacency);
    }

    /// <summary>
    /// Builds a hybrid graph combining k-NN structure with quantile-based clustering.
    /// Uses k-NN for local structure and quantile bins for global market segments.
    /// </summary>
    public static Tensor BuildHybridGraph(Tensor features, float[] sizes, int k, int numBins, Device device)
    {
        var n = (int)features.shape[0];
        k = Math.Min(k, n - 1);
        numBins = Math.Min(numBins, n);

        // Get k-NN adjacency
        using var knnAdj = BuildKnnGraph(features, k, device);

        // Get quantile adjacency
        using var quantileAdj = BuildQuantileGraph(sizes, numBins, device);

        // Combine: average of both (both are normalized)
        var combined = (knnAdj * 0.5f + quantileAdj * 0.5f);

        // Re-normalize the combined matrix
        return NormalizeAdjacency(combined);
    }

    /// <summary>
    /// Creates GraphData from raw features and targets.
    /// </summary>
    /// <param name="features">Feature matrix (N x D).</param>
    /// <param name="targets">Target values (N,).</param>
    /// <param name="k">Number of nearest neighbors for graph construction.</param>
    /// <param name="device">Target device.</param>
    /// <returns>GraphData ready for model input with normalized targets.</returns>
    public static GraphData CreateGraphData(
        float[,] features,
        float[] targets,
        int k,
        Device device)
    {
        var n = features.GetLength(0);
        var d = features.GetLength(1);

        // Convert to tensors
        var featureData = new float[n * d];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < d; j++)
            {
                featureData[i * d + j] = features[i, j];
            }
        }

        var featureTensor = tensor(featureData, new long[] { n, d }, dtype: ScalarType.Float32, device: device);

        // Normalize targets (z-score normalization)
        var targetMean = targets.Average();
        var targetStd = (float)Math.Sqrt(targets.Select(t => Math.Pow(t - targetMean, 2)).Average());
        if (targetStd < 1e-8f) targetStd = 1f; // Prevent division by zero

        var normalizedTargets = targets.Select(t => (t - (float)targetMean) / targetStd).ToArray();
        var targetTensor = tensor(normalizedTargets, dtype: ScalarType.Float32, device: device).unsqueeze(1);

        // Build graph
        var adjacency = BuildKnnGraph(featureTensor, k, device);
        var edgeWeights = ones_like(adjacency); // Uniform weights

        return new GraphData(featureTensor, adjacency, edgeWeights, targetTensor, (float)targetMean, targetStd);
    }

    /// <summary>
    /// Creates GraphData from a single feature array (house sizes).
    /// Features and targets are normalized for better training stability.
    /// </summary>
    /// <param name="sizes">House sizes.</param>
    /// <param name="prices">House prices (targets).</param>
    /// <param name="gnnInputDim">Target feature dimension (unused, kept for compatibility).</param>
    /// <param name="k">Number of nearest neighbors.</param>
    /// <param name="device">Target device.</param>
    /// <param name="graphType">Graph construction strategy (default: Quantile for better GNN performance).</param>
    /// <param name="numBins">Number of quantile bins (for Quantile/Hybrid graph types).</param>
    /// <returns>GraphData with normalized features and targets.</returns>
    public static GraphData CreateGraphDataFromSizes(
        float[] sizes,
        float[] prices,
        int gnnInputDim,
        int k,
        Device device,
        GraphType graphType = GraphType.Quantile,
        int numBins = 10)
    {
        var n = sizes.Length;

        // Normalize sizes (z-score)
        var sizeMean = sizes.Average();
        var sizeStd = (float)Math.Sqrt(sizes.Select(s => Math.Pow(s - sizeMean, 2)).Average());
        if (sizeStd < 1e-8f) sizeStd = 1f;

        // Create normalized features
        var features = new float[n, 1];
        for (var i = 0; i < n; i++)
        {
            features[i, 0] = (sizes[i] - (float)sizeMean) / sizeStd;
        }

        // Create graph based on selected strategy
        return CreateGraphDataWithStrategy(features, sizes, prices, k, device, graphType, numBins);
    }

    /// <summary>
    /// Creates GraphData using the specified graph construction strategy.
    /// </summary>
    private static GraphData CreateGraphDataWithStrategy(
        float[,] features,
        float[] sizes,
        float[] targets,
        int k,
        Device device,
        GraphType graphType,
        int numBins)
    {
        var n = features.GetLength(0);
        var d = features.GetLength(1);

        // Convert features to tensor
        var featureData = new float[n * d];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < d; j++)
            {
                featureData[i * d + j] = features[i, j];
            }
        }
        var featureTensor = tensor(featureData, new long[] { n, d }, dtype: ScalarType.Float32, device: device);

        // Normalize targets (z-score normalization)
        var targetMean = targets.Average();
        var targetStd = (float)Math.Sqrt(targets.Select(t => Math.Pow(t - targetMean, 2)).Average());
        if (targetStd < 1e-8f) targetStd = 1f;

        var normalizedTargets = targets.Select(t => (t - (float)targetMean) / targetStd).ToArray();
        var targetTensor = tensor(normalizedTargets, dtype: ScalarType.Float32, device: device).unsqueeze(1);

        // Build graph based on strategy
        Tensor adjacency = graphType switch
        {
            GraphType.Quantile => BuildQuantileGraph(sizes, numBins, device),
            GraphType.Hybrid => BuildHybridGraph(featureTensor, sizes, k, numBins, device),
            _ => BuildKnnGraph(featureTensor, k, device) // KNN is default fallback
        };

        var edgeWeights = ones_like(adjacency);

        return new GraphData(featureTensor, adjacency, edgeWeights, targetTensor, (float)targetMean, targetStd);
    }

    /// <summary>
    /// Splits data indices into train/validation/test sets.
    /// </summary>
    /// <param name="numSamples">Total number of samples.</param>
    /// <param name="trainRatio">Training set ratio.</param>
    /// <param name="valRatio">Validation set ratio.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Tuple of (trainIndices, valIndices, testIndices).</returns>
    public static (int[] train, int[] val, int[] test) SplitIndices(
        int numSamples,
        float trainRatio,
        float valRatio,
        int seed = 42)
    {
        var indices = Enumerable.Range(0, numSamples).ToArray();
        var rng = new Random(seed);

        // Fisher-Yates shuffle
        for (var i = indices.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var trainEnd = (int)(numSamples * trainRatio);
        var valEnd = trainEnd + (int)(numSamples * valRatio);

        return (
            indices[..trainEnd],
            indices[trainEnd..valEnd],
            indices[valEnd..]
        );
    }

    /// <summary>
    /// Creates a boolean mask tensor from indices.
    /// </summary>
    /// <param name="numNodes">Total number of nodes.</param>
    /// <param name="indices">Indices to include in mask.</param>
    /// <param name="device">Target device.</param>
    /// <returns>Boolean mask tensor.</returns>
    public static Tensor CreateMask(int numNodes, int[] indices, Device device)
    {
        var mask = zeros(numNodes, dtype: ScalarType.Bool, device: device);
        foreach (var idx in indices)
        {
            mask[idx] = true;
        }
        return mask;
    }
}
