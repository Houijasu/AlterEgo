namespace AlterEgo.CLI.Commands;

using System.ComponentModel;

using Spectre.Console.Cli;

/// <summary>
/// Settings for the hybrid CNN+GNN command.
/// </summary>
public sealed class HybridSettings : CommandSettings
{
    [CommandOption("-d|--data <PATH>")]
    [Description("Path to the housing data CSV file")]
    public string? DataPath { get; init; }

    [CommandOption("-g|--gnn <TYPE>")]
    [Description("GNN type: GCN (default)")]
    [DefaultValue("GCN")]
    public string GnnType { get; init; } = "GCN";

    [CommandOption("-e|--epochs <COUNT>")]
    [Description("Number of training epochs")]
    [DefaultValue(2000)]
    public int Epochs { get; init; } = 2000;

    [CommandOption("-l|--learning-rate <RATE>")]
    [Description("Learning rate")]
    [DefaultValue(5e-4f)]
    public float LearningRate { get; init; } = 5e-4f;

    [CommandOption("-k|--graph-k <K>")]
    [Description("Number of neighbors for k-NN graph")]
    [DefaultValue(10)]
    public int GraphK { get; init; } = 10;

    [CommandOption("--gnn-hidden <DIM>")]
    [Description("GNN hidden dimension (reduce for less memory)")]
    [DefaultValue(16)]
    public int GnnHiddenDim { get; init; } = 16;

    [CommandOption("--gnn-heads <COUNT>")]
    [Description("Number of attention heads for fusion layer")]
    [DefaultValue(2)]
    public int GnnHeads { get; init; } = 2;

    [CommandOption("--dropout <RATE>")]
    [Description("Dropout rate")]
    [DefaultValue(0.05f)]
    public float Dropout { get; init; } = 0.05f;

    [CommandOption("--patience <EPOCHS>")]
    [Description("Early stopping patience")]
    [DefaultValue(150)]
    public int Patience { get; init; } = 150;

    [CommandOption("-v|--verbose")]
    [Description("Show detailed training progress")]
    public bool Verbose { get; init; } = true;

    [CommandOption("--force-cpu")]
    [Description("Force CPU mode even if GPU is available")]
    public bool ForceCpu { get; init; } = false;

    [CommandOption("-s|--save <PATH>")]
    [Description("Path to save the trained model")]
    public string? SavePath { get; init; }

    [CommandOption("--export-metrics <PATH>")]
    [Description("Path to export training metrics CSV")]
    public string? ExportMetricsPath { get; init; }

    [CommandOption("--ensemble <COUNT>")]
    [Description("Train an ensemble of N models for improved accuracy (default: disabled)")]
    public int? EnsembleCount { get; init; }
}
