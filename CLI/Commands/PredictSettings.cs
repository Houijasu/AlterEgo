namespace AlterEgo.CLI.Commands
{
    using System.ComponentModel;

    using Spectre.Console.Cli;

    /// <summary>
    /// Settings for the predict command.
    /// </summary>
    public sealed class PredictSettings : CommandSettings
    {
        [CommandArgument(0, "[SIZE]")]
        [Description("House size in square feet")]
        public float? Size { get; init; }

        [CommandOption("-d|--data <PATH>")]
        [Description("Path to the dataset file (default: housing.csv)")]
        public string? DataPath { get; init; }

        [CommandOption("-r|--retrain")]
        [Description("Force model retraining")]
        public bool ForceRetrain { get; init; }

        [CommandOption("-m|--model <TYPE>")]
        [Description("Model type: mlnet (default) or hybrid")]
        [DefaultValue("mlnet")]
        public string ModelType { get; init; } = "mlnet";

        [CommandOption("--model-path <PATH>")]
        [Description("Path to hybrid model checkpoint (auto-detects latest if not specified)")]
        public string? ModelPath { get; init; }
    }
}
