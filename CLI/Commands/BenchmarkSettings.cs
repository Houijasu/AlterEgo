namespace AlterEgo.CLI.Commands
{
    using System.ComponentModel;

    using Spectre.Console.Cli;

    /// <summary>
    /// Settings for the benchmark command.
    /// </summary>
    public sealed class BenchmarkSettings : CommandSettings
    {
        [CommandOption("-d|--data <PATH>")]
        [Description("Path to the dataset file (default: housing.csv)")]
        public string? DataPath { get; init; }

        [CommandOption("-i|--iterations <COUNT>")]
        [Description("Number of cross-validation folds")]
        [DefaultValue(5)]
        public int Iterations { get; init; } = 5;

        [CommandOption("-t|--timeout <SECONDS>")]
        [Description("AutoML timeout in seconds (default: 60)")]
        [DefaultValue(60u)]
        public uint Timeout { get; init; } = 60;
    }
}
