namespace AlterEgo.CLI.Commands
{
    using System.ComponentModel;

    using Spectre.Console.Cli;

    /// <summary>
    /// Settings for the AutoML command.
    /// </summary>
    public sealed class AutoMLSettings : CommandSettings
    {
        [CommandOption("-d|--data <PATH>")]
        [Description("Path to the dataset file (default: housing.csv)")]
        public string? DataPath { get; init; }

        [CommandOption("-t|--timeout <SECONDS>")]
        [Description("Experiment timeout in seconds")]
        [DefaultValue(120u)]
        public uint Timeout { get; init; } = 120;

        [CommandOption("-s|--save")]
        [Description("Save the best model found as the default model")]
        public bool Save { get; init; }
    }
}
