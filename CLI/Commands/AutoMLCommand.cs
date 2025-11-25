namespace AlterEgo.CLI.Commands
{
    using AlterEgo.Services;

    using Microsoft.ML;

    using Spectre.Console;
    using Spectre.Console.Cli;

    /// <summary>
    /// Command for running AutoML experiments.
    /// </summary>
    public sealed class AutoMLCommand : Command<AutoMLSettings>
    {
        private const string DataFileName = "housing.csv";
        private const string ModelFileName = "house_price_model.zip";

        public override int Execute(CommandContext context, AutoMLSettings settings, CancellationToken cancellationToken)
        {
            AnsiConsole.Write(new Rule("[blue]ML.NET AutoML Experiment[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var dataPath = settings.DataPath ?? Path.Combine(AppContext.BaseDirectory, DataFileName);
                var modelPath = Path.Combine(AppContext.BaseDirectory, ModelFileName);

                var autoMLService = new AutoMLExperimentService(mlContext);
                autoMLService.RunExperiment(dataPath, settings.Timeout, settings.Save ? modelPath : null);

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }
        }
    }
}
