namespace AlterEgo.CLI.Commands
{
    using System.Globalization;
    using System.Text.RegularExpressions;

    using AlterEgo.Models.Neural;
    using AlterEgo.Services;

    using Microsoft.ML;

    using Spectre.Console;
    using Spectre.Console.Cli;

    /// <summary>
    /// Command for making house price predictions.
    /// </summary>
    public sealed class PredictCommand : Command<PredictSettings>
    {
        private const string ModelFileName = "house_price_model.zip";
        private const string DataFileName = "housing.csv";
        private const string CheckpointDir = "checkpoints";

        public override int Execute(CommandContext context, PredictSettings settings, CancellationToken cancellationToken)
        {
            var isHybrid = settings.ModelType.Equals("hybrid", StringComparison.OrdinalIgnoreCase);

            if (isHybrid)
            {
                return ExecuteHybridPrediction(settings);
            }

            return ExecuteMLNetPrediction(settings);
        }

        private static int ExecuteMLNetPrediction(PredictSettings settings)
        {
            AnsiConsole.Write(new Rule("[blue]ML.NET House Price Prediction[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var modelPath = Path.Combine(AppContext.BaseDirectory, ModelFileName);
                var dataPath = settings.DataPath ?? Path.Combine(AppContext.BaseDirectory, DataFileName);

                var modelService = new HousePriceModelService(mlContext);

                var model = modelService.LoadOrTrainModel(dataPath, modelPath, settings.ForceRetrain);
                if (model == null)
                {
                    return 1;
                }

                var requestedSize = ResolveRequestedSize(settings.Size);
                if (requestedSize < 0)
                {
                    return 1;
                }

                var predictedPrice = modelService.Predict(model, requestedSize);
                PrintPrediction(requestedSize, predictedPrice, "ML.NET");

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }
        }

        private static int ExecuteHybridPrediction(PredictSettings settings)
        {
            AnsiConsole.Write(new Rule("[purple]Hybrid CNN+GCN House Price Prediction[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var dataPath = settings.DataPath ?? Path.Combine(AppContext.BaseDirectory, DataFileName);

                // Find model checkpoint
                var modelPath = settings.ModelPath ?? FindLatestCheckpoint();
                if (modelPath is null)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No hybrid model checkpoint found.");
                    AnsiConsole.MarkupLine("[grey]Train a hybrid model first with:[/] [yellow]AlterEgo hybrid[/]");
                    AnsiConsole.MarkupLine("[grey]Or specify a checkpoint with:[/] [yellow]--model-path <path>[/]");
                    return 1;
                }

                if (!File.Exists(modelPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Model file not found: {Markup.Escape(modelPath)}");
                    return 1;
                }

                if (!File.Exists(dataPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Data file not found: {Markup.Escape(dataPath)}");
                    AnsiConsole.MarkupLine("[grey]The hybrid model requires training data for graph context.[/]");
                    return 1;
                }

                var requestedSize = ResolveRequestedSize(settings.Size);
                if (requestedSize < 0)
                {
                    return 1;
                }

                // Load training data for graph context (GNN needs neighbors)
                AnsiConsole.MarkupLine($"[grey]Loading graph context from {Markup.Escape(Path.GetFileName(dataPath))}...[/]");
                var (existingSizes, existingPrices) = HybridModelService.LoadData(dataPath);
                AnsiConsole.MarkupLine($"[grey]Loaded {existingSizes.Length} samples for graph construction[/]");

                // Create and load hybrid model
                var config = new HybridConfig();
                using var hybridService = new HybridModelService(config);

                AnsiConsole.MarkupLine($"[grey]Loading checkpoint: {Markup.Escape(Path.GetFileName(modelPath))}[/]");
                hybridService.LoadModel(modelPath);

                // Make prediction
                AnsiConsole.MarkupLine("[grey]Running inference...[/]");
                var predictedPrice = hybridService.Predict(requestedSize, existingSizes, existingPrices);

                PrintPrediction(requestedSize, predictedPrice, "Hybrid CNN+GCN");

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                if (ex.InnerException is not null)
                {
                    AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.InnerException.Message)}[/]");
                }
                return 1;
            }
        }

        private static string? FindLatestCheckpoint()
        {
            var checkpointPath = Path.Combine(AppContext.BaseDirectory, CheckpointDir);

            if (!Directory.Exists(checkpointPath))
            {
                return null;
            }

            // Pattern: hybrid_cnn_gcn_best_epoch0123_r20.8500.pt
            var checkpoints = Directory.GetFiles(checkpointPath, "hybrid_cnn_gcn_best_*.pt");

            if (checkpoints.Length == 0)
            {
                return null;
            }

            // Find the one with highest R² score (parsed from filename)
            var bestCheckpoint = checkpoints
                .Select(path =>
                {
                    var fileName = Path.GetFileName(path);
                    var match = Regex.Match(fileName, @"r2([\d.]+)\.pt$");
                    var r2 = match.Success ? double.Parse(match.Groups[1].Value) : 0;
                    return (path, r2);
                })
                .OrderByDescending(x => x.r2)
                .FirstOrDefault();

            return bestCheckpoint.path;
        }

        private static float ResolveRequestedSize(float? sizeFromArgs)
        {
            if (sizeFromArgs.HasValue && sizeFromArgs.Value > 0)
            {
                return sizeFromArgs.Value;
            }

            if (Console.IsInputRedirected)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] No size provided in non-interactive mode. Use [yellow]<SIZE>[/] argument.");
                return -1;
            }

            return AnsiConsole.Prompt(
                new TextPrompt<float>("Enter house size in [green]square feet[/]:")
                    .PromptStyle("yellow")
                    .ValidationErrorMessage("[red]Please enter a valid positive number[/]")
                    .Validate(size => size > 0
                        ? ValidationResult.Success()
                        : ValidationResult.Error("[red]Size must be positive[/]")));
        }

        private static void PrintPrediction(float size, float price, string modelType)
        {
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Property[/]")
                .AddColumn("[green]Value[/]");

            table.AddRow("Model", modelType);
            table.AddRow("House Size", $"{size:N0} sq ft");
            table.AddRow("Predicted Price", price.ToString("C0", CultureInfo.CurrentCulture));

            AnsiConsole.Write(new Panel(table)
                .Header("[yellow]Price Estimate[/]")
                .BorderColor(Color.Green));
        }
    }
}
