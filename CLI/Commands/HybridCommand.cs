namespace AlterEgo.CLI.Commands;

using AlterEgo.Models.Neural;
using AlterEgo.Services;

using Spectre.Console;
using Spectre.Console.Cli;

/// <summary>
/// Command for training the hybrid CNN+GCN model.
/// </summary>
public sealed class HybridCommand : Command<HybridSettings>
{
    private const string DataFileName = "housing.csv";

    public override int Execute(CommandContext context, HybridSettings settings, CancellationToken cancellationToken)
    {
        AnsiConsole.Write(new FigletText("Hybrid CNN+GNN")
            .Color(Color.Purple)
            .Centered());

        AnsiConsole.Write(new Rule("[grey]Neural Network House Price Prediction[/]")
            .RuleStyle("purple"));
        AnsiConsole.WriteLine();

        try
        {
            var dataPath = settings.DataPath ?? Path.Combine(AppContext.BaseDirectory, DataFileName);

            if (!File.Exists(dataPath))
            {
                AnsiConsole.MarkupLine($"[red]Error: Data file not found at {Markup.Escape(dataPath)}[/]");
                return 1;
            }

            // GNN type (currently only GCN is supported)
            var gnnType = GnnType.GCN;

            // Create configuration from settings
            var config = new HybridConfig
            {
                GnnType = gnnType,
                Epochs = settings.Epochs,
                LearningRate = settings.LearningRate,
                GraphK = settings.GraphK,
                GnnHiddenDim = settings.GnnHiddenDim,
                GnnHeads = settings.GnnHeads,
                Dropout = settings.Dropout,
                EarlyStopPatience = settings.Patience
            };

            // Display configuration
            DisplayConfig(config, dataPath);

            // Initialize and train
            using var service = new HybridModelService(config, settings.ForceCpu);

            AnsiConsole.WriteLine();
            service.Initialize();
            AnsiConsole.WriteLine();

            HybridTrainingResult? result = null;
            double? ensembleBestR2 = null;

            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("purple"))
                .Start("[purple]Preparing training...[/]", ctx =>
                {
                    // Status updates handled inside Train method
                    ctx.Status("[purple]Training in progress...[/]");
                });

            // Use ensemble training if requested
            if (settings.EnsembleCount.HasValue && settings.EnsembleCount > 1)
            {
                ensembleBestR2 = service.TrainEnsemble(dataPath, settings.EnsembleCount.Value, settings.Verbose);
            }
            else
            {
                result = service.Train(dataPath, settings.Verbose);
            }

            // Display results
            if (result is not null)
            {
                DisplayResults(result);
            }
            else if (ensembleBestR2.HasValue)
            {
                DisplayEnsembleResults(ensembleBestR2.Value, settings.EnsembleCount!.Value);
            }

            // Save model if requested
            if (settings.SavePath is not null)
            {
                service.SaveModel(settings.SavePath);
            }

            // Export metrics if requested
            if (settings.ExportMetricsPath is not null)
            {
                service.ExportTrainingMetrics(settings.ExportMetricsPath);
            }

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            if (ex.InnerException is not null)
            {
                AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.InnerException.Message)}[/]");
            }
            return 1;
        }
    }

    private static void DisplayConfig(HybridConfig config, string dataPath)
    {
        var gnnTypeStr = "GCN (Graph Convolution)";

        var panel = new Panel(
            new Markup(
                $"[grey]Data file:[/] [yellow]{Markup.Escape(Path.GetFileName(dataPath))}[/]\n" +
                $"[grey]GNN Type:[/] [cyan]{gnnTypeStr}[/]\n" +
                $"[grey]CNN Channels:[/] [yellow]{string.Join(" -> ", config.CnnChannels)}[/]\n" +
                $"[grey]GNN Hidden Dim:[/] [yellow]{config.GnnHiddenDim}[/]\n" +
                $"[grey]GNN Layers:[/] [yellow]{config.GnnLayers}[/]\n" +
                $"[grey]Attention Heads:[/] [yellow]{config.GnnHeads}[/]\n" +
                $"[grey]Graph K:[/] [yellow]{config.GraphK}[/]\n" +
                $"[grey]Epochs:[/] [yellow]{config.Epochs}[/]\n" +
                $"[grey]Learning Rate:[/] [yellow]{config.LearningRate}[/]\n" +
                $"[grey]Dropout:[/] [yellow]{config.Dropout}[/]\n" +
                $"[grey]Early Stop Patience:[/] [yellow]{config.EarlyStopPatience}[/]"))
            .Header("[purple]Configuration[/]")
            .Border(BoxBorder.Rounded)
            .BorderColor(Color.Purple);

        AnsiConsole.Write(panel);
    }

    private static void DisplayResults(HybridTrainingResult result)
    {
        AnsiConsole.WriteLine();

        var resultPanel = new Panel(
            new Markup(
                $"[grey]Best Validation R²:[/] [green bold]{result.BestValR2:F4}[/]\n" +
                $"[grey]Best Epoch:[/] [yellow]{result.BestEpoch + 1}[/]\n" +
                $"[grey]Final Train Loss:[/] [yellow]{result.FinalTrainLoss:F4}[/]\n" +
                $"[grey]Final Val R²:[/] [cyan]{result.FinalValR2:F4}[/]\n" +
                $"[grey]Total Epochs:[/] [grey]{result.TotalEpochs}[/]\n" +
                $"[grey]Training Time:[/] [cyan]{result.TrainingTime:hh\\:mm\\:ss}[/]"))
            .Header("[green]Training Results[/]")
            .Border(BoxBorder.Double)
            .BorderColor(Color.Green);

        AnsiConsole.Write(resultPanel);

        // Interpretation
        AnsiConsole.WriteLine();
        DisplayQualityInterpretation(result.BestValR2);
    }

    private static void DisplayEnsembleResults(double bestR2, int ensembleCount)
    {
        AnsiConsole.WriteLine();

        var resultPanel = new Panel(
            new Markup(
                $"[grey]Best Individual R²:[/] [green bold]{bestR2:F4}[/]\n" +
                $"[grey]Ensemble Size:[/] [cyan]{ensembleCount} models[/]\n" +
                $"[grey]Note:[/] [yellow]Ensemble predictions average all models[/]"))
            .Header("[green]Ensemble Training Results[/]")
            .Border(BoxBorder.Double)
            .BorderColor(Color.Green);

        AnsiConsole.Write(resultPanel);

        AnsiConsole.WriteLine();
        DisplayQualityInterpretation(bestR2);
    }

    private static void DisplayQualityInterpretation(double r2)
    {
        var interpretation = r2 switch
        {
            >= 0.9 => "[green]Excellent[/] - The model explains most of the variance",
            >= 0.7 => "[yellow]Good[/] - The model captures the main patterns",
            >= 0.5 => "[yellow]Moderate[/] - The model provides some predictive power",
            _ => "[red]Poor[/] - Consider more data or feature engineering"
        };

        AnsiConsole.MarkupLine($"[grey]Model Quality:[/] {interpretation}");
    }
}
