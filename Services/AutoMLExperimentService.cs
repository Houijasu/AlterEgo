namespace AlterEgo.Services
{
    using AlterEgo.Models;

    using Microsoft.ML;
    using Microsoft.ML.AutoML;
    using Microsoft.ML.Data;

    using Spectre.Console;

    /// <summary>
    /// Service for running AutoML experiments to find the best regression algorithm.
    /// </summary>
    public class AutoMLExperimentService
    {
        private readonly MLContext _mlContext;

        public AutoMLExperimentService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// Runs an AutoML experiment to find the best regression algorithm.
        /// </summary>
        public ExperimentResult<RegressionMetrics>? RunExperiment(
            string dataPath,
            uint timeoutSeconds = 120,
            string? modelSavePath = null,
            bool printResults = true)
        {
            if (!File.Exists(dataPath))
            {
                AnsiConsole.MarkupLine($"[red]Error: Data file not found at {Markup.Escape(Path.GetFullPath(dataPath))}[/]");
                return null;
            }

            if (printResults)
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[blue]Running AutoML Experiment...[/]");
                AnsiConsole.MarkupLine("[grey]This will test multiple algorithms to find the best one.[/]");
                AnsiConsole.WriteLine();
            }

            var dataView = _mlContext.Data.LoadFromTextFile<HouseData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var experimentSettings = CreateExperimentSettings(timeoutSeconds);

            if (printResults)
            {
                PrintExperimentHeader(experimentSettings);
            }

            var experiment = _mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            var progressHandler = printResults ? CreateProgressHandler() : null;
            var dataSplit = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var result = experiment.Execute(
                trainData: dataSplit.TrainSet,
                validationData: dataSplit.TestSet,
                labelColumnName: nameof(HouseData.Price),
                progressHandler: progressHandler);

            if (printResults)
            {
                PrintResults(result);
            }

            if (!string.IsNullOrEmpty(modelSavePath) && result.BestRun?.Model != null)
            {
                SaveBestModel(result.BestRun.Model, dataView.Schema, modelSavePath);
            }

            return result;
        }

        private void SaveBestModel(ITransformer model, DataViewSchema schema, string modelPath)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Saving best model to {Markup.Escape(modelPath)}...[/]");
            _mlContext.Model.Save(model, schema, modelPath);
            AnsiConsole.MarkupLine("[green]Best model saved and set as default![/]");
        }

        private static RegressionExperimentSettings CreateExperimentSettings(uint timeoutSeconds)
        {
            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = timeoutSeconds,
                OptimizingMetric = RegressionMetric.RSquared
            };

            // Include all available regression trainers
            settings.Trainers.Clear();
            settings.Trainers.Add(RegressionTrainer.FastForest);
            settings.Trainers.Add(RegressionTrainer.FastTree);
            settings.Trainers.Add(RegressionTrainer.FastTreeTweedie);
            settings.Trainers.Add(RegressionTrainer.LightGbm);
            settings.Trainers.Add(RegressionTrainer.LbfgsPoissonRegression);
            settings.Trainers.Add(RegressionTrainer.StochasticDualCoordinateAscent);

            return settings;
        }

        private static void PrintExperimentHeader(RegressionExperimentSettings settings)
        {
            AnsiConsole.MarkupLine($"[grey]Experiment timeout:[/] [yellow]{settings.MaxExperimentTimeInSeconds}[/] seconds");
            AnsiConsole.MarkupLine($"[grey]Optimizing for:[/] [yellow]{settings.OptimizingMetric}[/]");
            AnsiConsole.Write(new Rule().RuleStyle("grey"));
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[blue]{"Trainer",-35}[/] [green]{"R²",8}[/]  [yellow]{"RMSE",12}[/]");
            AnsiConsole.Write(new Rule().RuleStyle("grey"));
        }

        private static Progress<RunDetail<RegressionMetrics>> CreateProgressHandler()
        {
            return new Progress<RunDetail<RegressionMetrics>>(progress =>
            {
                if (progress.ValidationMetrics != null)
                {
                    AnsiConsole.MarkupLine(
                        $"  [grey]{progress.TrainerName,-35}[/] " +
                        $"R²: [green]{progress.ValidationMetrics.RSquared,8:F4}[/]  " +
                        $"RMSE: [yellow]{progress.ValidationMetrics.RootMeanSquaredError,12:N0}[/]");
                }
            });
        }

        private static void PrintResults(ExperimentResult<RegressionMetrics> result)
        {
            AnsiConsole.Write(new Rule().RuleStyle("grey"));
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[blue]AutoML Results[/]").RuleStyle("blue"));
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Metric[/]")
                .AddColumn("[green]Value[/]");

            table.AddRow("Best Algorithm", $"[yellow]{result.BestRun.TrainerName}[/]");
            table.AddRow("Best R²", $"[green]{result.BestRun.ValidationMetrics.RSquared:F4}[/]");
            table.AddRow("Best RMSE", $"{result.BestRun.ValidationMetrics.RootMeanSquaredError:N0}");
            table.AddRow("Best MAE", $"{result.BestRun.ValidationMetrics.MeanAbsoluteError:N0}");

            AnsiConsole.Write(table);
            PrintAllRuns(result);
        }

        private static void PrintAllRuns(ExperimentResult<RegressionMetrics> result)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[grey]All Runs (sorted by R²)[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            var sortedRuns = result.RunDetails
                .Where(r => r.ValidationMetrics != null)
                .OrderByDescending(r => r.ValidationMetrics!.RSquared)
                .ToList();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Rank[/]")
                .AddColumn("[blue]Trainer[/]")
                .AddColumn("[green]R²[/]")
                .AddColumn("[yellow]RMSE[/]");

            for (var i = 0; i < sortedRuns.Count; i++)
            {
                var run = sortedRuns[i];
                var isBest = run.TrainerName == result.BestRun.TrainerName;
                var rank = isBest ? $"[green]*{i + 1}[/]" : $"{i + 1}";
                var name = isBest ? $"[green]{run.TrainerName}[/]" : run.TrainerName;

                table.AddRow(
                    rank,
                    name,
                    $"{run.ValidationMetrics!.RSquared:F4}",
                    $"{run.ValidationMetrics.RootMeanSquaredError:N0}");
            }

            AnsiConsole.Write(table);
            AnsiConsole.MarkupLine("[grey]* = Best model[/]");
            AnsiConsole.MarkupLine($"[grey]Total algorithms tested:[/] [yellow]{sortedRuns.Count}[/]");
        }
    }
}
