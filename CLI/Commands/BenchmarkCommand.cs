namespace AlterEgo.CLI.Commands
{
    using AlterEgo.Services;

    using Microsoft.ML;

    using Spectre.Console;
    using Spectre.Console.Cli;

    /// <summary>
    /// Command for benchmarking all regression algorithms.
    /// </summary>
    public sealed class BenchmarkCommand : Command<BenchmarkSettings>
    {
        private const string DataFileName = "housing.csv";

        public override int Execute(CommandContext context, BenchmarkSettings settings, CancellationToken cancellationToken)
        {
            AnsiConsole.Write(new FigletText("Benchmark")
                .Color(Color.Blue)
                .Centered());

            AnsiConsole.Write(new Rule("[grey]ML.NET Regression Algorithm Comparison[/]")
                .RuleStyle("blue"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var dataPath = settings.DataPath ?? Path.Combine(AppContext.BaseDirectory, DataFileName);

                if (!File.Exists(dataPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error: Data file not found at {Markup.Escape(dataPath)}[/]");
                    return 1;
                }

                var benchmarkService = new BenchmarkService(mlContext);
                var autoMLService = new AutoMLExperimentService(mlContext);

                // Show configuration panel
                var configPanel = new Panel(
                    new Markup($"[grey]Cross-validation folds:[/] [yellow]{settings.Iterations}[/]\n" +
                              $"[grey]Data file:[/] [yellow]{Path.GetFileName(dataPath)}[/]\n" +
                              $"[grey]AutoML Timeout:[/] [yellow]{settings.Timeout}[/]s"))
                    .Header("[blue]Configuration[/]")
                    .Border(BoxBorder.Rounded)
                    .BorderColor(Color.Blue);

                AnsiConsole.Write(configPanel);
                AnsiConsole.WriteLine();

                // Run benchmark with progress
                var results = RunBenchmarkWithProgress(benchmarkService, autoMLService, dataPath, settings.Iterations, settings.Timeout);

                // Display results
                DisplayResults(results, showAutoMLInfo: true);

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        private static List<BenchmarkService.BenchmarkResult> RunBenchmarkWithProgress(
            BenchmarkService service,
            AutoMLExperimentService autoMLService,
            string dataPath,
            int folds,
            uint autoMLTimeout)
        {
            var results = new List<BenchmarkService.BenchmarkResult>();

            // 1. Run Custom Algorithms
            AnsiConsole.Status()
                .AutoRefresh(true)
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start("[yellow]Benchmarking Custom Algorithms...[/]", ctx =>
                {
                    results.AddRange(service.RunBenchmark(
                        dataPath,
                        folds,
                        (name, current, total) =>
                        {
                            ctx.Status($"[yellow]Testing Custom:[/] [white]{name}[/] [grey]({current}/{total})[/]");
                        }));
                });

            // 2. Run AutoML Algorithms
            AnsiConsole.Status()
                .AutoRefresh(true)
                .Spinner(Spinner.Known.Aesthetic)
                .SpinnerStyle(Style.Parse("green"))
                .Start("[green]Benchmarking AutoML Algorithms...[/]", ctx =>
                {
                    var autoMLResult = autoMLService.RunExperiment(
                        dataPath,
                        autoMLTimeout,
                        modelSavePath: null,
                        printResults: false); // Silent run

                    if (autoMLResult?.RunDetails != null)
                    {
                        var autoMLBenchmarks = autoMLResult.RunDetails
                            .Where(r => r.ValidationMetrics != null)
                            .Select(r => new BenchmarkService.BenchmarkResult(
                                Name: r.TrainerName,
                                Category: CategorizeTrainer(r.TrainerName),
                                InAutoML: true,
                                RSquared: r.ValidationMetrics!.RSquared,
                                RMSE: r.ValidationMetrics.RootMeanSquaredError,
                                MAE: r.ValidationMetrics.MeanAbsoluteError,
                                TrainingTime: TimeSpan.FromSeconds(r.RuntimeInSeconds)
                            ));

                        results.AddRange(autoMLBenchmarks);
                    }
                });

            AnsiConsole.WriteLine();
            return results.OrderByDescending(r => r.RSquared).ToList();
        }

        private static string CategorizeTrainer(string trainerName)
        {
            if (trainerName.Contains("FastTree") || trainerName.Contains("FastForest") || trainerName.Contains("LightGbm"))
                return "Tree";
            if (trainerName.Contains("Sdca") || trainerName.Contains("Lbfgs") || trainerName.Contains("Ols"))
                return "Linear";
            return "Other";
        }

        private static void DisplayResults(List<BenchmarkService.BenchmarkResult> results, bool showAutoMLInfo)
        {
            // === MAIN RESULTS TABLE ===
            AnsiConsole.Write(new Rule("[green]Results[/]").RuleStyle("green"));
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey)
                .Title("[blue]Algorithm Performance Rankings[/]")
                .Caption("[grey]Sorted by R² (higher is better)[/]");

            table.AddColumn(new TableColumn("[blue]#[/]").Centered());
            table.AddColumn(new TableColumn("[blue]Algorithm[/]"));
            table.AddColumn(new TableColumn("[blue]Category[/]").Centered());

            if (showAutoMLInfo)
            {
                table.AddColumn(new TableColumn("[blue]AutoML[/]").Centered());
            }

            table.AddColumn(new TableColumn("[green]R²[/]").RightAligned());
            table.AddColumn(new TableColumn("[yellow]RMSE[/]").RightAligned());
            table.AddColumn(new TableColumn("[cyan]MAE[/]").RightAligned());
            table.AddColumn(new TableColumn("[grey]Time[/]").RightAligned());

            var rank = 0;
            var bestResult = results.FirstOrDefault();

            foreach (var result in results)
            {
                rank++;
                var isBest = result == bestResult;
                var isFailed = double.IsNaN(result.RSquared);

                var rankText = isBest ? $"[green]★ {rank}[/]" : $"{rank}";
                var nameText = isBest ? $"[green]{result.Name}[/]" :
                               isFailed ? $"[red strikethrough]{result.Name}[/]" : result.Name;

                var categoryColor = result.Category switch
                {
                    "Linear" => "cyan",
                    "Tree" => "green",
                    "Interpretable" => "magenta",
                    "Ensemble" => "purple",
                    "Neural" => "red",
                    _ => "white"
                };

                var categoryText = $"[{categoryColor}]{result.Category}[/]";
                var autoMLText = result.InAutoML ? "[green]✓[/]" : "[yellow]✗[/]";

                var r2Text = isFailed ? "[red]FAILED[/]" :
                             isBest ? $"[green bold]{result.RSquared:F4}[/]" : $"{result.RSquared:F4}";
                var rmseText = isFailed ? "[grey]-[/]" : $"{result.RMSE:N0}";
                var maeText = isFailed ? "[grey]-[/]" : $"{result.MAE:N0}";
                var timeText = isFailed ? "[grey]-[/]" : $"{result.TrainingTime.TotalMilliseconds:N0}ms";

                if (showAutoMLInfo)
                {
                    table.AddRow(rankText, nameText, categoryText, autoMLText, r2Text, rmseText, maeText, timeText);
                }
                else
                {
                    table.AddRow(rankText, nameText, categoryText, r2Text, rmseText, maeText, timeText);
                }
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // === WINNER PANEL ===
            if (bestResult != null && !double.IsNaN(bestResult.RSquared))
            {
                var winnerContent = new Grid()
                    .AddColumn()
                    .AddColumn();

                winnerContent.AddRow(
                    new Markup("[grey]Algorithm:[/]"),
                    new Markup($"[green bold]{bestResult.Name}[/]"));
                winnerContent.AddRow(
                    new Markup("[grey]Category:[/]"),
                    new Markup($"[white]{bestResult.Category}[/]"));
                winnerContent.AddRow(
                    new Markup("[grey]R² Score:[/]"),
                    new Markup($"[green]{bestResult.RSquared:F4}[/]"));
                winnerContent.AddRow(
                    new Markup("[grey]In AutoML:[/]"),
                    new Markup(bestResult.InAutoML ? "[green]Yes[/]" : "[yellow]No (AutoML misses this!)[/]"));

                var winnerPanel = new Panel(winnerContent)
                    .Header("[yellow]🏆 Winner[/]")
                    .Border(BoxBorder.Double)
                    .BorderColor(Color.Yellow);

                AnsiConsole.Write(winnerPanel);
                AnsiConsole.WriteLine();
            }

            // === CATEGORY BREAKDOWN ===
            DisplayCategoryBreakdown(results);

            // === LEGEND ===
            DisplayLegend(showAutoMLInfo);
        }

        private static void DisplayCategoryBreakdown(List<BenchmarkService.BenchmarkResult> results)
        {
            AnsiConsole.Write(new Rule("[grey]Category Analysis[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            var categories = results
                .Where(r => !double.IsNaN(r.RSquared))
                .GroupBy(r => r.Category)
                .Select(g => new
                {
                    Category = g.Key,
                    Best = g.OrderByDescending(r => r.RSquared).First(),
                    AvgR2 = g.Average(r => r.RSquared),
                    Count = g.Count()
                })
                .OrderByDescending(c => c.AvgR2);

            var categoryTable = new Table()
                .Border(TableBorder.Simple)
                .BorderColor(Color.Grey)
                .HideHeaders();

            categoryTable.AddColumn("Category");
            categoryTable.AddColumn("Best Algorithm");
            categoryTable.AddColumn("Avg R²");

            foreach (var cat in categories)
            {
                var catColor = cat.Category switch
                {
                    "Linear" => "cyan",
                    "Tree" => "green",
                    "Interpretable" => "magenta",
                    "Ensemble" => "purple",
                    "Neural" => "red",
                    _ => "white"
                };

                categoryTable.AddRow(
                    $"[{catColor}]{cat.Category}[/] [grey]({cat.Count} algorithms)[/]",
                    $"[white]{cat.Best.Name}[/]",
                    $"[grey]avg[/] [white]{cat.AvgR2:F4}[/]");
            }

            AnsiConsole.Write(categoryTable);
            AnsiConsole.WriteLine();
        }

        private static void DisplayLegend(bool showAutoMLInfo)
        {
            var legendItems = new List<string>
            {
                "[green]★[/] = Best performer",
                "[cyan]Linear[/] = Linear regression algorithms",
                "[green]Tree[/] = Decision tree algorithms",
                "[magenta]Interpretable[/] = Explainable models",
                "[purple]Ensemble[/] = Combined models",
                "[red]Neural[/] = Hybrid CNN+GCN Network"
            };

            if (showAutoMLInfo)
            {
                legendItems.Add("[green]✓[/] = Supported by AutoML");
                legendItems.Add("[yellow]✗[/] = Not in AutoML");
            }

            var legend = new Panel(
                new Rows(legendItems.Select(l => new Markup(l))))
                .Header("[grey]Legend[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Grey);

            AnsiConsole.Write(legend);
        }
    }
}
