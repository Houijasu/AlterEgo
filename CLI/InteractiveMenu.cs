namespace AlterEgo.CLI
{
    using AlterEgo.CLI.Commands;
    using AlterEgo.Models.Neural;
    using AlterEgo.Services;

    using Microsoft.ML;

    using Spectre.Console;

    /// <summary>
    /// Interactive menu for the application.
    /// </summary>
    public static class InteractiveMenu
    {
        private const string DataFileName = "housing.csv";
        private const string ModelFileName = "house_price_model.zip";

        public static int Show()
        {
            while (true)
            {
                Console.Clear();
                PrintHeader();

                var choice = AnsiConsole.Prompt(
                    new SelectionPrompt<string>()
                        .Title("[blue]What would you like to do?[/]")
                        .PageSize(10)
                        .HighlightStyle(Style.Parse("green bold"))
                        .AddChoices([
                            "🏠 Predict House Price",
                            "🤖 Run AutoML Experiment",
                            "📊 Benchmark All Algorithms",
                            "🔗 Create Ensemble Model",
                            "🧠 Train Hybrid CNN+GCN Model",
                            "❌ Exit"
                        ]));

                if (choice.Contains("Exit"))
                {
                    AnsiConsole.MarkupLine("[grey]Goodbye![/]");
                    return 0;
                }

                var result = choice switch
                {
                    var c when c.Contains("Predict") => RunPredict(),
                    var c when c.Contains("AutoML") => RunAutoML(),
                    var c when c.Contains("Benchmark") => RunBenchmark(),
                    var c when c.Contains("Ensemble") => RunEnsemble(),
                    var c when c.Contains("Hybrid") => RunHybrid(),
                    _ => 0
                };

                if (result != 0)
                {
                    AnsiConsole.MarkupLine($"\n[yellow]Operation completed with code {result}[/]");
                }

                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[grey]Press any key to return to menu...[/]");
                Console.ReadKey(true);
            }
        }

        private static void PrintHeader()
        {
            AnsiConsole.Write(new FigletText("AlterEgo")
                .Color(Color.Blue)
                .Centered());

            AnsiConsole.Write(new Rule("[grey]ML.NET House Price Prediction[/]")
                .RuleStyle("blue"));
            AnsiConsole.WriteLine();
        }

        private static int RunPredict()
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[blue]House Price Prediction[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var modelPath = Path.Combine(AppContext.BaseDirectory, ModelFileName);
                var dataPath = Path.Combine(AppContext.BaseDirectory, DataFileName);

                var modelService = new HousePriceModelService(mlContext);

                // Ask if user wants to retrain
                var forceRetrain = AnsiConsole.Confirm("Force model retraining?", false);

                var model = modelService.LoadOrTrainModel(dataPath, modelPath, forceRetrain);
                if (model == null)
                {
                    return 1;
                }

                // Get house size from user
                var size = AnsiConsole.Prompt(
                    new TextPrompt<float>("Enter house size in [green]square feet[/]:")
                        .PromptStyle("yellow")
                        .ValidationErrorMessage("[red]Please enter a valid positive number[/]")
                        .Validate(s => s > 0
                            ? ValidationResult.Success()
                            : ValidationResult.Error("[red]Size must be positive[/]")));

                var predictedPrice = modelService.Predict(model, size);
                PrintPrediction(size, predictedPrice);

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        private static void PrintPrediction(float size, float price)
        {
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Property[/]")
                .AddColumn("[green]Value[/]");

            table.AddRow("House Size", $"{size:N0} sq ft");
            table.AddRow("Predicted Price", $"[green bold]{price:C0}[/]");

            AnsiConsole.Write(new Panel(table)
                .Header("[yellow]Price Estimate[/]")
                .BorderColor(Color.Green));
        }

        private static int RunAutoML()
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[blue]AutoML Experiment[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var dataPath = Path.Combine(AppContext.BaseDirectory, DataFileName);
                var modelPath = Path.Combine(AppContext.BaseDirectory, ModelFileName);

                // Get timeout from user
                var timeout = AnsiConsole.Prompt(
                    new TextPrompt<uint>("Experiment timeout in [green]seconds[/]:")
                        .PromptStyle("yellow")
                        .DefaultValue(60u)
                        .ValidationErrorMessage("[red]Please enter a valid number[/]"));

                // Ask if user wants to save best model
                var saveModel = AnsiConsole.Confirm("Save best model as default?", true);

                var autoMLService = new AutoMLExperimentService(mlContext);
                autoMLService.RunExperiment(
                    dataPath,
                    timeout,
                    saveModel ? modelPath : null,
                    printResults: true);

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        private static int RunBenchmark()
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[blue]Algorithm Benchmark[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var dataPath = Path.Combine(AppContext.BaseDirectory, DataFileName);

                // Get settings from user
                var folds = AnsiConsole.Prompt(
                    new TextPrompt<int>("Cross-validation [green]folds[/]:")
                        .PromptStyle("yellow")
                        .DefaultValue(5)
                        .ValidationErrorMessage("[red]Please enter a valid number[/]")
                        .Validate(f => f >= 2 && f <= 20
                            ? ValidationResult.Success()
                            : ValidationResult.Error("[red]Folds must be between 2 and 20[/]")));

                var timeout = AnsiConsole.Prompt(
                    new TextPrompt<uint>("AutoML timeout in [green]seconds[/]:")
                        .PromptStyle("yellow")
                        .DefaultValue(60u)
                        .ValidationErrorMessage("[red]Please enter a valid number[/]"));

                AnsiConsole.WriteLine();

                var benchmarkService = new BenchmarkService(mlContext);
                var autoMLService = new AutoMLExperimentService(mlContext);

                var results = RunBenchmarkWithProgress(benchmarkService, autoMLService, dataPath, folds, timeout);
                DisplayBenchmarkResults(results);

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        private static int RunEnsemble()
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[blue]Ensemble Model Creator[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var mlContext = new MLContext(seed: 1);
                var dataPath = Path.Combine(AppContext.BaseDirectory, DataFileName);

                // Choose ensemble type
                var ensembleType = AnsiConsole.Prompt(
                    new SelectionPrompt<string>()
                        .Title("Select ensemble type:")
                        .AddChoices([
                            "🔵 Linear Ensemble (OLS + SDCA + SGD)",
                            "🌲 Tree Ensemble (FastTree + FastForest + LightGBM)",
                            "📚 Stacking Ensemble (ML.NET Native - SDCA + FastTree + FastForest + LightGBM)"
                        ]));

                var folds = AnsiConsole.Prompt(
                    new TextPrompt<int>("Cross-validation [green]folds[/] for evaluation:")
                        .PromptStyle("yellow")
                        .DefaultValue(5));

                AnsiConsole.WriteLine();

                var ensembleService = new EnsembleService(mlContext);

                // Handle Stacking Ensemble (diverse algorithms combined)
                if (ensembleType.Contains("Stacking"))
                {
                    EnsembleService.LinearEnsemble? stackingEnsemble = null;
                    EnsembleMetrics? stackingMetrics = null;

                    AnsiConsole.Status()
                        .Spinner(Spinner.Known.Dots)
                        .SpinnerStyle(Style.Parse("magenta"))
                        .Start("[magenta]Building Stacking Ensemble...[/]", ctx =>
                        {
                            (stackingEnsemble, stackingMetrics) = ensembleService.CreateStackingEnsemble(dataPath, folds);
                        });

                    AnsiConsole.WriteLine();
                    AnsiConsole.Write(new Rule("[green]Stacking Ensemble[/]").RuleStyle("green"));
                    AnsiConsole.WriteLine();

                    // Display component models with weights
                    var componentTable = new Table()
                        .Border(TableBorder.Rounded)
                        .AddColumn("[blue]Model[/]")
                        .AddColumn(new TableColumn("[yellow]Weight[/]").RightAligned());

                    var models = stackingEnsemble!.GetModels();
                    var totalWeight = models.Sum(m => m.Weight);
                    foreach (var (name, weight) in models)
                    {
                        var pct = totalWeight > 0 ? (weight / totalWeight * 100) : 0;
                        componentTable.AddRow(name, $"{pct:F1}%");
                    }

                    AnsiConsole.Write(componentTable);

                    // Display metrics
                    AnsiConsole.WriteLine();
                    var resultPanel = new Panel(
                        new Markup($"[grey]Avg R² Score:[/] [green bold]{stackingMetrics!.RSquared:F4}[/]\n" +
                                  $"[grey]Avg RMSE:[/] [yellow]{stackingMetrics.RootMeanSquaredError:N0}[/]\n" +
                                  $"[grey]Avg MAE:[/] [cyan]{stackingMetrics.MeanAbsoluteError:N0}[/]"))
                        .Header("[yellow]📚 Stacking Ensemble Performance[/]")
                        .Border(BoxBorder.Double)
                        .BorderColor(Color.Yellow);

                    AnsiConsole.Write(resultPanel);

                    // Test prediction
                    AnsiConsole.WriteLine();
                    if (AnsiConsole.Confirm("Test a prediction with the stacking ensemble?", true))
                    {
                        var size = AnsiConsole.Prompt(
                            new TextPrompt<float>("Enter house size in [green]square feet[/]:")
                                .PromptStyle("yellow")
                                .DefaultValue(2000f));

                        var prediction = stackingEnsemble.Predict(new Models.HouseData { Size = size });
                        AnsiConsole.MarkupLine($"\n[grey]Predicted price for {size:N0} sq ft:[/] [green bold]{prediction:C0}[/]");
                    }
                }
                else
                {
                    // Handle Linear/Tree ensembles (custom weighted)
                    EnsembleService.LinearEnsemble ensemble;
                    List<EnsembleService.ModelScore> scores;

                    AnsiConsole.Status()
                        .Spinner(Spinner.Known.Dots)
                        .SpinnerStyle(Style.Parse("blue"))
                        .Start("[yellow]Building ensemble...[/]", ctx =>
                        {
                            if (ensembleType.Contains("Linear"))
                            {
                                (ensemble, scores) = ensembleService.CreateLinearEnsemble(dataPath, folds);
                            }
                            else
                            {
                                (ensemble, scores) = ensembleService.CreateTreeEnsemble(dataPath, folds);
                            }
                        });

                    // Rebuild ensemble outside status block for display
                    if (ensembleType.Contains("Linear"))
                    {
                        (ensemble, scores) = ensembleService.CreateLinearEnsemble(dataPath, folds);
                    }
                    else
                    {
                        (ensemble, scores) = ensembleService.CreateTreeEnsemble(dataPath, folds);
                    }

                    // Display component models
                    AnsiConsole.WriteLine();
                    AnsiConsole.Write(new Rule("[green]Ensemble Components[/]").RuleStyle("green"));
                    AnsiConsole.WriteLine();

                    var componentTable = new Table()
                        .Border(TableBorder.Rounded)
                        .AddColumn("[blue]Model[/]")
                        .AddColumn(new TableColumn("[green]R²[/]").RightAligned())
                        .AddColumn(new TableColumn("[yellow]Weight[/]").RightAligned());

                    var totalWeight = scores.Where(s => s.Weight > 0).Sum(s => s.Weight);
                    foreach (var score in scores)
                    {
                        var pct = totalWeight > 0 ? (score.Weight / totalWeight * 100) : 0;
                        var r2Text = double.IsNaN(score.RSquared) ? "[red]FAILED[/]" : $"{score.RSquared:F4}";
                        var weightText = score.Weight > 0 ? $"{pct:F1}%" : "[grey]excluded[/]";
                        componentTable.AddRow(score.Name, r2Text, weightText);
                    }

                    AnsiConsole.Write(componentTable);

                    // Evaluate ensemble
                    AnsiConsole.WriteLine();
                    var (ensembleR2, ensembleRMSE, ensembleMAE) = ensembleService.EvaluateEnsemble(ensemble, dataPath);

                    var resultPanel = new Panel(
                        new Markup($"[grey]R² Score:[/] [green bold]{ensembleR2:F4}[/]\n" +
                                  $"[grey]RMSE:[/] [yellow]{ensembleRMSE:N0}[/]\n" +
                                  $"[grey]MAE:[/] [cyan]{ensembleMAE:N0}[/]"))
                        .Header("[yellow]🔗 Ensemble Performance[/]")
                        .Border(BoxBorder.Double)
                        .BorderColor(Color.Yellow);

                    AnsiConsole.Write(resultPanel);

                    // Test prediction
                    AnsiConsole.WriteLine();
                    if (AnsiConsole.Confirm("Test a prediction with the ensemble?", true))
                    {
                        var size = AnsiConsole.Prompt(
                            new TextPrompt<float>("Enter house size in [green]square feet[/]:")
                                .PromptStyle("yellow")
                                .DefaultValue(2000f));

                        var prediction = ensemble.Predict(new Models.HouseData { Size = size });
                        AnsiConsole.MarkupLine($"\n[grey]Predicted price for {size:N0} sq ft:[/] [green bold]{prediction:C0}[/]");
                    }
                }

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
                            ctx.Status($"[yellow]Testing:[/] [white]{name}[/] [grey]({current}/{total})[/]");
                        }));
                });

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
                        printResults: false);

                    if (autoMLResult?.RunDetails != null)
                    {
                        var autoMLBenchmarks = autoMLResult.RunDetails
                            .Where(r => r.ValidationMetrics != null)
                            .Select(r => new BenchmarkService.BenchmarkResult(
                                Name: SimplifyTrainerName(r.TrainerName),
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

            return results.OrderByDescending(r => r.RSquared).ToList();
        }

        private static string SimplifyTrainerName(string fullName)
        {
            // Extract just the algorithm name from pipeline string
            // e.g., "ReplaceMissingValues=>Concatenate=>FastTreeRegression" -> "FastTree"
            var parts = fullName.Split("=>");
            var lastPart = parts[^1];

            // Remove common suffixes
            return lastPart
                .Replace("Regression", "")
                .Replace("Regressor", "")
                .Trim();
        }

        private static string CategorizeTrainer(string trainerName)
        {
            if (trainerName.Contains("FastTree") || trainerName.Contains("FastForest") || trainerName.Contains("LightGbm"))
                return "Tree";
            if (trainerName.Contains("Sdca") || trainerName.Contains("Lbfgs") || trainerName.Contains("Ols"))
                return "Linear";
            return "Other";
        }

        private static void DisplayBenchmarkResults(List<BenchmarkService.BenchmarkResult> results)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[green]Results[/]").RuleStyle("green"));
            AnsiConsole.WriteLine();

            // Group by algorithm name and take best of each
            var bestByAlgorithm = results
                .Where(r => !double.IsNaN(r.RSquared))
                .GroupBy(r => r.Name)
                .Select(g => g.OrderByDescending(r => r.RSquared).First())
                .OrderByDescending(r => r.RSquared)
                .ToList();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .Title("[blue]Best Run Per Algorithm[/]")
                .Caption("[grey]Sorted by R² (higher is better)[/]");

            table.AddColumn(new TableColumn("[blue]#[/]").Centered());
            table.AddColumn("[blue]Algorithm[/]");
            table.AddColumn(new TableColumn("[blue]Category[/]").Centered());
            table.AddColumn(new TableColumn("[blue]AutoML[/]").Centered());
            table.AddColumn(new TableColumn("[green]R²[/]").RightAligned());
            table.AddColumn(new TableColumn("[yellow]RMSE[/]").RightAligned());

            var rank = 0;
            foreach (var result in bestByAlgorithm.Take(15))
            {
                rank++;
                var isBest = rank == 1;

                var rankText = isBest ? $"[green]★ {rank}[/]" : $"{rank}";
                var nameText = isBest ? $"[green]{result.Name}[/]" : result.Name;

                var categoryColor = result.Category switch
                {
                    "Linear" => "cyan",
                    "Tree" => "green",
                    "Tree-Tuned" => "green",
                    "Interpretable" => "magenta",
                    "Polynomial" => "yellow",
                    "Regularized" => "blue",
                    _ => "white"
                };

                table.AddRow(
                    rankText,
                    nameText,
                    $"[{categoryColor}]{result.Category}[/]",
                    result.InAutoML ? "[green]✓[/]" : "[yellow]✗[/]",
                    isBest ? $"[green bold]{result.RSquared:F4}[/]" : $"{result.RSquared:F4}",
                    $"{result.RMSE:N0}");
            }

            AnsiConsole.Write(table);

            // Winner panel
            var best = bestByAlgorithm.FirstOrDefault();
            if (best != null)
            {
                AnsiConsole.WriteLine();

                var winnerGrid = new Grid().AddColumn().AddColumn();
                winnerGrid.AddRow("[grey]Algorithm:[/]", $"[green bold]{best.Name}[/]");
                winnerGrid.AddRow("[grey]R² Score:[/]", $"[green]{best.RSquared:F4}[/]");
                winnerGrid.AddRow("[grey]In AutoML:[/]",
                    best.InAutoML ? "[green]Yes[/]" : "[yellow]No (Custom algorithm!)[/]");

                AnsiConsole.Write(new Panel(winnerGrid)
                    .Header("[yellow]🏆 Winner[/]")
                    .Border(BoxBorder.Double)
                    .BorderColor(Color.Yellow));
            }
        }

        private static int RunHybrid()
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[purple]Hybrid CNN+GCN Neural Network[/]").RuleStyle("grey"));
            AnsiConsole.WriteLine();

            try
            {
                var dataPath = Path.Combine(AppContext.BaseDirectory, DataFileName);

                if (!File.Exists(dataPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error: Data file not found at {Markup.Escape(dataPath)}[/]");
                    return 1;
                }

                // Get settings from user
                var epochs = AnsiConsole.Prompt(
                    new TextPrompt<int>("Number of training [green]epochs[/]:")
                        .PromptStyle("yellow")
                        .DefaultValue(200)
                        .ValidationErrorMessage("[red]Please enter a valid number[/]")
                        .Validate(e => e >= 10 && e <= 2000
                            ? ValidationResult.Success()
                            : ValidationResult.Error("[red]Epochs must be between 10 and 2000[/]")));

                var k = AnsiConsole.Prompt(
                    new TextPrompt<int>("k-NN graph [green]neighbors[/]:")
                        .PromptStyle("yellow")
                        .DefaultValue(15)
                        .ValidationErrorMessage("[red]Please enter a valid number[/]")
                        .Validate(k => k >= 3 && k <= 50
                            ? ValidationResult.Success()
                            : ValidationResult.Error("[red]K must be between 3 and 50[/]")));

                var verbose = AnsiConsole.Confirm("Show detailed training progress?", true);

                AnsiConsole.WriteLine();

                // Create configuration
                var config = new HybridConfig
                {
                    Epochs = epochs,
                    GraphK = k
                };

                // Display configuration
                var configPanel = new Panel(
                    new Markup(
                        $"[grey]CNN Channels:[/] [yellow]{string.Join(" -> ", config.CnnChannels)}[/]\n" +
                        $"[grey]GNN Hidden Dim:[/] [yellow]{config.GnnHiddenDim}[/]\n" +
                        $"[grey]Attention Heads:[/] [yellow]{config.GnnHeads}[/]\n" +
                        $"[grey]Graph K:[/] [yellow]{config.GraphK}[/]\n" +
                        $"[grey]Epochs:[/] [yellow]{config.Epochs}[/]\n" +
                        $"[grey]Learning Rate:[/] [yellow]{config.LearningRate}[/]"))
                    .Header("[purple]Configuration[/]")
                    .Border(BoxBorder.Rounded)
                    .BorderColor(Color.Purple);

                AnsiConsole.Write(configPanel);
                AnsiConsole.WriteLine();

                // Initialize and train
                using var service = new HybridModelService(config);
                service.Initialize();
                AnsiConsole.WriteLine();

                var result = service.Train(dataPath, verbose);

                // Display results
                AnsiConsole.WriteLine();
                var resultPanel = new Panel(
                    new Markup(
                        $"[grey]Best Validation R²:[/] [green bold]{result.BestValR2:F4}[/]\n" +
                        $"[grey]Best Epoch:[/] [yellow]{result.BestEpoch + 1}[/]\n" +
                        $"[grey]Final Train Loss:[/] [yellow]{result.FinalTrainLoss:F4}[/]\n" +
                        $"[grey]Training Time:[/] [cyan]{result.TrainingTime:hh\\:mm\\:ss}[/]"))
                    .Header("[green]Training Results[/]")
                    .Border(BoxBorder.Double)
                    .BorderColor(Color.Green);

                AnsiConsole.Write(resultPanel);

                // Interpretation
                AnsiConsole.WriteLine();
                var r2 = result.BestValR2;
                var interpretation = r2 switch
                {
                    >= 0.9 => "[green]Excellent[/] - The model explains most of the variance",
                    >= 0.7 => "[yellow]Good[/] - The model captures the main patterns",
                    >= 0.5 => "[yellow]Moderate[/] - The model provides some predictive power",
                    _ => "[red]Poor[/] - Consider more data or feature engineering"
                };

                AnsiConsole.MarkupLine($"[grey]Model Quality:[/] {interpretation}");

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
    }
}
