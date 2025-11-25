namespace AlterEgo.Services
{
    using AlterEgo.Models;

    using Microsoft.ML;

    using Spectre.Console;

    /// <summary>
    /// Service for training, loading, and saving house price prediction models.
    /// </summary>
    public class HousePriceModelService
    {
        private readonly MLContext _mlContext;

        public HousePriceModelService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// Loads an existing model or trains a new one if necessary.
        /// </summary>
        public ITransformer? LoadOrTrainModel(string dataPath, string modelPath, bool forceRetrain)
        {
            var modelExists = File.Exists(modelPath);
            var dataExists = File.Exists(dataPath);

            var retrainNeeded = forceRetrain || !modelExists;

            if (!retrainNeeded && dataExists)
            {
                var modelTimestamp = File.GetLastWriteTimeUtc(modelPath);
                var dataTimestamp = File.GetLastWriteTimeUtc(dataPath);
                if (dataTimestamp > modelTimestamp)
                {
                    AnsiConsole.MarkupLine("[yellow]Detected newer dataset. Retraining model...[/]");
                    retrainNeeded = true;
                }
            }

            if (retrainNeeded)
            {
                return TrainAndSaveModel(dataPath, modelPath);
            }

            if (!modelExists)
            {
                AnsiConsole.MarkupLine("[red]No trained model available. Provide a dataset or run with --retrain.[/]");
                return null;
            }

            var loadedModel = LoadModel(modelPath);
            if (loadedModel == null)
            {
                // LoadModel already printed a warning about retraining
                return TrainAndSaveModel(dataPath, modelPath);
            }

            return loadedModel;
        }

        /// <summary>
        /// Loads a model from disk.
        /// </summary>
        public ITransformer? LoadModel(string modelPath)
        {
            AnsiConsole.MarkupLine("[grey]Loading saved model...[/]");
            try
            {
                var model = _mlContext.Model.Load(modelPath, out _);
                AnsiConsole.MarkupLine("[green]Model loaded![/]");
                return model;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning: Failed to load existing model ({Markup.Escape(ex.Message)}). Retraining...[/]");
                return null;
            }
        }

        /// <summary>
        /// Trains a new model and saves it to disk.
        /// </summary>
        public ITransformer? TrainAndSaveModel(string dataPath, string modelPath)
        {
            if (!File.Exists(dataPath))
            {
                AnsiConsole.MarkupLine($"[red]Error: Data file not found at {Markup.Escape(Path.GetFullPath(dataPath))}[/]");
                return null;
            }

            var dataView = _mlContext.Data.LoadFromTextFile<HouseData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var dataSplit = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms.Concatenate("Features", nameof(HouseData.Size))
                .Append(_mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(HouseData.Price),
                    maximumNumberOfIterations: 1000));

            AnsiConsole.MarkupLine("[grey]Training model...[/]");
            var model = pipeline.Fit(dataSplit.TrainSet);
            AnsiConsole.MarkupLine("[green]Model trained![/]");

            EvaluateModel(model, dataSplit.TestSet);
            SaveModel(model, dataView.Schema, modelPath);

            return model;
        }

        /// <summary>
        /// Makes a price prediction for a given house size.
        /// </summary>
        public float Predict(ITransformer model, float size)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);
            using (predictionEngine)
            {
                var prediction = predictionEngine.Predict(new HouseData { Size = size });
                return prediction.Price;
            }
        }

        private void EvaluateModel(ITransformer model, IDataView testData)
        {
            var testPredictions = model.Transform(testData);
            var metrics = _mlContext.Regression.Evaluate(testPredictions, labelColumnName: nameof(HouseData.Price));

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[blue]Model metrics:[/]");
            AnsiConsole.MarkupLine($"  R²: [green]{metrics.RSquared:F3}[/]");
            AnsiConsole.MarkupLine($"  RMSE: [yellow]{metrics.RootMeanSquaredError:C0}[/]");
        }

        private void SaveModel(ITransformer model, DataViewSchema schema, string modelPath)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Saving model to {Markup.Escape(modelPath)}...[/]");
            _mlContext.Model.Save(model, schema, modelPath);
            AnsiConsole.MarkupLine("[green]Model saved![/]");
        }
    }
}
