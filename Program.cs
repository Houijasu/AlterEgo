namespace AlterEgo
{
    using AlterEgo.CLI;
    using AlterEgo.CLI.Commands;

    using Spectre.Console.Cli;

    public class Program
    {
        public static int Main(string[] args)
        {
            // Show interactive menu if no arguments provided
            if (args.Length == 0 && !Console.IsInputRedirected)
            {
                return InteractiveMenu.Show();
            }

            var app = new CommandApp<PredictCommand>();

            app.Configure(config =>
            {
                config.SetApplicationName("AlterEgo");
                config.SetApplicationVersion("1.0.0");

                config.AddCommand<AutoMLCommand>("automl")
                    .WithDescription("Run AutoML experiment to find the best algorithm")
                    .WithExample(["automl"])
                    .WithExample(["automl", "-t", "60", "-s"]);

                config.AddCommand<BenchmarkCommand>("benchmark")
                    .WithDescription("Benchmark ALL regression algorithms (including those AutoML skips)")
                    .WithExample(["benchmark"])
                    .WithExample(["benchmark", "-i", "10", "-t", "60"]);

                config.AddCommand<HybridCommand>("hybrid")
                    .WithDescription("Train hybrid CNN+GCN neural network for house price prediction")
                    .WithExample(["hybrid"])
                    .WithExample(["hybrid", "-e", "200", "-v"])
                    .WithExample(["hybrid", "--epochs", "500", "--save", "model.pt"]);

                config.AddCommand<PredictCommand>("predict")
                    .WithDescription("Predict house price using ML.NET or Hybrid model")
                    .WithExample(["predict", "2000"])
                    .WithExample(["predict", "2000", "-m", "hybrid"])
                    .WithExample(["predict", "2000", "-m", "hybrid", "--model-path", "checkpoints/model.pt"]);
            });

            return app.Run(args);
        }
    }
}
