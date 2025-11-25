namespace AlterEgo.Models
{
    using Microsoft.ML.Data;

    /// <summary>
    /// Output prediction model for house price.
    /// </summary>
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
