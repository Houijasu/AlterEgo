namespace AlterEgo.Models
{
    using Microsoft.ML.Data;

    /// <summary>
    /// Input data model for house price prediction.
    /// </summary>
    public class HouseData
    {
        [LoadColumn(0)]
        public float Size { get; set; }

        [LoadColumn(1)]
        public float Price { get; set; }
    }
}
