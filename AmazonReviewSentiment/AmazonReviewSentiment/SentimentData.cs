using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AmazonReviewSentiment
{
    public class SentimentData
    {
        [LoadColumn(2)]
        public string? SentimentText;

        [LoadColumn(0), ColumnName("Label")]
  
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
