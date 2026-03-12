// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;
using AmazonReviewSentiment;
using static Microsoft.ML.DataOperationsCatalog;

string _trainPath = Path.Combine(Environment.CurrentDirectory, "Data", "Train.tsv");
string _testPath = Path.Combine(Environment.CurrentDirectory, "Data", "Test.tsv");




MLContext mlContext = new MLContext();

TrainTestData splitDataView = LoadData(mlContext);


//Loads data and splits it into test and train datasets
TrainTestData LoadData(MLContext mlContext)
{

    IDataView trainData = mlContext.Data.LoadFromTextFile<SentimentData>(_trainPath, hasHeader: true,  separatorChar: '\t');


    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(trainData, testFraction: 0.2);

    ITransformer model = BuildAndTrainModel(mlContext, trainData);
    Evaluate(mlContext, model, splitDataView.TestSet);
    UseModelWithSingleItem(mlContext, model);
    UseModelWithBatchItems(mlContext, model);
    return splitDataView;
}




//Extracts and transforms the data. Trains the model. Predicts sentiment based on test data. Returns the model.
ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;

}

void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");

}


void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
   
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    SentimentData sampleStatement = new SentimentData
    {
        SentimentText = "This was a very bad steak"
    };
    var resultPrediction = predictionFunction.Predict(sampleStatement);
    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();

}



void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{

    IDataView testData = mlContext.Data.LoadFromTextFile<SentimentData>(_testPath, hasHeader: true, separatorChar: '\t');

    IDataView predictions = model.Transform(testData);

    // Use model to predict whether comment data is Positive (1) or Negative (0).
    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");
}

