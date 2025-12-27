from src.ml.data_prepare import DataPreparer
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_trainer import ModelTrainer


def test_end_to_end_ml_pipeline(spark, ml_config, sample_data):
    preparer = DataPreparer(ml_config, spark)
    engineer = FeatureEngineer(ml_config, spark)
    trainer = ModelTrainer(ml_config, spark)

    train_df, val_df, test_df = preparer._split_data(sample_data)

    train_df, feature_names, stages = engineer.prepare_features(
        train_df, is_training=True
    )
    test_df, _, _ = engineer.prepare_features(test_df, is_training=False)

    assert engineer.validate_features(train_df)

    model = trainer.train(stages, train_df, val_df)
    assert model is not None

    metrics = trainer.evaluate(test_df, stage="test")

    assert metrics["accuracy"] > 0.0

    predictions = trainer.predict(test_df)
    assert "prediction" in predictions.columns
