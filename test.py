import unittest
import sys
import os

# Add the current directory to the path so we can import the assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assignment import find_pipeline_for_cancer_dataset, create_feature_store, evaluate_pipeline_performance
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

class TestAutoML(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.cancer = load_breast_cancer()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.cancer.data, self.cancer.target, train_size=0.75, test_size=0.25, random_state=42
        )
    
    def test_find_pipeline_for_cancer_dataset_returns_pipeline(self):
        """Test that the function returns a valid pipeline"""
        try:
            pipeline = find_pipeline_for_cancer_dataset()
            self.assertIsInstance(pipeline, Pipeline)
            self.assertGreater(len(pipeline.steps), 0)
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")
    
    def test_pipeline_can_predict(self):
        """Test that the returned pipeline can make predictions"""
        try:
            pipeline = find_pipeline_for_cancer_dataset()
            predictions = pipeline.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))
            self.assertTrue(all(isinstance(pred, (int, float)) for pred in predictions))
        except Exception as e:
            self.fail(f"Pipeline prediction failed: {e}")
    
    def test_pipeline_has_reasonable_accuracy(self):
        """Test that the pipeline achieves reasonable accuracy"""
        try:
            pipeline = find_pipeline_for_cancer_dataset()
            score = pipeline.score(self.X_test, self.y_test)
            self.assertGreaterEqual(score, 0.7)  # Should achieve at least 70% accuracy
            print(f"Pipeline achieved {score:.3f} accuracy")
        except Exception as e:
            self.fail(f"Pipeline scoring failed: {e}")
    
    def test_create_feature_store(self):
        """Test the feature store creation function"""
        try:
            feature_store = create_feature_store(self.cancer.data, self.cancer.feature_names, self.cancer.target)
            self.assertIsInstance(feature_store, dict)
            self.assertIn('data', feature_store)
            self.assertIn('feature_names', feature_store)
            self.assertIn('feature_stats', feature_store)
            self.assertEqual(feature_store['total_samples'], len(self.cancer.data))
            self.assertEqual(feature_store['feature_count'], len(self.cancer.feature_names))
            self.assertIsInstance(feature_store['data'], pd.DataFrame)
            stats = feature_store['feature_stats']
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)
        except Exception as e:
            self.fail(f"Feature store creation failed: {e}")
    
    def test_evaluate_pipeline_performance(self):
        """Test the pipeline performance evaluation function"""
        try:
            pipeline = find_pipeline_for_cancer_dataset()
            metrics = evaluate_pipeline_performance(pipeline, self.X_test, self.y_test)
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)
            self.assertGreaterEqual(metrics['accuracy'], 0.0)
            self.assertLessEqual(metrics['accuracy'], 1.0)
            self.assertGreaterEqual(metrics['precision'], 0.0)
            self.assertLessEqual(metrics['precision'], 1.0)
            self.assertGreaterEqual(metrics['recall'], 0.0)
            self.assertLessEqual(metrics['recall'], 1.0)
            self.assertGreaterEqual(metrics['f1_score'], 0.0)
            self.assertLessEqual(metrics['f1_score'], 1.0)
            print(f"Performance metrics: {metrics}")
        except Exception as e:
            self.fail(f"Performance evaluation failed: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
