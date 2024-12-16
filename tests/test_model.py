import unittest
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up resources for testing, such as loading data and preprocessing.
        This method runs once before all tests.
        """
        # Load dataset
        cls.df = pd.read_csv("data/tips.csv")
        
        # Preprocess dataset
        lb = LabelEncoder()
        cls.df['sex'] = lb.fit_transform(cls.df['sex'])
        cls.df['smoker'] = lb.fit_transform(cls.df['smoker'])
        cls.df['day'] = lb.fit_transform(cls.df['day'])
        cls.df['time'] = lb.fit_transform(cls.df['time'])
        
        cls.x = cls.df.drop(columns=['total_bill'], axis=1)  # Input Data
        cls.y = cls.df['total_bill']  # Target Data
        
        # Split dataset
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(
            cls.x, cls.y, test_size=0.2, random_state=42
        )
        
        # Initialize Linear Regression model
        cls.model = LinearRegression()

    def test_model_training(self):
        """
        Test that the model is trained without errors and produces predictions.
        """
        # Train the model
        self.model.fit(self.x_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.x_test)
        
        # Ensure predictions have the same length as test labels
        self.assertEqual(len(y_pred), len(self.y_test), "Prediction length mismatch.")
        
        # Calculate evaluation metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Check that metrics are within a reasonable range
        self.assertGreater(r2, 0.5, "R2 score is too low.")
        self.assertLess(mse, 20, "Mean Squared Error is too high.")

    def test_model_saving(self):
        """
        Test that the trained model is saved correctly.
        """
        # Save the model
        model_path = 'models/tips_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(model_path), "Model file not saved.")
        
        # Load the model and verify it's a LinearRegression instance
        loaded_model = joblib.load(model_path)
        self.assertIsInstance(loaded_model, LinearRegression, "Loaded model is not a LinearRegression instance.")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources after all tests.
        """
        # Remove the saved model file if it exists
        model_path = 'models/tips_model.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    unittest.main()
