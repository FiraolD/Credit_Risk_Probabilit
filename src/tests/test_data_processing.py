# tests/test_data_processing.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Data_Processor import   (  
    AggregateCustomerFeatures,
    TimeBasedFeatureExtractor,
    CategoricalEncoder,
    MissingValueImputer,
    NumericalScaler
)


class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2', 'T3'],
            'CustomerId': ['C1', 'C1', 'C2'],
            'Amount': [100, 200, -50],
            'Value': [100, 200, 50],
            'TransactionStartTime': [
                '2023-01-01T10:00:00Z',
                '2023-01-02T11:00:00Z',
                '2023-01-03T12:00:00Z'
            ]
        })

    def test_aggregate_features_creator(self):
        """Test that AggregateFeaturesCreator creates correct aggregated features"""
    
        
        aggregator = AggregateFeaturesCreator()
        aggregated = aggregator.transform(cleaned)
        
        # Check that new features were added
        expected_features = ['TotalTransactionAmount', 'AvgTransactionAmount', 
                            'TransactionCount', 'StdTransactionAmount']
        for feature in expected_features:
            self.assertIn(feature, aggregated.columns)
        
        # Check values for Customer C1
        c1_data = aggregated[aggregated['CustomerId'] == 'C1']
        self.assertEqual(c1_data.iloc[0]['TotalTransactionAmount'], 300)
        self.assertEqual(c1_data.iloc[0]['AvgTransactionAmount'], 150)
        self.assertEqual(c1_data.iloc[0]['TransactionCount'], 2)
        self.assertAlmostEqual(c1_data.iloc[0]['StdTransactionAmount'], 70.71, delta=0.01)
    
    def test_missing_value_handling(self):
        """Test that missing values are handled correctly"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'Amount'] = np.nan
        
        mv_handler = MissingValueImputer(strategy='median')
        mv_handler.fit(data_with_missing)
        filled = mv_handler.transform(data_with_missing)
        
        # Check that median value was used to fill missing
        self.assertEqual(filled.iloc[0]['Amount'], 125)  # Median of [100, 200, -50] is 100
        
    def test_time_based_features(self):
        """Test that time-based features are correctly extracted"""

        
        time_extractor = TimeBasedFeatureExtractor()
        time_features = time_extractor.transform(text)
        
        # Check that all time-based features were added
        expected_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth',
                            'TransactionYear', 'TransactionDayOfWeek', 'IsWeekend',
                            'DaysSinceFirstTransaction']
        for feature in expected_features:
            self.assertIn(feature, time_features.columns)
        
        # Check specific values
        self.assertEqual(time_features.iloc[0]['TransactionDay'], 1)
        self.assertEqual(time_features.iloc[0]['TransactionMonth'], 1)
        self.assertEqual(time_features.iloc[0]['TransactionYear'], 2023)
        self.assertEqual(time_features.iloc[0]['TransactionDayOfWeek'], 6)  # Sunday
        self.assertEqual(time_features.iloc[0]['IsWeekend'], 1)
        
        # Check days since first transaction
        self.assertEqual(time_features.iloc[0]['DaysSinceFirstTransaction'], 0)
        self.assertEqual(time_features.iloc[1]['DaysSinceFirstTransaction'], 1)


if __name__ == '__main__':
    unittest.main()