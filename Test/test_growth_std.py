

import numpy as np
import sys
import os

# Add parent directory to path to import growth_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from growth_model import grow_defects

def test_std_dev_growth_linear():
    """
    Verify that standard deviation grows linearly: std_new = std_old + (std_rate * time)
    """
    print("Running test_std_dev_growth_linear...")
    # Setup
    current_depths = np.array([2.0, 3.0])
    current_lengths = np.array([10.0, 15.0])
    corrosion_rates = np.array([0.1, 0.2])
    current_std_devs = np.array([0.05, 0.05])
    max_depths = np.array([10.0, 10.0])
    std_dev_corrosion = np.array([0.01, 0.02])
    time_step = 2.0
    
    # Expected values
    # Std Dev 1: 0.05 + (0.01 * 2.0) = 0.07
    # Std Dev 2: 0.05 + (0.02 * 2.0) = 0.09
    expected_std_devs = np.array([0.07, 0.09])
    
    # Execute
    next_depths, next_lengths, next_std_devs = grow_defects(
        current_depths,
        current_lengths,
        corrosion_rates,
        current_std_devs,
        max_depths,
        std_dev_corrosion,
        time_step
    )
    
    # Verify
    np.testing.assert_almost_equal(next_std_devs, expected_std_devs, decimal=5)
    print("test_std_dev_growth_linear PASSED")
    
def test_std_dev_no_growth():
    """
    Verify that standard deviation does not change if rate is 0.
    """
    print("Running test_std_dev_no_growth...")
    # Setup
    current_depths = np.array([2.0])
    current_lengths = np.array([10.0])
    corrosion_rates = np.array([0.1])
    current_std_devs = np.array([0.05])
    max_depths = np.array([10.0])
    std_dev_corrosion = np.array([0.0])
    time_step = 5.0
    
    # Expected values
    expected_std_devs = np.array([0.05])
    
    # Execute
    _, _, next_std_devs = grow_defects(
        current_depths,
        current_lengths,
        corrosion_rates,
        current_std_devs,
        max_depths,
        std_dev_corrosion,
        time_step
    )
    
    # Verify
    np.testing.assert_almost_equal(next_std_devs, expected_std_devs, decimal=5)
    print("test_std_dev_no_growth PASSED")

if __name__ == "__main__":
    test_std_dev_growth_linear()
    test_std_dev_no_growth()
