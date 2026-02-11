
import unittest
import numpy as np
from scipy import stats
from pof_calculator import (
    calculate_folias_factor,
    calculate_hoop_stress,
    calculate_critical_depth,
    calculate_pof_analytics
)

class TestPOFCalculator(unittest.TestCase):
    
    def test_pof_calculation(self):
        # datos simulados para la prueba
        smys = 60000
        pressure = 2050.107129
        diameter = 355.6
        thickness = 9.53
        current_depths = np.array([0.953])
        current_lengths = np.array([22])
        std_devs = np.array([1.4295])

        flow_stress = smys + 10000 # B31G Modified standard
        hoop_stress = calculate_hoop_stress(pressure, diameter, thickness)
        print(f"Hoop Stress: {hoop_stress}")
    
        # 2. Geometry Factors
        folias_factor = calculate_folias_factor(current_lengths, diameter, thickness)
        print(f"Folias Factor: {folias_factor}")
    
        # 3. Limit State (Critical Depth)
        d_crit = calculate_critical_depth(hoop_stress, flow_stress, thickness, folias_factor)
        print(f"Critical Depth: {d_crit}")
    
        # 4. POF Calculation
        pof = stats.norm.sf(d_crit, loc=current_depths, scale=std_devs)
        print(f"POF: {pof}")

if __name__ == '__main__':
    unittest.main()
