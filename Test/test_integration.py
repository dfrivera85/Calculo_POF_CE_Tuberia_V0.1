import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import main

class TestPOFSimulation(unittest.TestCase):
    def setUp(self):
        # Create dummy dataframes mimicking the 10 inputs
        self.dfs = {}
        
        # 1. Juntas
        self.dfs['juntas_soldadura'] = pd.DataFrame({
            'junta': ['J1', 'J2', 'J3', 'J4', 'J5'],
            'altura_m': [0]*5,
            'longitud_tubo_m': [10]*5,
            'tipo_costura': ['S']*5,
            'distancia_inicio_m': [0.0, 100.0, 200.0, 300.0, 400.0],
            'distancia_fin_m': [100.0, 200.0, 300.0, 400.0, 500.0],
            'lat': [0.0]*5,
            'lon': [0.0]*5,
            'diametro': [300.0]*5, # mm
            'espesor': [10.0]*5, # mm
            'SMYS': [300.0]*5, # MPa
        })
        
        # 2. Anomalies (ILI)
        # Anomaly at 150m (Joint 100m) with depth 2mm
        self.dfs['anomalias'] = pd.DataFrame({
            'corrida_ILI': ['ILI2023'],
            'distancia_m': [150.0],
            'profundidad_mm': [2.0],
            'ancho': [10.0],
            'largo': [20.0],
            'tipo_defecto': ['GENE'],
            'latitud': [0.0],
            'longitud': [0.0],
            'posicion_horaria_hh_mm': ['12:00'],
            'tipo_anomalia': ['EXT'],
            'posicion_pared': ['EXT'],
            'reduccion_diametro_o_deformacion_porcentaje': [0.0],
            'comentarios': ['None'],
            'junta_anterior_m': [100.0],
            'junta_posterior_m': [200.0],
            'estado_A_I_R': ['A']
        })
        
        # 3. Environment
        env_files = [
            ('resistividad', 'resistividad_suelo_ohm_cm'),
            ('tipo_suelo', 'tipo_suelo'),
            ('potencial', 'cp_potencial_on_mv'),
            ('interferencia', 'interferencia_dc'),
            ('tipo_recubrimiento', 'tipo_recubrimiento'),
            ('edad_recubrimiento', 'edad_recubrimiento_anos'),
            ('presion', 'presion'),
            ('cgr_corrosion_externa', 'tasa_corrosion_mm_ano')
        ]
        
        for key, val_col in env_files:
            if key == 'cgr_corrosion_externa':
                self.dfs[key] = pd.DataFrame({
                    'distancia_inicio_m': [0.0],
                    'distancia_fin_m': [500.0],
                    'tasa_corrosion_mm_ano': [0.1],
                    'desviacion_estandar_corrosion_mm_ano': [0.01]
                })
            elif key == 'tipo_suelo' or key == 'tipo_recubrimiento':
                self.dfs[key] = pd.DataFrame({
                    'distancia_inicio_m': [0.0], 'distancia_fin_m': [500.0], 
                    val_col: ['Clay'] if 'suelo' in key else ['FBE']
                })
            else:
                self.dfs[key] = pd.DataFrame({
                    'distancia_inicio_m': [0.0], 'distancia_fin_m': [500.0], 
                    val_col: [10.0]
                })

        # Add Field Data (Validation)
        self.dfs['inspecciones_directas'] = pd.DataFrame({
            'distancia_m': [150.0],
            'profundidad_campo_mm': [2.2], 
            'ancho_campo_mm': [10.0],
            'largo_campo_mm': [20.0],
            'tipo_defecto_campo': ['GENE']
        })
        
        # Tolerances
        self.tolerances_df = pd.DataFrame({
            'Defect Type': ['GENE', 'Nivel de Confianza'],
            'Tolerance': [0.10, 0.80]
        })
        
    def test_run_simulation(self):
        ili_date = datetime(2023, 1, 1)
        target_date = datetime(2025, 1, 1)
        
        try:
            results = main.run_simulation(
                self.dfs, 
                ili_date, 
                target_date, 
                self.tolerances_df, 
                detection_threshold=0.10
            )
            
            # Checks
            self.assertIn('master_df', results)
            self.assertIn('pof_results', results)
            
            master = results['master_df']
            self.assertFalse(master.empty)
            
            pof_res = results['pof_results']
            self.assertFalse(pof_res.empty)
            
            # Check Years
            unique_years = sorted(pof_res['Year'].unique())
            self.assertEqual(unique_years, [2023, 2024, 2025])
            
            # Find row near 150m (at T=0)
            res_2023 = pof_res[pof_res['Year'] == 2023]
            row_150 = res_2023[res_2023['Distance'] == 150.0]
            if not row_150.empty:
                self.assertAlmostEqual(row_150['Depth'].values[0], 2.2, places=2)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Simulation failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
