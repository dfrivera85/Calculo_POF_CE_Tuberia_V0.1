import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Attempt to import PyMC and Arviz for Level 3
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

class ILIValidationLevel2:
    def __init__(self, ili_tolerance, ili_certainty, field_uncertainty_std=0.0):
        """
        Inicializa la clase de validación.

        Parámetros:
        - ili_tolerance (float): Tolerancia de la herramienta ILI (ej. 10% = 10.0).
        - ili_certainty (float): Certeza declarada por el proveedor (ej. 80% = 0.80).
        - field_uncertainty_std (float): Desviación estándar del error de medición en campo.
          Si es 0, se asume medición perfecta (conservador, pero a veces no realista).
        """
        self.ili_tolerance = ili_tolerance
        self.ili_certainty = ili_certainty
        self.field_uncertainty_std = field_uncertainty_std
        self.alpha = 0.10  # Para un nivel de confianza del 90% en la validación (API 1163 recomienda >= 80%)

        # Z-score para la certeza de la herramienta (usualmente 80% -> Z=1.28)
        self.z_ili = norm.ppf(0.5 + self.ili_certainty / 2)

    def calculate_combined_tolerance(self):
        """
        1) & 2) Account for Field Measurement Uncertainty & Depth Tolerance.
        Calcula la tolerancia combinada (Delta_e_comb) según Ec. (8) de API 1163.
        """
        # Convertir la incertidumbre de campo (std dev) a tolerancia con la misma certeza que el ILI
        # API 1163 Sec 8.2.4.6.1: tolerancia_campo = std_dev * Z_score_certeza
        field_tolerance = self.field_uncertainty_std * self.z_ili

        # Ec. (8): Raíz cuadrada de la suma de los cuadrados de las tolerancias
        delta_e_comb = np.sqrt(self.ili_tolerance**2 + field_tolerance**2)

        return delta_e_comb

    def validate_measurements(self, df):
        """
        3) Determine Number of Measurements within Stated Tolerance.
        Evalúa cada punto según Ec. (9).
        """
        delta_e_comb = self.calculate_combined_tolerance()

        # Calcular el error: e = ILI - Campo (Ec. 7)
        # Asumiendo que las columnas son 'ILI_Depth' y 'Field_Depth'
        if 'ILI_Depth' not in df.columns or 'Field_Depth' not in df.columns:
             raise ValueError("El DataFrame debe contener las columnas 'ILI_Depth' y 'Field_Depth'")

        df['Error'] = df['ILI_Depth'] - df['Field_Depth']
        df['Abs_Error'] = df['Error'].abs()

        # Determinar si está dentro de tolerancia (Ec. 9: |e| <= delta_e_comb)
        # Nota: El estándar dice "fuera si e > tol", por ende "dentro si e <= tol"
        df['Within_Tolerance'] = df['Abs_Error'] <= delta_e_comb

        # Contar éxitos (X) y total (n)
        x_success = df['Within_Tolerance'].sum()
        n_total = len(df)
        return df, x_success, n_total, delta_e_comb

    def calculate_confidence_bounds(self, x, n):
        """
        4) Determine Upper and Lower Confidence Bounds on Actual Certainty.
        Usa el método de Agresti-Coull según Ecs. (11) a (14) de API 1163.
        """
        if n == 0:
            return 0, 0, 0

        # Nivel de confianza para el intervalo (API sugiere 90% para rechazo, alpha=0.10)
        z_alpha_2 = norm.ppf(1 - self.alpha / 2) # Z para 90% confianza es ~1.645

        z_sq = z_alpha_2**2

        # Ec. (13): n_tilde
        n_tilde = n + z_sq

        # Ec. (14): p_tilde (estimador ajustado)
        p_tilde = (x + (z_sq / 2)) / n_tilde

        # Término de error estándar
        se = np.sqrt((p_tilde * (1 - p_tilde)) / n_tilde)

        # Ecs. (11) y (12): Límites superior e inferior
        p_upper = p_tilde + (z_alpha_2 * se)
        p_lower = p_tilde - (z_alpha_2 * se)

        # Acotar entre 0 y 1
        p_upper = min(p_upper, 1.0)
        p_lower = max(p_lower, 0.0)

        return p_lower, p_upper, p_tilde

    def evaluate_outcome(self, p_lower, p_upper):
        """
        5) Compare Stated Certainty to Confidence Bounds.
        Clasifica el resultado según Sec. 8.2.4.9 (Outcomes 1, 2, 3).
        """
        p_stated = self.ili_certainty
        
        outcome_msg = ""
        # Keep outcomes concise for UI display
        if p_upper < p_stated:
            outcome_msg = "RESULTADO: FALLO.\nLa certeza observada es confiablemente MENOR a la especificación."
        elif p_lower <= p_stated <= p_upper:
            outcome_msg = "RESULTADO: POSIBLE CUMPLIMIENTO.\nLa especificación cae dentro del intervalo de confianza."
        elif p_stated < p_lower:
            outcome_msg = "RESULTADO: ÉXITO SUPERIOR.\nLa herramienta superó la especificación con confianza estadística."
        else:
            outcome_msg = "Resultado Indeterminado."
            
        return outcome_msg

class ILIValidationLevel3:
    def __init__(self):
        pass

    def perform_validation(self, df, vendor_tolerance=10.0, confidence_level=0.80):
        """
        Ejecuta la validación Nivel 3 usando PyMC.
        Retorna (fig, mensaje_estado)
        """
        if not HAS_PYMC:
            return None, "Error: Librerías 'pymc' y 'arviz' no instaladas. Requeridas para Nivel 3."

        if 'ILI_Depth' not in df.columns or 'Field_Depth' not in df.columns:
             return None, "Error: El DataFrame debe contener las columnas 'ILI_Depth' y 'Field_Depth'"

        # Prepara datos
        data = df.dropna()
        field = data['Field_Depth'].values
        ili = data['ILI_Depth'].values
        
        if len(field) < 10:
             return None, "Error: Se requieren al menos 10 puntos de datos para la validación Nivel 3."

        # ---------------------------------------------------------
        # 2. Modelo Bayesiano (API 1163 Nivel 3)
        # ---------------------------------------------------------
        with pm.Model() as model_ili:
            # Priors para los parámetros del modelo lineal (y = alpha + beta * x)
            alpha = pm.Normal('alpha', mu=0, sigma=10)       # Intercepto (Sesgo constante)
            beta = pm.Normal('beta', mu=1, sigma=0.5)        # Pendiente (Sesgo proporcional)

            # Priors para el error (sigma = sigma0 + sigma1 * x)
            # Usamos HalfNormal para asegurar que la desviación estándar sea positiva
            sigma0 = pm.HalfNormal('sigma0', sigma=5)
            sigma1 = pm.HalfNormal('sigma1', sigma=1)

            # Definición de la media y la desviación estándar esperada
            mu_est = alpha + beta * field
            sigma_est = sigma0 + sigma1 * field

            # Likelihood (Verosimilitud) de los datos observados
            y_obs = pm.Normal('y_obs', mu=mu_est, sigma=sigma_est, observed=ili)

            # -----------------------------------------------------
            # 3. Inferencia (Muestreo)
            # -----------------------------------------------------
            # Reducimos muestras para respuesta más rápida en UI, ajusta según necesidad
            trace = pm.sample(1000, tune=500, return_inferencedata=True, target_accept=0.9, progressbar=True)

        # ---------------------------------------------------------
        # 4. Predicción Posterior (Generación de Intervalos)
        # ---------------------------------------------------------
        # Generamos una línea de tendencia para todo el rango de profundidades (0-100%)
        x_pred = np.linspace(0, 100, 100)

        # Calculamos manualmente los intervalos predictivos usando el trace
        post = trace.posterior
        alpha_samples = post['alpha'].values.flatten()
        beta_samples = post['beta'].values.flatten()
        sigma0_samples = post['sigma0'].values.flatten()
        sigma1_samples = post['sigma1'].values.flatten()

        # Generar predicciones para cada punto x_pred
        # Matriz de (n_samples, n_x_points)
        mu_pred = alpha_samples[:, None] + beta_samples[:, None] * x_pred
        sigma_pred = sigma0_samples[:, None] + sigma1_samples[:, None] * x_pred

        # Simular observaciones futuras (Posterior Predictive)
        y_pred_samples = np.random.normal(mu_pred, sigma_pred)

        # Calcular intervalos HDI (High Density Interval) al nivel de confianza deseado
        hdi_prob = confidence_level
        lower_bound = np.quantile(y_pred_samples, (1 - hdi_prob) / 2, axis=0)
        upper_bound = np.quantile(y_pred_samples, 1 - (1 - hdi_prob) / 2, axis=0)
        mean_pred = np.mean(y_pred_samples, axis=0)

        # ---------------------------------------------------------
        # 5. Visualización de Resultados
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Datos Reales
        ax.scatter(field, ili, color='black', alpha=0.6, label='Datos')
        
        # Línea de Unidad (Ideal)
        ax.plot(x_pred, x_pred, 'k--', linewidth=1, label='Ideal 1:1')
        
        # Intervalo de Inferencia Bayesiana
        ax.plot(x_pred, mean_pred, color='blue', label='Tendencia Media')
        ax.fill_between(x_pred, lower_bound, upper_bound, color='blue', alpha=0.2, 
                         label=f'Inferencia ({int(hdi_prob*100)}%)')
        
        # Tolerancia Vendor
        ax.plot(x_pred, x_pred + vendor_tolerance, color='red', linestyle='--', linewidth=1, label=f'Vendor +/-{int(vendor_tolerance)}')
        ax.plot(x_pred, x_pred - vendor_tolerance, color='red', linestyle='--', linewidth=1)
        
        ax.set_title(f'Validación Nivel 3 (Confianza {int(hdi_prob*100)}%)')
        ax.set_xlabel('Prof. Campo')
        ax.set_ylabel('Prof. ILI')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        return fig, "Validación Nivel 3 Completada."
