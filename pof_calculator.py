import numpy as np
from scipy import stats

def calculate_folias_factor(length, diameter, thickness):
    """
    Calculates the Folias Factor (M) for B31G Modified.
    Includes conditional for short and long defects.
    """
    z = (length**2) / (diameter * thickness)
    term_short = 1 + 0.6275 * z - 0.003375 * (z**2)
    m_short = np.sqrt(np.maximum(0, term_short))
    m_long = 0.032 * z + 3.3
    return np.where(z <= 50, m_short, m_long)

def calculate_hoop_stress(pressure, diameter, thickness):
    """
    Calculates Hoop Stress (Barlow's formula approximation).
    Sigma_h = (P * D) / (2 * t)
    
    Args:
        pressure: Internal Pressure in PSI.
        diameter: Pipe Diameter in mm.
        thickness: Wall Thickness in mm.
        
    Returns:
        Hoop Stress in PSI.
    """
    return (pressure * diameter) / (2 * thickness)

def calculate_critical_depth(hoop_stress, flow_stress, thickness, folias_factor):
    """
    Calculates Critical Depth based on Modified B31G inversion.
    d_crit = (t / 0.85) * (Sigma_h - S_flow) / (Sigma_h/M - S_flow)
    """
    raw_d_crit = ((hoop_stress-flow_stress)*thickness)/(((hoop_stress/folias_factor)-flow_stress)*0.85)
    
    return np.clip(raw_d_crit, 0, thickness)

def calculate_pof_analytics(current_depths, current_lengths, pressure, diameter, thickness, smys, std_devs):
    """
    Calculates POF using analytical approximation (inverted B31G Mod).
    
    Args:
        current_depths (np.array): Mean depths (mm).
        current_lengths (np.array): Defect lengths (mm).
        pressure (float): Operating pressure (psi).
        diameter (float): Pipe outer diameter (mm).
        thickness (float): Wall thickness (mm).
        smys (float): Yield strength (psi).
        std_devs (np.array): Standard deviation of defect depth distribution (mm).
        
    Returns:
        np.array: Probability of Failure (0.0 to 1.0).
    """
    # 1. Properties
    flow_stress = smys + 10000 # B31G Modified standard
    hoop_stress = calculate_hoop_stress(pressure, diameter, thickness)
    
    # 2. Geometry Factors
    folias_factor = calculate_folias_factor(current_lengths, diameter, thickness)
    
    # 3. Limit State (Critical Depth)
    d_crit = calculate_critical_depth(hoop_stress, flow_stress, thickness, folias_factor)
    
    # 4. POF Calculation
    # Probability that Actual Depth > Critical Depth
    # Actual Depth ~ Normal(current_depths, std_devs)
    # POF = 1 - CDF(d_crit) = SF(d_crit)
    
    # Handle cases where std_dev is 0 to avoid nan (though unlikely in this model)
    # If std is 0, if depth > d_crit -> 1, else 0.
    
    # Using scipy.stats.norm.sf (Survival Function)
    # loc = mean, scale = std
    pof = stats.norm.sf(d_crit, loc=current_depths, scale=std_devs)
    
    return pof
