import numpy as np

def grow_defects(current_depths, current_lengths, corrosion_rates, current_std_devs, max_depths, std_dev_corrosion, time_step=1.0):
    """
    Applies corrosion growth to defects over time using vectorization.

    Args:
        current_depths (np.array): Depths at t-1 (mm).
        current_lengths (np.array): Lengths at t-1 (mm).
        corrosion_rates (np.array): Corrosion rates (mm/year).
        current_std_devs (np.array): Standard deviations at t-1 (mm).
        max_depths (np.array): Wall thickness (mm), limit for depth.
        std_dev_corrosion (np.array): Standard deviation of corrosion rate (mm/year).
        time_step (float): Time increment in years.

    Returns:
        tuple: (next_depths, next_lengths, next_std_devs) as np.arrays.
    """
    # Calculate next state
    growth_amount = corrosion_rates * time_step
    
    # Linear growth of standard deviation: Std_new = Std_old + (Std_rate * Time)
    growth_std_dev = std_dev_corrosion * time_step
    
    # Apply growth
    raw_next_depths = current_depths + growth_amount
    next_lengths = current_lengths #por ahora no se incluye el crecimiento en longitud
    next_std_devs = current_std_devs + growth_std_dev
    
    # Enforce physical limits (depth <= wall thickness)
    next_depths = np.minimum(raw_next_depths, max_depths)
    
    return next_depths, next_lengths, next_std_devs

