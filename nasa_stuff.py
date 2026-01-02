import datetime
from astroquery.jplhorizons import Horizons
import numpy as np

# JPL Horizons ID mapping
BODIES = {
    'sun': '10',
    'mercury': '199',
    'venus': '299',
    'earth': '399',
    'mars': '499',
    'jupiter': '599',
    'saturn': '699',
    'uranus': '799',
    'neptune': '899',
    'moon': '301',
    'pluto': '999',
}

def get_body_state(name, epoch='2024-01-01'):
    """
    Fetch position and velocity vectors from JPL Horizons.
    Returns state vectors in km and km/s.
    """
    name = name.lower()
    body_id = BODIES.get(name, name)
    
    # Calculate end time for query (1 day span required for vector calculation)
    start_date = datetime.date.fromisoformat(epoch)
    end_date = start_date + datetime.timedelta(days=1)
    
    epochs = {
        'start': start_date.isoformat(), 
        'stop': end_date.isoformat(), 
        'step': '1d'
    }
    
    print(f"Querying JPL Horizons for {name}...")
    
    obj = Horizons(id=body_id, location='@sun', epochs=epochs)
    
    try:
        vec = obj.vectors()
        # Horizon returns AU and AU/d
        pos_au = np.array([vec['x'][0], vec['y'][0], vec['z'][0]]) 
        vel_au_d = np.array([vec['vx'][0], vec['vy'][0], vec['vz'][0]])
        
        # Constants
        AU_KM = 1.49597871e8
        DAY_SEC = 86400.0
        
        r_vec = pos_au * AU_KM
        v_vec = vel_au_d * AU_KM / DAY_SEC
        
        return {'pos': r_vec, 'vel': v_vec, 'name': name}
    except Exception as e:
        print(f"Error fetching data for {name}: {e}")
        return None

def get_solar_system_state(epoch='2024-01-01'):
    targets = ['sun', 'mercury', 'venus', 'earth', 'mars']
    system_state = []
    for body in targets:
        state = get_body_state(body, epoch)
        if state:
            system_state.append(state)
    return system_state

# Planetary masses (kg)
# Source: NASA Planetary Fact Sheet
MASSES = {
    'sun': 1.9885e30,
    'mercury': 3.3011e23,
    'venus': 4.8675e24,
    'earth': 5.9724e24,
    'mars': 6.4171e23,
    'jupiter': 1.8982e27,
    'saturn': 5.6834e26,
    'uranus': 8.6810e25,
    'neptune': 1.0241e26,
    'moon': 7.342e22,
    'pluto': 1.3090e22,
}

def get_mass(name):
    return MASSES.get(name.lower(), 1.0) # Default to 1.0 if unknown (avoid div/0)

def to_sim_units(positions, velocities, masses, g_ref=1.0):
    """
    Normalize units for numerical stability.
    Scales system such that G = 1, M_sun = 1, and R_scale = 1 AU.
    """
    M_sun_kg = 1.9885e30
    AU_km = 1.49597871e8
    G_si = 6.67430e-11
    
    # Scaling factors
    m_scale = M_sun_kg
    r_scale = AU_km
    
    # Derivation for v_scale:
    # F = G * M * m / r^2
    # v^2 / r ~ G * M / r^2  => v ~ sqrt(G * M / r)
    # To make G_sim = 1:
    # v_scale = sqrt(G_si * m_scale / (r_scale_m))
    # Note: input r is in km, G uses meters.
    
    r_scale_m = r_scale * 1e3
    v_scale = np.sqrt(G_si * m_scale / r_scale_m) / 1e3 # result in km/s
    
    pos_n = positions / r_scale
    vel_n = velocities / v_scale
    mass_n = masses / m_scale
    
    return pos_n, vel_n, mass_n, {'r_scale': r_scale, 'v_scale': v_scale, 'm_scale': m_scale}
