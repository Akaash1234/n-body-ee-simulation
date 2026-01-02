# nasa_stuff.py
# JPL horizons wrapper
# need astroquery: pip install astroquery

from astroquery.jplhorizons import Horizons
import numpy as np
import pandas as pd

# body IDs, had to look these up lol
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
    'pluto': '999',  # idrc what they say its a planet
}

def get_body(name, epoch='2024-01-01'):
    """get pos/vel from jpl"""
    name = name.lower()
    body_id = BODIES.get(name, name)
    
    # epoch format is weird idk
    epochs = {
        'start': epoch, 
        'stop': (pd.Timestamp(epoch) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 
        'step': '1d'
    }
    print(f"fetching {name}...")
    
    obj = Horizons(id=body_id, location='@sun', epochs=epochs)
    
    try:
        vec = obj.vectors()
        pos = np.array([vec['x'][0], vec['y'][0], vec['z'][0]])  # AU
        vel = np.array([vec['vx'][0], vec['vy'][0], vec['vz'][0]])  # AU/day
        
        # convert
        AU_KM = 1.496e8
        pos_km = pos * AU_KM
        vel_kms = vel * AU_KM / 86400
        
        return {'pos': pos_km, 'vel': vel_kms, 'name': name}
    except Exception as e:
        print(f"failed: {e}")
        return None

def get_solar_system(epoch='2024-01-01'):
    bodies = ['sun', 'mercury', 'venus', 'earth', 'mars']
    data = []
    for b in bodies:
        d = get_body(b, epoch)
        if d: data.append(d)
    return data

# masses, horizons is annoying about this so hardcoded
MASSES = {
    'sun': 1.989e30,
    'mercury': 3.301e23,
    'venus': 4.867e24,
    'earth': 5.972e24,
    'mars': 6.417e23,
    'jupiter': 1.898e27,
    'saturn': 5.683e26,
    'uranus': 8.681e25,
    'neptune': 1.024e26,
    'moon': 7.342e22,
    'pluto': 1.309e22,
}

def get_mass(name):
    return MASSES.get(name.lower(), 1e20)

def normalize_for_sim(positions, velocities, masses):
    """
    SI -> sim units (G=1)
    this took forever to get right lw
    """
    M_sun = 1.989e30
    AU_km = 1.496e8
    
    m_scale = M_sun
    r_scale = AU_km
    # v scale for G=1... idr the derivation
    v_scale = np.sqrt(6.674e-11 * M_sun / (AU_km * 1e3)) / 1e3
    
    pos_n = positions / r_scale
    vel_n = velocities / v_scale
    mass_n = masses / m_scale
    
    return pos_n, vel_n, mass_n, {'r': r_scale, 'v': v_scale, 'm': m_scale}
