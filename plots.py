# plots.py
# viz stuff

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# colors from somewhere idr
COLORS = ['#08f7fe', '#fe53bb', '#f5d300', '#ffacfc', '#09fbd3', '#7122fa', '#ff0000', '#00ff00']

def orbit_2d(hist, names=None):
    """2d orbit plot"""
    if isinstance(hist, list) and isinstance(hist[0], dict):
        hist = np.array([h['pos'] for h in hist])
    
    fig = go.Figure()
    n = hist.shape[1]
    
    for i in range(n):
        name = names[i] if names else f'body {i}'
        c = COLORS[i % len(COLORS)]
        
        fig.add_trace(go.Scatter(
            x=hist[:, i, 0], y=hist[:, i, 1],
            mode='lines', name=name,
            line=dict(color=c, width=2),
            opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            x=[hist[-1, i, 0]], y=[hist[-1, i, 1]],
            mode='markers', name=f'{name} (now)',
            marker=dict(color=c, size=12, line=dict(width=2, color='white')),
            showlegend=False
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Orbits',
        xaxis_title='x', yaxis_title='y',
        xaxis=dict(scaleanchor='y'),
        height=600
    )
    return fig

def orbit_3d(hist, names=None):
    # same thing but 3d lol
    if isinstance(hist, list) and isinstance(hist[0], dict):
        hist = np.array([h['pos'] for h in hist])
    
    fig = go.Figure()
    n = hist.shape[1]
    
    for i in range(n):
        name = names[i] if names else f'body {i}'
        fig.add_trace(go.Scatter3d(
            x=hist[:, i, 0], y=hist[:, i, 1], z=hist[:, i, 2],
            mode='lines', name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=4),
            opacity=0.8
        ))
    
    fig.update_layout(template='plotly_dark', title='3D Orbits', height=700)
    return fig

def energy_plot(hist, masses, method=''):
    from analysis import total_energy
    
    if isinstance(hist, list) and isinstance(hist[0], dict):
        energies = [total_energy(h['pos'], h['vel'], masses) for h in hist]
    else:
        energies = hist
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(energies))), y=energies, 
        mode='lines', name=f'E ({method})',
        line=dict(color='#00ff88', width=2)
    ))
    fig.update_layout(
        template='plotly_dark', 
        title='Energy',
        xaxis_title='Step', yaxis_title='E'
    )
    return fig

def compare_methods(times, energy_dict):
    """compare energy error, idk if the colors are right"""
    fig = go.Figure()
    
    colors = {'euler': '#ff6b6b', 'verlet': '#4ecdc4', 'leapfrog': '#45b7d1', 'rk4': '#ffeaa7'}
    
    for method, energies in energy_dict.items():
        e0 = energies[0] if energies[0] != 0 else 1
        err = np.abs((np.array(energies) - e0) / e0)
        fig.add_trace(go.Scatter(
            x=times, y=err, mode='lines', name=method,
            line=dict(color=colors.get(method, '#fff'), width=2)
        ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Energy Error',
        xaxis_title='t', yaxis_title='|ΔE/E₀|',
        yaxis_type='log'
    )
    return fig

def phase_space(pos, vel, body_idx=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pos[:, body_idx, 0], y=vel[:, body_idx, 0],
        mode='lines', line=dict(color='#a29bfe', width=1)
    ))
    fig.update_layout(
        template='plotly_dark',
        title=f'Phase (body {body_idx})',
        xaxis_title='x', yaxis_title='vₓ'
    )
    return fig

def lagrange_contour(X, Y, Z, pts, mu):
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=X[0], y=Y[:,0], z=Z,
        colorscale='Viridis',
        contours=dict(start=-5, end=0, size=0.2),
        showscale=False
    ))
    
    for name, (lx, ly) in pts.items():
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode='markers+text', name=name,
            marker=dict(size=12, color='red'),
            text=[name], textposition='top center'
        ))
    
    fig.add_trace(go.Scatter(x=[-mu], y=[0], mode='markers', name='M1',
                              marker=dict(size=20, color='yellow')))
    fig.add_trace(go.Scatter(x=[1-mu], y=[0], mode='markers', name='M2',
                              marker=dict(size=10, color='cyan')))
    
    fig.update_layout(
        template='plotly_dark',
        title=f'U_eff (μ={mu:.4f})',
        xaxis_title='x', yaxis_title='y',
        xaxis=dict(scaleanchor='y'),
        height=600
    )
    return fig

def animated_orbits(hist, names=None, skip=10):
    # animation, this was annoying to figure out lol
    if isinstance(hist, list) and isinstance(hist[0], dict):
        hist = np.array([h['pos'] for h in hist])
    
    n_bodies = hist.shape[1]
    
    fig = go.Figure()
    for i in range(n_bodies):
        name = names[i] if names else f'body {i}'
        fig.add_trace(go.Scatter(
            x=[hist[0, i, 0]], y=[hist[0, i, 1]],
            mode='markers', name=name,
            marker=dict(color=COLORS[i % len(COLORS)], size=12)
        ))
    
    frames = []
    for t in range(0, len(hist), skip):
        fdata = []
        for i in range(n_bodies):
            fdata.append(go.Scatter(x=[hist[t, i, 0]], y=[hist[t, i, 1]]))
        frames.append(go.Frame(data=fdata, name=str(t)))
    
    fig.frames = frames
    
    fig.update_layout(
        template='plotly_dark',
        title='Animation',
        xaxis=dict(
            range=[hist[:,:,0].min()*1.1, hist[:,:,0].max()*1.1],
            scaleanchor='y'
        ),
        yaxis=dict(range=[hist[:,:,1].min()*1.1, hist[:,:,1].max()*1.1]),
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[dict(
                label='Play', method='animate',
                args=[None, dict(frame=dict(duration=50), fromcurrent=True)]
            )]
        )],
        height=600
    )
    return fig
