# app.py - main streamlit app
# run with: streamlit run app.py

import streamlit as st
import numpy as np

from solver import Sim
from analysis import total_energy, energy_error, angular_momentum
from lagrange import (lagrange_pts, potential_grid, critical_mu, is_stable, 
                      jacobi_constant, stability_eigenvalues, zero_velocity_curves,
                      simulate_cr3bp, flow_field, poincare_section)
from plots import (orbit_2d, orbit_3d, energy_plot, compare_methods, phase_space, 
                   lagrange_contour, animated_orbits)

st.set_page_config(page_title='N-Body Sim', layout='wide')
st.title('N-Body Simulation & Analysis')

# navigation
st.sidebar.header('Settings')
tab = st.sidebar.radio('Mode', ['Simulation', 'Lagrange Points', 'NASA Data'])


# ==================== SIMULATION TAB ====================
if tab == 'Simulation':
    st.sidebar.subheader('Initial Conditions')
    preset = st.sidebar.selectbox('Preset', ['Two Body', 'Three Body (Figure-8)', 'Random', 'Custom'])
    
    n_bodies = None
    if preset == 'Random':
        n_bodies = st.sidebar.slider('N Bodies', 2, 50, 3)
    
    st.sidebar.subheader('Simulation')
    method = st.sidebar.selectbox('Integrator', ['verlet', 'leapfrog', 'rk4', 'euler'])
    dt = st.sidebar.slider('dt', 0.0001, 0.1, 0.01, format='%.4f')
    steps = st.sidebar.slider('Steps', 100, 10000, 2000)
    g_scale = st.sidebar.slider('G scale', 0.1, 10.0, 1.0)
    
    # setup ICs based on preset
    if preset == 'Custom':
        from editor import create_editor, dynamics_from_df
        
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = [
                {'Name': 'Body 1', 'Mass': 100.0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'Color': '#FF0000'},
                {'Name': 'Body 2', 'Mass': 10.0, 'x': 2.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 4.0, 'vz': 0.0, 'Color': '#00FF00'},
                {'Name': 'Body 3', 'Mass': 1.0, 'x': 3.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 3.0, 'vz': 0.0, 'Color': '#0000FF'},
            ]
        
        edited_df = create_editor(st.session_state['editor_data'])
        pos, vel, mass, names, colors = dynamics_from_df(edited_df)
        
    elif preset == 'Two Body':
        # circular orbit setup
        # v = sqrt(G*M/r) for the orbiting body
        M1, M2 = 10.0, 1.0
        r = 1.0
        v_orb = np.sqrt(g_scale * (M1 + M2) / r)
        # split velocity for COM conservation
        v1 = v_orb * M2 / (M1 + M2)
        v2 = v_orb * M1 / (M1 + M2)
        # positions centered on COM
        x1 = -r * M2 / (M1 + M2)
        x2 = r * M1 / (M1 + M2)
        pos = [[x1, 0, 0], [x2, 0, 0]]
        vel = [[0, -v1, 0], [0, v2, 0]]
        mass = [M1, M2]
        names = ['Star', 'Planet']
        
    elif preset == 'Three Body (Figure-8)':
        # famous figure-8 solution
        # values from the original paper (Chenciner & Montgomery I think)
        pos = [[-0.97, 0.243, 0], [0.97, -0.243, 0], [0, 0, 0]]
        vel = [[0.466, 0.432, 0], [0.466, 0.432, 0], [-0.932, -0.864, 0]]
        mass = [1, 1, 1]
        names = ['A', 'B', 'C']
        
    elif preset == 'Random':
        np.random.seed(42)  # reproducible
        pos = (np.random.rand(n_bodies, 3) - 0.5) * 2
        vel = (np.random.rand(n_bodies, 3) - 0.5) * 0.5
        mass = np.random.rand(n_bodies) * 5 + 0.5
        names = [f'B{i}' for i in range(n_bodies)]
    
    pos = np.array(pos)
    vel = np.array(vel)
    mass = np.array(mass)
    
    if st.sidebar.button('Run Simulation'):
        sim = Sim(pos, vel, mass)
        sim.g = g_scale
        
        hist = [sim.pos.copy()]
        energies = [total_energy(sim.pos, sim.vel, sim.m, sim.g)]
        
        prog = st.progress(0)
        for i in range(steps):
            sim.step(dt, method)
            hist.append(sim.pos.copy())
            if i % 10 == 0:
                energies.append(total_energy(sim.pos, sim.vel, sim.m, sim.g))
            prog.progress((i+1) / steps)
        
        hist = np.array(hist)
        times = np.arange(len(energies)) * dt * 10
        
        # save to session state
        st.session_state['hist'] = hist
        st.session_state['energies'] = energies
        st.session_state['times'] = times
        st.session_state['names'] = names
        st.session_state['method'] = method
    
    # show results
    if 'hist' in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(orbit_2d(st.session_state['hist'], st.session_state['names']), use_container_width=True)
        with c2:
            st.plotly_chart(energy_plot(st.session_state['times'], st.session_state['energies'], 
                                        st.session_state['method']), use_container_width=True)
        
        st.plotly_chart(animated_orbits(st.session_state['hist'], st.session_state['names']), use_container_width=True)
        
        with st.expander('3D View'):
            st.plotly_chart(orbit_3d(st.session_state['hist'], st.session_state['names']), use_container_width=True)
        
        with st.expander('Phase Space'):
            idx = st.selectbox('Body', range(len(st.session_state['names'])))
            vel_hist = np.diff(st.session_state['hist'], axis=0)
            if len(vel_hist) > 0:
                st.plotly_chart(phase_space(st.session_state['hist'][:-1], vel_hist, idx), use_container_width=True)
        
        with st.expander('Compare Integrators'):
            st.write('Compare energy drift across methods')
            if st.button('Run Comparison'):
                methods = ['euler', 'verlet', 'leapfrog', 'rk4']
                edicts = {}
                for m in methods:
                    s = Sim(pos.copy(), vel.copy(), mass.copy())
                    s.g = g_scale
                    es = [total_energy(s.pos, s.vel, s.m, s.g)]
                    for _ in range(1000):
                        s.step(0.01, m)
                        es.append(total_energy(s.pos, s.vel, s.m, s.g))
                    edicts[m] = es
                
                ts = np.arange(len(es)) * 0.01
                st.plotly_chart(compare_methods(ts, edicts), use_container_width=True)


# ==================== LAGRANGE TAB ====================
elif tab == 'Lagrange Points':
    st.header('Lagrange Point Analysis')
    st.markdown('*CR3BP - Circular Restricted Three-Body Problem*')
    
    mu = st.sidebar.slider('μ = M₂/(M₁+M₂)', 0.001, 0.5, 0.01, format='%.4f')
    show_zvc = st.sidebar.checkbox('Zero-Velocity Curves', False)
    cj_val = st.sidebar.slider('Jacobi C_J', 2.5, 4.0, 3.0) if show_zvc else None
    
    pts = lagrange_pts(mu)
    X, Y, Z = potential_grid(mu)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if show_zvc:
            import plotly.graph_objects as go
            X2, Y2, Csurf, allowed = zero_velocity_curves(mu, cj_val)
            fig = go.Figure()
            fig.add_trace(go.Contour(x=X2[0], y=Y2[:,0], z=Csurf, colorscale='Plasma',
                                      contours=dict(start=2.5, end=4, size=0.1), showscale=True))
            # forbidden zone
            fig.add_trace(go.Heatmap(x=X2[0], y=Y2[:,0], z=np.where(~allowed, 1, np.nan),
                                      colorscale=[[0,'rgba(50,50,50,0.5)'],[1,'rgba(50,50,50,0.5)']],
                                      showscale=False))
            for n, (lx, ly) in pts.items():
                fig.add_trace(go.Scatter(x=[lx], y=[ly], mode='markers+text',
                                          marker=dict(size=10, color='red'), text=[n], textposition='top center'))
            fig.add_trace(go.Scatter(x=[-mu], y=[0], mode='markers', marker=dict(size=18, color='yellow')))
            fig.add_trace(go.Scatter(x=[1-mu], y=[0], mode='markers', marker=dict(size=10, color='cyan')))
            fig.update_layout(template='plotly_dark', title=f'ZVC (C_J={cj_val:.2f})',
                              xaxis=dict(scaleanchor='y'), height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(lagrange_contour(X, Y, Z, pts, mu), use_container_width=True)
    
    with col2:
        st.subheader('L-Points')
        for n, (x, y) in pts.items():
            st.write(f'**{n}**: ({x:.4f}, {y:.4f})')
        
        st.divider()
        
        crit = critical_mu()
        st.metric('Critical μ', f'{crit:.4f}')
        if is_stable(mu):
            st.success('L4/L5 STABLE')
        else:
            st.error('L4/L5 UNSTABLE')
        
        st.divider()
        st.subheader('Eigenvalues (L4)')
        for i, e in enumerate(stability_eigenvalues(mu)):
            if isinstance(e, complex):
                st.write(f'λ{i+1} = {e.real:.4f} ± {abs(e.imag):.4f}i')
            else:
                st.write(f'λ{i+1} = {e:.4f}')
    
    # particle sim
    st.divider()
    st.subheader('Test Particle')
    
    sc1, sc2 = st.columns(2)
    with sc1:
        start_pt = st.selectbox('Start near', ['L1', 'L2', 'L3', 'L4', 'L5', 'Custom'])
        if start_pt == 'Custom':
            x0 = st.number_input('x₀', value=0.5)
            y0 = st.number_input('y₀', value=0.5)
        else:
            lx, ly = pts[start_pt]
            offset = st.slider('Offset', 0.0, 0.2, 0.01)
            x0 = lx + offset
            y0 = ly + offset * 0.1
        
        vx0 = st.number_input('vₓ₀', value=0.0)
        vy0 = st.number_input('vᵧ₀', value=0.0)
    
    with sc2:
        sim_dt = st.slider('dt', 0.001, 0.1, 0.01, format='%.3f')
        sim_steps = st.slider('Steps', 100, 20000, 5000)
        view_3d = st.checkbox('3D View', False, key='lg3d')
        anim = st.checkbox('Animate', False, key='lganim')
        
        if st.button('Run'):
            tx, ty, jac = simulate_cr3bp(x0, y0, vx0, vy0, mu, sim_dt, sim_steps)
            
            import plotly.graph_objects as go
            
            if view_3d:
                tz = np.linspace(0, sim_steps * sim_dt, len(tx))
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines', line=dict(color='#00ff88', width=3)))
                for n, (lx, ly) in pts.items():
                    fig.add_trace(go.Scatter3d(x=[lx]*2, y=[ly]*2, z=[0, tz[-1]], mode='lines',
                                               line=dict(color='orange', dash='dash')))
                fig.update_layout(template='plotly_dark', title='3D (z=time)')
            elif anim:
                # animation, this was a pain
                frames = []
                step = max(1, len(tx) // 100)
                for i in range(0, len(tx), step):
                    frames.append(go.Frame(data=[
                        go.Scatter(x=tx[:i+1], y=ty[:i+1], mode='lines', line=dict(color='#00ff88')),
                        go.Scatter(x=[tx[i]], y=[ty[i]], mode='markers', marker=dict(color='white', size=8))
                    ]))
                fig = go.Figure(data=[go.Scatter(x=[tx[0]], y=[ty[0]], mode='markers', marker=dict(color='green', size=10))],
                                frames=frames)
                for n, (lx, ly) in pts.items():
                    fig.add_trace(go.Scatter(x=[lx], y=[ly], mode='markers+text', marker=dict(size=8, color='orange'),
                                              text=[n], textposition='top center'))
                fig.add_trace(go.Scatter(x=[-mu], y=[0], mode='markers', marker=dict(size=15, color='yellow')))
                fig.add_trace(go.Scatter(x=[1-mu], y=[0], mode='markers', marker=dict(size=8, color='cyan')))
                fig.update_layout(template='plotly_dark', title='Animated',
                                  xaxis=dict(scaleanchor='y', range=[min(tx)-0.1, max(tx)+0.1]),
                                  yaxis=dict(range=[min(ty)-0.1, max(ty)+0.1]),
                                  updatemenus=[dict(type='buttons', buttons=[
                                      dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50))])])])
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tx, y=ty, mode='lines', line=dict(color='#00ff88')))
                fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode='markers', marker=dict(color='green', size=10)))
                fig.add_trace(go.Scatter(x=[tx[-1]], y=[ty[-1]], mode='markers', marker=dict(color='red', size=10)))
                for n, (lx, ly) in pts.items():
                    fig.add_trace(go.Scatter(x=[lx], y=[ly], mode='markers+text', marker=dict(size=8, color='orange'),
                                              text=[n], textposition='top center'))
                fig.add_trace(go.Scatter(x=[-mu], y=[0], mode='markers', marker=dict(size=15, color='yellow')))
                fig.add_trace(go.Scatter(x=[1-mu], y=[0], mode='markers', marker=dict(size=8, color='cyan')))
                fig.update_layout(template='plotly_dark', title='Trajectory', xaxis=dict(scaleanchor='y'), height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # jacobi plot
            times = np.arange(len(jac)) * sim_dt
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=times, y=jac, mode='lines', line=dict(color='#ff6b6b')))
            fig2.update_layout(template='plotly_dark', title='Jacobi Constant', xaxis_title='t', yaxis_title='C_J')
            st.plotly_chart(fig2, use_container_width=True)
    
    # advanced stuff
    st.divider()
    with st.expander('Advanced (Chaos)'):
        ac1, ac2 = st.columns(2)
        
        with ac1:
            st.markdown('**Flow Field**')
            if st.button('Generate Flow'):
                Xf, Yf, Uf, Vf = flow_field(mu)
                import plotly.figure_factory as ff
                import plotly.graph_objects as go
                fig = ff.create_quiver(Xf.flatten(), Yf.flatten(), Uf.flatten(), Vf.flatten(),
                                       scale=0.1, arrow_scale=0.3, line_color='cyan')
                fig.add_trace(go.Scatter(x=[-mu], y=[0], mode='markers', marker=dict(color='yellow', size=10)))
                fig.add_trace(go.Scatter(x=[1-mu], y=[0], mode='markers', marker=dict(color='cyan', size=8)))
                fig.update_layout(template='plotly_dark', xaxis=dict(scaleanchor='y'), height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with ac2:
            st.markdown('**Poincaré Section**')
            if st.button('Generate Poincaré'):
                target_cj = cj_val if show_zvc else 3.0
                pp = poincare_section(mu, target_cj)
                if len(pp) > 0:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pp[:,0], y=pp[:,1], mode='markers', marker=dict(size=2, color='#ff00ff')))
                    fig.update_layout(template='plotly_dark', title=f'Poincaré (C_J={target_cj:.2f})',
                                      xaxis_title='x', yaxis_title='vₓ', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('No stable orbits found')


# ==================== NASA TAB ====================
elif tab == 'NASA Data':
    st.header('NASA Horizons Data')
    st.info('Load real solar system data from JPL')
    
    try:
        from nasa_stuff import get_body, get_solar_system, get_mass, BODIES, normalize_for_sim
        
        st.sidebar.subheader('Fetch Body')
        body = st.sidebar.selectbox('Body', list(BODIES.keys()))
        epoch = st.sidebar.text_input('Date', '2024-01-01').strip()
        
        if st.sidebar.button('Fetch'):
            with st.spinner('Fetching...'):
                data = get_body(body, epoch)
                if data:
                    st.success(f'Got {body}')
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader('Position (km)')
                        st.write(f'x: {data["pos"][0]:.2e}')
                        st.write(f'y: {data["pos"][1]:.2e}')
                        st.write(f'z: {data["pos"][2]:.2e}')
                    with c2:
                        st.subheader('Velocity (km/s)')
                        st.write(f'vx: {data["vel"][0]:.4f}')
                        st.write(f'vy: {data["vel"][1]:.4f}')
                        st.write(f'vz: {data["vel"][2]:.4f}')
                    st.metric('Mass', f'{get_mass(body):.3e} kg')
                else:
                    st.error('Failed')
        
        st.divider()
        st.subheader('Solar System Simulation')
        
        st.sidebar.subheader('Sim Settings')
        nasa_int = st.sidebar.selectbox('Integrator', ['verlet', 'leapfrog', 'rk4', 'euler'], key='nasaint')
        nasa_dur = st.sidebar.slider('Duration (yrs)', 0.5, 5.0, 1.0, 0.5, key='nasadur')
        nasa_stp = st.sidebar.slider('Steps/year', 100, 1000, 365, key='nasasteps')
        
        st.sidebar.subheader('View')
        nasa_3d = st.sidebar.checkbox('3D', False, key='nasa3d')
        nasa_anim = st.sidebar.checkbox('Animate', False, key='nasaanim')
        nasa_energy = st.sidebar.checkbox('Energy', False, key='nasae')
        
        if st.button('Load & Simulate'):
            with st.spinner('Fetching planets...'):
                data = get_solar_system(epoch)
                if data:
                    pos = np.array([d['pos'] for d in data])
                    vel = np.array([d['vel'] for d in data])
                    names = [d['name'] for d in data]
                    mass = np.array([get_mass(n) for n in names])
                    
                    pos_n, vel_n, mass_n, scales = normalize_for_sim(pos, vel, mass)
                    
                    st.write(f'Loaded: {names}')
                    
                    total = int(nasa_dur * nasa_stp)
                    dt = 2 * np.pi * nasa_dur / total
                    
                    sim = Sim(pos_n, vel_n, mass_n)
                    hist = sim.run(dt, total, nasa_int)
                    
                    if nasa_3d:
                        st.plotly_chart(orbit_3d(hist, names), use_container_width=True)
                    elif nasa_anim:
                        st.plotly_chart(animated_orbits(hist, names), use_container_width=True)
                    else:
                        st.plotly_chart(orbit_2d(hist, names), use_container_width=True)
                    
                    if nasa_energy:
                        st.plotly_chart(energy_plot(hist, mass_n), use_container_width=True)
                        
    except ImportError as e:
        st.warning(f'NASA module not available: {e}')
        st.info('pip install astroquery')
