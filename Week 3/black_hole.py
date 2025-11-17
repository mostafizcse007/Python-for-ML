import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------
# Utilities: Schwarzschild metric and Christoffel (analytic partials)
# ---------------------------
M = 1.0                      # black hole mass (geometric units G=c=1)
r_s = 2.0 * M

def metric_schwarzschild(r, theta):
    """Return metric g_{mu,nu} and inverse g^{mu,nu} at (r,theta).
       Coordinates: (t, r, theta, phi) -> indices 0..3
    """
    f = 1.0 - 2.0*M / r
    g = np.zeros((4,4))
    g[0,0] = -f
    g[1,1] = 1.0 / f
    g[2,2] = r*r
    g[3,3] = r*r * (np.sin(theta)**2)
    # inverse
    gi = np.zeros((4,4))
    gi[0,0] = -1.0 / f
    gi[1,1] = f
    gi[2,2] = 1.0 / (r*r)
    gi[3,3] = 1.0 / (r*r * (np.sin(theta)**2))
    return g, gi

def christoffel_schwarzschild(r, theta):
    """Compute nonzero Christoffel symbols Γ^μ_{αβ} for Schwarzschild metric.
       We'll return full Γ of shape (4,4,4) but exploit metric symmetries.
       Derived analytically; only r and theta dependence present.
    """
    f = 1.0 - 2.0*M / r
    df_dr = 2.0*M / (r*r)   # derivative of (1-2M/r) is +2M/r^2 for g_tt = -(1-2M/r) etc.
    # Note: careful sign convention used in metric_schwarzschild; these Γ match the geodesic eqn below.
    Gamma = np.zeros((4,4,4))
    # nonzero components (symmetric in lower two indices)
    # Gamma^t_{tr} = Gamma^t_{rt} = (M)/(r*(r-2M))  = df_dr/(2*f)
    Gamma[0,0,1] = Gamma[0,1,0] = M / (r*(r-2.0*M))
    # Gamma^r_{tt} = f * M / r^2   but we compute exact:
    Gamma[1,0,0] = M * (r - 2.0*M) / (r**3)
    # Gamma^r_{rr} = -M/(r*(r-2M))
    Gamma[1,1,1] = -M / (r*(r - 2.0*M))
    # Gamma^r_{θθ} = -(r - 2M)
    Gamma[1,2,2] = -(r - 2.0*M)
    # Gamma^r_{φφ} = -(r - 2M) * sin^2 θ
    Gamma[1,3,3] = -(r - 2.0*M) * (np.sin(theta)**2)
    # Gamma^θ_{rθ} = Gamma^θ_{θr} = 1/r
    Gamma[2,1,2] = Gamma[2,2,1] = 1.0 / r
    # Gamma^θ_{φφ} = - sinθ cosθ
    Gamma[2,3,3] = -np.sin(theta)*np.cos(theta)
    # Gamma^φ_{rφ} = Gamma^φ_{φr} = 1/r
    Gamma[3,1,3] = Gamma[3,3,1] = 1.0 / r
    # Gamma^φ_{θφ} = Gamma^φ_{φθ} = cotθ
    Gamma[3,2,3] = Gamma[3,3,2] = np.cos(theta) / np.sin(theta)
    return Gamma

# ---------------------------
# Geodesic ODE (2nd-order -> first-order system)
# state y = [t, r, theta, phi, vt, vr, vtheta, vphi]
# vt = dt/dλ, etc.
# dy/dλ = [vt, vr, vtheta, vphi,
#          -Γ^t_{αβ} v^α v^β, -Γ^r_{αβ} v^α v^β, ...]
# ---------------------------
def geodesic_rhs(lambda_, y):
    t, r, th, ph, vt, vr, vth, vph = y
    Gamma = christoffel_schwarzschild(r, th)
    v = np.array([vt, vr, vth, vph])
    acc = np.zeros(4)
    # -Gamma^mu_{alpha beta} v^alpha v^beta
    for mu in range(4):
        s = 0.0
        for a in range(4):
            for b in range(4):
                s += -Gamma[mu,a,b] * v[a] * v[b]
        acc[mu] = s
    dydl = np.concatenate([v, acc])
    return dydl

# ---------------------------
# Camera / tetrad at observer: map pixel directions -> coordinate velocities
# ---------------------------
def camera_tetrad(obs_r, obs_th, obs_ph):
    """Return orthonormal tetrad basis vectors e_(a)^mu at observer (contravariant components).
       We'll return a 4x4 array E[a,mu] where a=0..3 labels local frame (t_hat, x_hat, y_hat, z_hat)
       and mu indexes coordinate basis (t,r,theta,phi).
       We assume a static observer (4-velocity aligned with timelike Killing vector).
    """
    f = 1.0 - 2.0*M / obs_r
    sqrt_f = np.sqrt(f)
    # e_(0)^mu = 1/sqrt(f) ∂_t
    e0 = np.array([1.0/np.sqrt(f), 0.0, 0.0, 0.0])
    # e_(r)^mu = sqrt(f) ∂_r
    e_r = np.array([0.0, np.sqrt(f), 0.0, 0.0])
    # e_(th)^mu = (1/r) ∂_θ
    e_th = np.array([0.0, 0.0, 1.0/obs_r, 0.0])
    # e_(ph)^mu = (1/(r sinθ)) ∂_φ
    e_ph = np.array([0.0, 0.0, 0.0, 1.0/(obs_r * np.sin(obs_th))])
    # Build camera axes: choose forward = -e_r (towards BH), right = e_ph, up = e_th
    # local frame axes for screen: x_right = e_ph, y_up = e_th, z_forward = -e_r
    E = np.vstack([e0, e_ph, e_th, -e_r])  # rows: local 0..3 ; columns: coordinate indices mu
    return E

def pixel_to_initial_state(i, j, width, height, fov_rad, obs_pos_sph):
    """Map pixel to initial coordinate state y0 for solve_ivp.
       obs_pos_sph = (r_obs, theta_obs, phi_obs)
    """
    r_obs, th_obs, ph_obs = obs_pos_sph
    E = camera_tetrad(r_obs, th_obs, ph_obs)
    # normalized screen coords in camera local frame:
    # x in [-1,1] left->right, y in [-1,1] bottom->top
    x = ( (i + 0.5) / width - 0.5 ) * 2.0
    y = ( (j + 0.5) / height - 0.5 ) * 2.0
    aspect = width / height
    x *= np.tan(fov_rad/2) * aspect
    y *= np.tan(fov_rad/2)
    # local unit direction vector in camera frame (right, up, forward)
    # forward is negative z in earlier code; here we use (x, y, 1) pointing "into scene"
    # But in local orthonormal frame the null vector must satisfy |spatial| = k^0 for null.
    local_spatial = np.array([x, y, 1.0])
    local_spatial /= np.linalg.norm(local_spatial)  # unit spatial direction
    # set local null 4-vector k^(a): k^(0) = 1, k^(i) = local_spatial * 1  => ensures null in orthonormal frame
    k_local = np.concatenate([[1.0], local_spatial])   # (k^0, k^1, k^2, k^3)
    # convert to coordinate basis: k^mu = sum_a E[a,mu] * k^(a)
    k_coord = np.dot(k_local, E)   # shape (4,)
    # initial coordinate position (t, r, th, ph) ; pick t=0
    y0_pos = np.array([0.0, r_obs, th_obs, ph_obs])
    y0_vel = k_coord.copy()
    # return combined state y0
    y0 = np.concatenate([y0_pos, y0_vel])
    return y0

# ---------------------------
# Disk / emitter model & redshift
# ---------------------------
def keplerian_omega(r):
    """Keplerian angular velocity for circular equatorial orbits (Schwarzschild): Ω = sqrt(M/r^3)"""
    return np.sqrt(M / r**3)

def emitter_four_velocity(r, th, ph):
    """Return 4-velocity u^mu of emitter on circular prograde equatorial orbit at (r,theta=pi/2).
       Only valid for theta ~ pi/2.
    """
    assert abs(th - np.pi/2) < 1e-6, "emitter in equatorial plane"
    Omega = keplerian_omega(r)
    f = 1.0 - 2.0*M/r
    gphph = r*r
    # u^t = 1 / sqrt( -g_tt - 2 g_tφ Ω - g_φφ Ω^2); in Schwarzschild g_tφ = 0
    ut = 1.0 / np.sqrt( -(-f) - gphph * (Omega**2) )
    uphi = Omega * ut
    u = np.array([ut, 0.0, 0.0, uphi])
    return u

def observed_redshift(k_at_em, u_em, k_at_obs, u_obs):
    """Compute redshift g = (k·u_em)/(k·u_obs). We need covariant components k_mu = g_{mu,nu} k^nu."""
    # compute covariant components at emitter and observer using local metric
    # here we pass coordinate k^mu at event location and u^mu at emitter/observer location
    # convert to covariant: k_mu = g_{mu,nu} k^nu
    # but note we must supply metric at the given r,theta for emitter and obs
    # k_at_em: k^mu at emitter location (4,)
    # k_at_obs: k^mu at observer location (4,)
    # u_em, u_obs given
    # return scalar g
    # compute for emitter
    # caller will compute metric and pass appropriate arguments
    raise NotImplementedError("We compute redshift in event handler where we have metric available.")

# ---------------------------
# Event functions for solve_ivp
# ---------------------------
def event_horizon(lambda_, y):
    """Event when r - r_s = 0 (photon hits horizon)."""
    r = y[1]
    return r - r_s
event_horizon.terminal = True
event_horizon.direction = -1.0  # decreasing r

def event_equator(lambda_, y):
    """Event when theta - pi/2 = 0 (cross equatorial plane)"""
    th = y[2]
    return th - 0.5*np.pi
event_equator.terminal = True
# direction both -> detect both crossing directions
event_equator.direction = 0.0

# ---------------------------
# Rendering loop: trace pixel-by-pixel (slow but clear)
# ---------------------------
def render_image(width=300, height=200, fov_deg=30.0, r_obs=15.0, th_obs=np.deg2rad(60.0), ph_obs=0.0):
    fov_rad = np.deg2rad(fov_deg)
    image = np.zeros((height, width, 3), dtype=np.float32)

    obs_pos_sph = (r_obs, th_obs, ph_obs)
    # observer 4-velocity (static observer) u_obs = e_t_hat = 1/sqrt(f) ∂_t
    f_obs = 1.0 - 2.0*M / r_obs
    u_obs = np.array([1.0 / np.sqrt(f_obs), 0.0, 0.0, 0.0])

    for j in range(height):
        for i in range(width):
            y0 = pixel_to_initial_state(i, j, width, height, fov_rad, obs_pos_sph)
            # integrate geodesic forward in λ until horizon or equator crossing or large radius
            lam_max = 1e4
            try:
                sol = solve_ivp(geodesic_rhs, (0.0, lam_max), y0, rtol=1e-7, atol=1e-9,
                                events=[event_horizon, event_equator], max_step=1.0)
            except Exception as e:
                # numerical failure -> black pixel
                image[height-1-j, i, :] = 0.0
                continue

            # check which event triggered
            if sol.status == 1 and sol.t_events[0].size > 0:
                # hit horizon
                image[height-1-j, i, :] = 0.0
                continue
            elif sol.status == 1 and sol.t_events[1].size > 0:
                # equator crossing occurred -> evaluate emission there
                # get state at event
                teq = sol.t_events[1][0]
                y_eq = sol.sol(teq)
                # y_eq: [t, r, th, ph, vt, vr, vth, vph]
                r_hit = y_eq[1]
                ph_hit = y_eq[3] % (2*np.pi)
                # ensure radial is within disk radii
                r_in = 6.0 * M   # ISCO-ish inner
                r_out = 30.0 * M
                if r_hit >= r_in and r_hit <= r_out:
                    # compute emitted intensity: simple power law emissivity j(r) ∝ r^-2
                    I_em = (r_hit**-2.0)
                    # compute redshift g = (k·u_em) / (k·u_obs)
                    # need k_at_em covariant and k_at_obs covariant
                    # k^mu at event is the velocity components y_eq[4:8]
                    k_em = y_eq[4:8]
                    # compute metric at emitter
                    g_em, gi_em = metric_schwarzschild(r_hit, 0.5*np.pi)
                    # covariant k_mu = g_{mu,nu} k^nu
                    kcov_em = g_em.dot(k_em)
                    # emitter four-velocity (Keplerian)
                    u_em = emitter_four_velocity(r_hit, 0.5*np.pi, ph_hit)
                    # compute k·u at emitter
                    kup_em = np.dot(kcov_em, u_em)
                    # compute k·u at observer: at y0 initial state
                    k_obs = y0[4:8]
                    g_obs, gi_obs = metric_schwarzschild(r_obs, th_obs)
                    kcov_obs = g_obs.dot(k_obs)
                    kup_obs = np.dot(kcov_obs, u_obs)
                    g_factor = kup_em / kup_obs
                    # observed specific intensity scales as I_obs = g^3 * I_em (for optically thin, frequency integrated approx)
                    I_obs = max(0.0, (g_factor**3) * I_em)
                    # simple color mapping: Doppler brightening blue on approaching side
                    # sign of azimuthal velocity dot photon direction determines blue/red
                    # approximate: if u_em^phi * k_em^phi < 0 -> approaching
                    doppler = np.clip(1.0 + 2.0*(g_factor-1.0), 0.2, 3.0)
                    color = np.array([1.0, 0.65, 0.25]) * doppler * (I_obs / (I_obs + 1.0))
                    image[height-1-j, i, :] = np.clip(color, 0.0, 1.0)
                    continue
                else:
                    # crossed equator outside disk -> sample background
                    pass

            # else: no interesting event -> background sky
            # use pixel direction in local frame as gradient
            # map pixel to direction vector for background color
            # Recompute local unit direction (fast)
            xloc = ( (i + 0.5)/width - 0.5)*2.0 * np.tan(fov_rad/2) * (width/height)
            yloc = ( (j + 0.5)/height - 0.5)*2.0 * np.tan(fov_rad/2)
            v = np.array([xloc, yloc, 1.0])
            v /= np.linalg.norm(v)
            sky_color = 0.5 + 0.5*v[1]   # simple gradient using y component
            image[height-1-j, i, :] = np.array([sky_color*0.6, sky_color*0.7, sky_color*0.9])

    return image

# ---------------------------
# Run and display (slow)
# ---------------------------
if __name__ == "__main__":
    W = 300
    H = 200
    img = render_image(width=W, height=H, fov_deg=30.0, r_obs=15.0, th_obs=np.deg2rad(60.0))
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Schwarzschild backward ray-tracing: thin equatorial disk")
    plt.show()
