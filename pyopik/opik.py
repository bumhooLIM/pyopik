"""
pyopik: A Python module for calculating intrinsic collision probabilities.

This module implements the algorithm described in:
Dell'Oro, A., & Paolicchi, P. (1998). "Statistical Properties of Encounters among Asteroids: 
[cite_start]A New, General Purpose, Formalism". Icarus, 136, 328-339. [cite: 1]

It calculates the intrinsic collision probability (P_i) and mean impact velocity (U)
between two Keplerian orbits using a Monte Carlo integration of the Kessler/Opik formalism.
"""

import numpy as np
import itertools

#
# Constants
#
M_TO_AU = 6.68459e-12              # meter to au                    [au m(-1)]
KM_TO_AU = 1e+3 * M_TO_AU          # km to au                       [au km(-1)]
YR_TO_SEC = 3.154e+7               # year to second                 [s yr(-1)]
MU_SUN = 1.327124e+20 * M_TO_AU**3 # solar gravitational parameter  [au(3) s(-2)]

def _kepler_velocity(a, e, i, f, w, o, mu=MU_SUN):
    """
    Internal helper: Return Cartesian-based velocity vector from Keplerian elements.
    
    Parameters:
        a (float): heliocentric distance [au]
        e (float): eccentricity []
        i (float): inclination [rad]
        f (float): true anomaly [rad]
        w (float): argument of perihelion [rad]
        o (float): longitude of the ascending node [rad]
        mu (float): gravitational parameter (default: Sun)

    Returns:
        tuple: (vx, vy, vz) in [au s(-1)]
    """
    p = a * (1 - e**2)
    # Radial distance calculation is done outside or implicitly handled by 'p' usage 
    # but strictly speaking velocity depends on a, e. 
    # Note: The input 'a' here functions as semimajor axis in the vis-viva context
    # for mean motion, but the specific position is defined by 'f'.
    
    # Pre-compute trig
    cf = np.cos(f)
    sf = np.sin(f)
    cw = np.cos(w)
    sw = np.sin(w)
    co = np.cos(o)
    so = np.sin(o)
    ci = np.cos(i)
    si = np.sin(i)

    # Orbital plane coordinates components
    # radial and tangential components could be used, but using direct rotation matrix application:
    # This matches the Dell'Oro Appendix logic or standard celestial mechanics.
    # The original code implementation:
    term1 = np.sqrt(mu/p)
    vx = -term1 * (co*(np.sin(w+f)+e*sw) + so*(np.cos(w+f)+e*cw)*ci)
    vy = -term1 * (so*(np.sin(w+f)+e*sw) - co*(np.cos(w+f)+e*cw)*ci)
    vz = term1 * (np.cos(w+f)+e*cw)*si

    return vx, vy, vz


def opik_probability(a, e, i, a0, e0, i0, N=10000, return_velocity=True):
    """
    Calculate intrinsic collisional probability between two orbits.

    This function performs a Monte Carlo integration over the target orbit (index 0)
    to find intersection points with the projectile orbit (no index), solving for
    [cite_start]geometric intersection as defined in Dell'Oro & Paolicchi (1998) Eq 9 and Appendix. [cite: 112, 440]

    Parameters:
        a  (float): Semi-major axis of object 1 [au]
        e  (float): Eccentricity of object 1
        i  (float): Inclination of object 1 [rad]
        a0 (float): Semi-major axis of object 2 [au]
        e0 (float): Eccentricity of object 2
        i0 (float): Inclination of object 2 [rad]
        N  (int)  : Number of Monte Carlo samples (default: 10000)
        return_velocity (bool): If True, returns (P_i, U). If False, returns P_i.

    Returns:
        P_i (float): Intrinsic collisional probability [km(-2) yr(-1)]
        U   (float): Mean impact velocity [km s(-1)] (Returned if return_velocity is True)
    """

    # 1. Monte Carlo Sampling of Target (Object 2 / subscript 0)
    # We assume uniform distribution for Mean Anomaly implies weighting by residence time.
    # The code samples True Anomaly (f0) uniformly then applies weighting (DEL) later.
    f0 = 2 * np.pi * np.random.rand(N)
    w0 = 2 * np.pi * np.random.rand(N)
    # [cite_start]
    o0 = np.zeros(N) # Fix node to 0 (relative inclination matters) [cite: 174]

    # Pre-calculate constants for Object 2 (Target)
    p0 = a0 * (1 - e0**2)
    # Pre-calculate constants for Object 1 (Projectile)
    p = a * (1 - e**2)
    
    # [cite_start]2. Target Position in Cartesian Space [cite: 448]
    r = p0 / (1 + e0 * np.cos(f0))
    # Coordinates rotated to a frame where Object 2's node is 0
    # x, y, z calculations
    cw0f0 = np.cos(w0 + f0)
    sw0f0 = np.sin(w0 + f0)
    co0 = np.cos(o0)
    so0 = np.sin(o0)
    ci0 = np.cos(i0)
    si0 = np.sin(i0)

    x = r * cw0f0 * co0 - r * sw0f0 * so0 * ci0
    y = r * cw0f0 * so0 + r * sw0f0 * co0 * ci0
    z = r * sw0f0 * si0

    # [cite_start]3. Geometric Filtering [cite: 451]
    lt = np.arcsin(z/r)   # Heliocentric latitude
    ln = np.arctan2(y, x) # Heliocentric longitude

    # Mask 1: Radial Overlap
    # Check if 'r' is reachable by Object 1 (Projectile)
    # r must be within [q, Q] of Object 1.
    # derived from cos(f) = (1/e)*(p/r - 1) being within [-1, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        mask1 = np.abs((1/e)*(p/r - 1)) < 1
    
    # Mask 2: Vertical Reach
    # Check if Object 1's inclination is high enough to reach latitude 'lt'
    with np.errstate(divide='ignore', invalid='ignore'):
        mask2 = np.abs(np.tan(lt) / np.tan(i)) < 1
    
    mask = mask1 * mask2

    # If no intersections possible in samples, return 0
    if mask.sum() == 0:
        return (0.0, 0.0) if return_velocity else 0.0

    # 4. Filter Arrays
    r_valid = r[mask]
    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = z[mask]
    lt_valid = lt[mask]
    ln_valid = ln[mask]
    f0_valid = f0[mask]
    w0_valid = w0[mask]
    o0_valid = o0[mask]

    # Velocity of Target (Object 2)
    vx0, vy0, vz0 = _kepler_velocity(a0, e0, i0, f0_valid, w0_valid, o0_valid)

    # [cite_start]5. Solve for Projectile (Object 1) Intersection Parameters [cite: 454-457]
    # True Anomaly solutions (f1, f2)
    arg_f = (1/e) * (p/r_valid - 1)
    f1 = np.arccos(arg_f)
    f2 = 2 * np.pi - f1

    # Node solutions (o1, o2)
    # Derived from spherical trigonometry relationship between latitude and inclination
    l_angle = np.arcsin(np.tan(lt_valid) / np.tan(i))
    o1 = ln_valid - l_angle
    o2 = ln_valid + l_angle - np.pi
    
    # Normalize angles to [0, 2pi]
    o1 = np.mod(o1, 2*np.pi)
    o2 = np.mod(o2, 2*np.pi)

    # 6. Integration Summation
    # There are 4 solution combinations for every valid point: (f1, o1), (f1, o2), (f2, o1), (f2, o2)
    FUNC = np.zeros(mask.sum())
    UU = np.zeros(mask.sum())

    for f_sol, o_sol in itertools.product([f1, f2], [o1, o2]):
        
        # Calculate Argument of Perihelion (w) for Projectile to hit point (x,y,z)
        # Using rotation matrix inversion logic
        cos_wf = (1./r_valid) * (x_valid*np.cos(o_sol) + y_valid*np.sin(o_sol))
        sin_wf = (1./r_valid) * (-x_valid*np.sin(o_sol)*np.cos(i) + 
                                 y_valid*np.cos(o_sol)*np.cos(i) + 
                                 z_valid*np.sin(i))
        wf = np.arctan2(sin_wf, cos_wf)
        w_sol = np.mod(wf - f_sol, 2*np.pi)

        # Velocity of Projectile (Object 1)
        vx, vy, vz = _kepler_velocity(a, e, i, f_sol, w_sol, o_sol)

        # [cite_start]Relative Velocity U [cite: 88]
        u_sq = (vx - vx0)**2 + (vy - vy0)**2 + (vz - vz0)**2
        u = np.sqrt(u_sq)
        UU += u

        # [cite_start]Jacobian Determinant (Transformation from elements to Cartesian) [cite: 493]
        # detJ = r^2 * sin(beta) ... 
        # Analytical form: a^3 * (1-e^2)^3 / (1+e*cos f)^4 * e * |sin f * cos(f+w) * sin i|
        detJ = (p**3 / (1 + e*np.cos(f_sol))**4) * e * \
               np.abs(np.sin(f_sol) * cos_wf * np.sin(i))

        # [cite_start]Density Weighting (Residence Time) [cite: 182]
        # Because we sampled f0 uniformly, we must weight by 1/r_dot ~ 1/(1+e*cos f)^2
        del_f = (1 - e**2)**1.5 / (1 + e * np.cos(f_sol))**2 / (2*np.pi)**3
        del_f0 = (1 - e0**2)**1.5 / (1 + e0 * np.cos(f0_valid))**2 / (2*np.pi)**3
        DEL = del_f * del_f0

        # [cite_start]Integrand [cite: 9]
        # P_i integrand = pi * DEL * (U / detJ)
        # Units converted to [km^-2 yr^-1]
        func = np.pi * DEL * (u / detJ) * KM_TO_AU**2 * YR_TO_SEC
        FUNC += func

    # 7. Final Integration
    # Average over N samples (Monte Carlo)
    P_i = ((2*np.pi)**3 / N) * np.sum(FUNC)

    # Mean Velocity
    # Weighted average of velocity by the probability flux
    if return_velocity:
        if np.sum(FUNC) > 0:
            # Note: The original code divides by 4. This is because UU summed 4 solutions, 
            # and FUNC summed 4 solutions. The weights cancel out correctly.
            U_avg = np.average(UU, weights=FUNC) / 4 / KM_TO_AU
        else:
            U_avg = 0.0
        return P_i, U_avg
    
    return P_i