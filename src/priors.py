
import numpy as np
from scipy.stats import cauchy

def t_student_prior_truncated(theta, scale=1.0, shift=0.0, vmin = -10, vmax = 10):
    slope = theta
    if slope <= vmin or slope >= vmax:
        return -np.inf
    log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
    return log_prior_slope

def t_student_prior(theta, scale=1.0, shift=0.0,vmin = -10, vmax = 10):
    slope = theta
    log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
    return log_prior_slope

def t_student_prior_damping(theta, scale = 1.0, shift = 0.0, damping = 1.0, vmin = -10, vmax = 10):
    slope = theta
    if slope <= vmin or slope >= vmax:
        return -np.inf
    log_prior_slope = cauchy.logpdf((slope - shift) / scale, loc=0, scale=scale)
    if slope > shift:
        log_prior_slope -=  damping * (slope - shift)
    return log_prior_slope

def gaussian_prior(theta, mu = 0, sigma = 1):
    return -0.5 * ((theta- mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

def gaussian_prior_truncated(theta, mu = 0, sigma = 1, vmin = -3, vmax = 3):
    if theta <= vmin or theta >= vmax:
        return -np.inf
    p = -0.5 * ((theta- mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
    p = -np.inf if np.isnan(p) == True or np.isfinite(p) == False else p 
    return p

def flat_prior(theta, vmin, vmax):
    if (vmin <= theta <= vmax):
        return 0.0
    else:
        return -np.inf


