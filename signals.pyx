# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3, boundscheck=False, wraparound=False, binding=True, embedsignature=True

from __future__ import division
from numpy.math cimport INFINITY
from libc.math cimport sqrt, cos, M_PI
cimport numpy as np
import numpy as np
from raynest.parameter cimport LivePoint
cimport cython


cdef double MSUN_SI = 1.9884099021470415e+30
cdef double C_SI    = 299792458.0
cdef double G_SI    = 6.6743e-11
cdef double PC_SI   = 3.085677581491367e+16

def orbital_frequency(np.ndarray time, double m, double t0):
    return np.array([_orbital_frequency(t, m, t0) for t in time])

cdef double _orbital_frequency(double t, double M, double t0 = 30000):
    h = G_SI*MSUN_SI*M/C_SI**2
    prefactor = (C_SI/(16*M_PI))*h**(-5./8.)
    return prefactor*(5./(C_SI*(t0-t)))**(3./8.) #exact expression

def amplitude(np.ndarray time, double M, double t0 = 30000):
    return np.array([_amplitude(t, M, t0 = t0) for t in time])

cdef double _amplitude(double t, double M, double t0 = 30000):
    h = G_SI*MSUN_SI*M/C_SI**2
    return h*(5*G_SI*MSUN_SI*M/((t0-t)*C_SI**3))**0.25
    
def distance(double r0, double z0):
    return _distance(r0, z0)

cdef inline double _distance(double r0, double z0):
    return sqrt(r0**2+z0**2)

def chirp(LivePoint x, np.ndarray t, double t0 = 30000):
    return _chirp(x, t, t0 = t0)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef np.ndarray _chirp(LivePoint x, np.ndarray t, double t0 = 30000):
    
    cdef unsigned int N = t.shape[0]
    cdef unsigned int i
    cdef double D = distance(x['r'],x['z'])
    cdef double phi = x['phi']
    cdef double M   = x['M']
    
    cdef np.ndarray[double, ndim=1, mode='c'] h = np.zeros(N, dtype=np.float64)
    cdef double[:] h_view = h
    
    for i in range(N):
        h_view[i] = (1./(2*D*PC_SI))*_amplitude(t[i], M, t0 = t0)*cos(4.0*M_PI*_orbital_frequency(t[i], M, t0 = t0)+phi)
        
    return h
