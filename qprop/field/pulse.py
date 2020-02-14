"""Routines for describing ultrashort laser pulse"""

from numpy import pi

def place_in_minus_pi_to_pi_range(phase):
    _phase = pi * ( (phase / pi + 1) % 2 - 1 )
    return _phase

from numpy import pi
def get_CEP_for_qprop_pulse(phase, num_cycles, from_minus_pi_to_pi=True):
    _CEP = pi * ((0.5 - num_cycles - phase / pi) % 2)
    if from_minus_pi_to_pi: _CEP = place_in_minus_pi_to_pi_range(_CEP)
    return _CEP

def get_phase_to_set_CEP_as(CEP, num_cycles, from_minus_pi_to_pi=True):
    _phase = pi * ((0.5 - num_cycles - CEP / pi) % 2)
    if from_minus_pi_to_pi: _phase = place_in_minus_pi_to_pi_range(_phase)
    return _phase

