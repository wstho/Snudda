TITLE Chloride dynamics

NEURON {
    SUFFIX cl_dynas
    USEION cl READ icl, cli WRITE cli VALENCE -1
    RANGE pump, clinf, taur, drive
}

UNITS {
    (molar) = (1/liter) 
    (mM) = (millimolar)
    (um) = (micron)
    (mA) = (milliamp)
    (msM) = (ms mM)
    FARADAY = (faraday) (coulomb)
}

PARAMETER {
    drive = 10000 (1)
    depth = 0.1 (um)
    clinf = 10 (mM)
    taur = 43 (ms)
    kt = 1e-4 (mM/ms)
    kd = 1e-4 (mM)
    pump = 0.02
}

STATE { cli (mM) }

INITIAL { cli = clinf }

ASSIGNED {
    icl (mA/cm2)
    drive_channel (mM/ms)
    drive_pump (mM/ms)
}
    
BREAKPOINT {
    SOLVE state METHOD cnexp
}

DERIVATIVE state { 
    drive_channel = -drive*icl/(2*FARADAY*depth)
    drive_pump    = -kt*(cli-clinf)/(cli+kd)
    
    if (drive_channel <= 0.) { drive_channel = 0. }
    
    cli' = drive_channel + pump*drive_pump + (clinf-cli)/taur

}

COMMENT

Original NEURON model by Wolf (2005) and Destexhe (1992).  Adaptation by
Alexander Kozlov <akozlov@kth.se>.

[1] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

ENDCOMMENT
