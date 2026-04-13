COMMENT

Chloride accumulation and diffusion with decay (time constant tau) to resting level cli0.
The decay approximates a reversible chloride pump with first order kinetics.
To eliminate the chloride pump, just use this hoc statement
To make the time constant effectively "infinite".
tau and the resting level are both RANGE variables

Adapted from: Lombardi et al 2019
Diffusion model is modified from Ca diffusion model in Hines & Carnevale:
Expanding NEURON with NMODL, Neural Computation 12: 839-851, 2000 (Example 8)

ENDCOMMENT

NEURON {
	SUFFIX Cl_dynamics
	USEION cl READ icl WRITE cli, ecl VALENCE -1 : Ion cl, use cl current to calculate cl internal concentration
	USEION hco3 READ ihco3 WRITE ehco3,hco3i VALENCE -1: Ion HCO3, use HCO3 internal concentration to calculate the external concentration
	GLOBAL vrat		:vrat must be GLOBAL, so it does not change with position. vrat = volumes of concentric shells
	RANGE tau, cli0, clo0, hco3i0, hco3o0, egaba, delta_egaba, init_egaba, ehco3_help, ecl_help,kcc2_p : all of these change with position
}

DEFINE Nannuli 4

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um) = (micron)
	(mA) = (milliamp)
	(mV)    = (millivolt)
	FARADAY = (faraday) (10000 coulomb)
	PI = (pi) (1)
	F = (faraday) (coulombs)
	R = (k-mole)  (joule/degC)
}

PARAMETER {
	DCl = 2 (um2/ms) : Kuner & Augustine, Neuron 27: 447 : diffusion coefficient of cl
    DH = 1.18 (um2/ms) : Hashitani & Kigoshi,  Bulletin of the Chemical Society of Japan, 38:1395-1396
	tau_NKCC1 = 174000 (ms)   : 174 s From Kolbaev, Lombardi kilb (in Prep) - kinetics after Cl decline
	tau_passive = 121000 (ms) : 321 s From Kolbaev, Lombardi Kilb (in prep) - kinetics after bumetanid washin
    tau_hco3 = 1000 (ms) : tau for Bicarbonate, just an arbitrary value
	cli0 = 6 (mM) : basal Cl internal concentration
	clo0 = 133 (mM) : basal Cl external concentration
	hco3i0 = 14.1	(mM) : basal HCO3 internal concentration
	hco3o0 = 24	(mM) : basal HCO3 external concentration
	hco3i_Start = 16 (mM) : HCO3- concentration at start
	celsius = 35    (degC)
    kcc2_p = 0.002

}

ASSIGNED {
	diam 	(um)
	icl 	(mA/cm2) : Cl current
    ihco3 	(mA/cm2) : HCO3- current current
	cli 	(mM) : Cl internal concentration
	hco3i	(mM) : HCO3 internal concentration
	hco3o	(mM) : HCO3 external concentration
	vrat[Nannuli]	: numeric value of vrat[i] equals the volume
			: of annulus i of a 1um diameter cylinder
			: multiply by diam^2 to get volume per um length
	ehco3 	(mV)
	ecl	(mV)
    ecl_help (mV) 
    ActPumpkcc2
    ActPumpnkcc :Binary value that defines if active pumping of passive outward diffusion
}

STATE {
	: cl[0] is equivalent to cli
	: cl[] are very small, so specify absolute tolerance
	cl[Nannuli]	(mM) <1e-10>
    hco3[Nannuli]	(mM) <1e-10>
}


BREAKPOINT {
		SOLVE state METHOD sparse
		ecl = log(cli/clo0)*(1000)*(celsius + 273.15)*R/F
        ecl_help = log(cli/clo0)*(1000)*(celsius + 273.15)*R/F
        ehco3 = log(hco3i/hco3o0)*(1000)*(celsius + 273.15)*R/F
}

LOCAL factors_done

INITIAL {
	if (factors_done == 0) {  	: flag becomes 1 in the first segment
		factors_done = 1	: all subsequent segments will have
		factors()		: vrat = 0 unless vrat is GLOBAL. We make sure that vrat is applied to the shell volumes
	}
	cli = cli0
	hco3i = hco3i0
	hco3o = hco3o0
	FROM i=0 TO Nannuli-1 { : So that at the begining the Cl [] is the same in all shells ( steady state)
		cl[i] = cli
	}
        FROM i=0 TO Nannuli-1 { : So that at the begining the HCO3 [] is the same in all shells ( steady state)
		hco3[i] = hco3i
	}
	ehco3 = log(hco3i/hco3o)*(1000)*(celsius + 273.15)*R/F : Nerst eq for HCO3 at time 0
    ecl_help = log(cli/clo0)*(1000)*(celsius + 273.15)*R/F
	ecl = log(cli/clo0)*(1000)*(celsius + 273.15)*R/F
}

LOCAL frat[Nannuli]	: scales the rate constants for model geometry

PROCEDURE factors() {
	LOCAL r, dr2
	r = 1/2			: starts at edge (half diam), diam = 1, length = 1
	dr2 = r/(Nannuli-1)/2	: full thickness of outermost annulus,
				: half thickness of all other annuli
	vrat[0] = 0
	frat[0] = 2*r		: = diam
	FROM i=0 TO Nannuli-2 {
		vrat[i] = vrat[i] + PI*(r-dr2/2)*2*dr2	: interior half
		r = r - dr2
		frat[i+1] = 2*PI*r/(2*dr2)	: outer radius of annulus Ai+1/delta_r=2PI*r*1/delta_r
						: div by distance between centers
		r = r - dr2
		vrat[i+1] = PI*(r+dr2/2)*2*dr2	: outer half of annulus
	}
}

KINETIC state {
    if (cli0 >= cl[0]) { : Under this condition the NKCC1 mediates active Cl- uptake ( positive inward flux)
		  ActPumpnkcc = 1
          ActPumpkcc2 = 0
		}
		else {     : Under this condition NKCC1 should be not functional ( negative inward flux)
		  ActPumpnkcc = 0
          ActPumpkcc2 = 1

		}

  	COMPARTMENT i, diam*diam*vrat[i] {cl}
		LONGITUDINAL_DIFFUSION i, DCl*diam*diam*vrat[i] {cl}
				~ cl[0] << ((icl*PI*diam/FARADAY) + ActPumpnkcc*(diam*diam*vrat[0]*(cli0 - cl[0])/tau_NKCC1)  - ActPumpkcc2*kcc2_p*(155*cl[0] - 4*clo0)+ (diam*diam*vrat[0]*(cli0 - cl[0])/tau_passive)) : icl is Cl- influx
	 	FROM i=0 TO Nannuli-2 {
		~ cl[i] <-> cl[i+1]	(DCl*frat[i+1], DCl*frat[i+1])
                }
            cli = cl[0]
	  
        COMPARTMENT i, diam*diam*vrat[i] {hco3}
		LONGITUDINAL_DIFFUSION i, DH*diam*diam*vrat[i] {hco3}
				~ hco3[0] << ((ihco3*PI*diam/FARADAY)  + (diam*diam*vrat[0]*(hco3i0 - hco3[0])/tau_hco3)) : ihco3 is HCO3- influx
	 	FROM i=0 TO Nannuli-2 {
		~ hco3[i] <-> hco3[i+1]	(DH*frat[i+1], DH*frat[i+1])
                }
	        hco3i = hco3[0] - 0.007*(155*hco3[0] - 4*hco3o0) +15
}
