TITLE fast KV3

COMMENT

Neuromodulation is added as functions:
    
    modulationDA = 1 + modDA*(maxModDA-1)*levelDA

where:
    
    modDA  [0]: is a switch for turning modulation on or off {1/0}
    maxModDA [1]: is the maximum modulation for this specific channel (read from the param file)
                    e.g. 10% increase would correspond to a factor of 1.1 (100% +10%) {0-inf}
    levelDA  [0]: is an additional parameter for scaling modulation. 
                Can be used simulate non static modulation by gradually changing the value from 0 to 1 {0-1}
									
	  Further neuromodulators can be added by for example:
          modulationDA = 1 + modDA*(maxModDA-1)
	  modulationACh = 1 + modACh*(maxModACh-1)
	  ....

	  etc. for other neuromodulators
	  
	   
								     
[] == default values
{} == ranges
    
ENDCOMMENT


NEURON {
    SUFFIX KV_SNr
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik, a, q
    RANGE modDA, maxModDA, levelDA

}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.0 	(S/cm2) 
    a = 0.8
    :q = 1	: room temperature 22-24 C
    q = 3	: body temperature 33 C
    modDA = 0
    maxModDA = 1
    levelDA = 0

}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    gk (S/cm2)
    minf
    mtau (ms)
    hinf
    htau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gk = gbar*m*m*m*(h*a+1-a)*modulationDA()
    ik = gk*(v-ek)
}

DERIVATIVE states {
    rates()
    m' = (minf-m)/mtau*q
    h' = (hinf-h)/htau*q
}

INITIAL {
    rates()
    m = minf
    h = hinf
}

PROCEDURE rates() {
    UNITSOFF
    minf = 1/(1+exp((v-(-20))/(-7.8)))
    mtau = 1+28*exp(-((v-(-34.3))/30.1)^2)
    hinf = 1/(1+exp((v-(-76))/21.5))
    htau = 0.1/(exp((v-(-76))/(-29.01))+exp((v-(-76))/200))
    : Du 2017
    :LOCAL alpha, beta, sum
    :UNITSOFF
    :alpha = 0.25/(1+exp((v-50)/(-20)))
    :beta = 0.05/(1+exp((v-(-90))/35))
    :sum = alpha+beta
    :minf = alpha/sum
    :mtau = 1/sum
    :
    :alpha = 0.0025/(1+exp((v-(-95))/16))
    :beta = 0.002/(1+exp((v-50)/(-70)))
    :sum = alpha+beta
    :hinf = a+(alpha/sum)*(1-a)
    :htau = 1/sum
    UNITSON
}

FUNCTION modulationDA() {
    : returns modulation factor
    
    modulationDA = 1 + modDA*(maxModDA-1)*levelDA 
}
