

class idx_matpower:
    """
    Define indices with name for convenience
    Value corresponds to the matpower data format
    """

    ## Branch data
    F_BUS = 0  # f, from bus number
    T_BUS = 1  # t, to bus number
    BR_R = 2  # r, resistance (p.u.)
    BR_X = 3  # x, reactance (p.u.)
    BR_B = 4  # b, total line charging susceptance (p.u.)
    RATE_A = 5  # rateA, MVA rating A (long term rating)
    RATE_B = 6  # rateB, MVA rating B (short term rating)
    RATE_C = 7  # rateC, MVA rating C (emergency rating)
    TAP = 8  # ratio, transformer off nominal turns ratio
    SHIFT = 9  # angle, transformer phase shift angle (degrees)
    BR_STATUS = 10  # initial branch status, 1 - in service, 0 - out of service
    ANGMIN = 11  # minimum angle difference, angle(Vf) - angle(Vt) (degrees)
    ANGMAX = 12  # maximum angle difference, angle(Vf) - angle(Vt) (degrees)


    ## Bus data
    BUS_I       = 0    # bus number (1 to 29997)
    BUS_TYPE    = 1    # bus type
    PD          = 2    # Pd, real power demand (MW)
    QD          = 3    # Qd, reactive power demand (MVAr)
    GS          = 4    # Gs, shunt conductance (MW at V = 1.0 p.u.)
    BS          = 5    # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
    BUS_AREA    = 6    # area number, 1-100
    VM          = 7    # Vm, voltage magnitude (p.u.)
    VA          = 8    # Va, voltage angle (degrees)
    BASE_KV     = 9    # baseKV, base voltage (kV)
    ZONE        = 10   # zone, loss zone (1-999)
    VMAX        = 11   # maxVm, maximum voltage magnitude (p.u.)
    VMIN        = 12   # minVm, minimum voltage magnitude (p.u.)
