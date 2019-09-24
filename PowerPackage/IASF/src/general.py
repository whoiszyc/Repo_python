import numpy as np
from scipy.sparse import csr_matrix
from collections import OrderedDict

class PowerSystemProblem:

    def get_network_data(self, ppc, idx):
        self.ppc = ppc
        self.idx = idx

        ## calculate base
        self.base_mva = ppc['baseMVA']     # MVA
        self.base_KV = ppc['baseKV']       # KV
        self.base_KA = self.base_mva/self.base_KV    # KA
        self.base_Ohm = self.base_KV/self.base_KA    # Ohm
        self.base_Siemens = 1/self.base_Ohm     # Siemens

        ## get size of bus, line, generator and time
        self.number_bus = ppc['bus'].shape[0] # shape returns (row,column)
        self.number_gen = ppc['gen'].shape[0]
        self.number_line = ppc['branch'].shape[0]

        ## Index number (for list and array starting from 0) as iterator
        self.iter_bus = range(0, self.number_bus)
        self.iter_line = range(0, self.number_line)
        self.iter_gen = range(0, self.number_gen)

        ## create name as iterator for indexing and searching
        self.iter_bus_name = []
        for i in self.iter_bus:
            self.iter_bus_name.append('bus_{}'.format(int(self.ppc['bus'][i, 0])))

        self.iter_line_name = []
        for i in self.iter_line:
            self.iter_line_name.append('line_{}_{}'.format(int(self.ppc['branch'][i, 0]),int(self.ppc['branch'][i, 1])))

        self.iter_gen_name = []
        for i in self.iter_gen:
            self.iter_gen_name.append('gen_{}'.format(int(self.ppc['gen'][i, 0])))


        ## create bus type list and store bus index
        ## for a generator bus, ['bus'] stores bus index, ['gen'] stores generator index connected to the bus
        self.bus_type_gen = OrderedDict()
        self.bus_type_gen['bus'] = []
        self.bus_type_gen['gen'] = []
        self.bus_type_ref = []
        self.bus_type_load = []
        for i in self.iter_bus:
            # load bus
            if self.ppc['bus'][i, idx.BUS_TYPE] == 1:
                self.bus_type_load.append(i)
            # generator bus with and without load
            elif self.ppc['bus'][i, idx.BUS_TYPE] == 2 or 3:
                self.bus_type_gen['bus'].append(i)
                k = list(self.ppc['gen'][:, 0]).index(i + 1)  # get the index of generator that is connected to bus i
                self.bus_type_gen['gen'].append(k)
            # reference bus
            elif self.ppc['bus'][i, idx.BUS_TYPE] == 3:
                self.bus_type_ref.append(i)

            else:
                print('No such type of bus')


    def make_ybus(self):
        """
        make Y bus from matpower data format
        """

        ## pass parameter
        idx = self.idx
        baseMVA = self.ppc["baseMVA"]
        bus = self.ppc["bus"]
        branch = self.ppc["branch"]

        ## below from pypower with indexing change
        nb = bus.shape[0]          # number of buses
        nl = branch.shape[0]       # number of lines

        ## for each branch, compute the elements of the branch admittance matrix where
        ##      | If |   | Yff  Yft |   | Vf |
        ##      |    | = |          | * |    |
        ##      | It |   | Ytf  Ytt |   | Vt |
        stat = branch[:, idx.BR_STATUS]              ## ones at in-service branches
        Ys = stat / (branch[:, idx.BR_R] + 1j * branch[:, idx.BR_X])  ## series admittance
        Bc = stat * branch[:, idx.BR_B]              ## line charging susceptance
        tap = np.ones(nl)                           ## default tap ratio = 1
        i = np.nonzero(branch[:, idx.TAP])              ## indices of non-zero tap ratios
        tap[i] = branch[i, idx.TAP]                  ## assign non-zero tap ratios
        tap = tap * np.exp(1j * np.pi / 180 * branch[:, idx.SHIFT])  ## add phase shifters
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * np.conj(tap))
        Yft = - Ys / np.conj(tap)
        Ytf = - Ys / tap

        ## compute shunt admittance
        ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
        ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
        ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
        ## i.e. Ysh = Psh + j Qsh, so ...
        ## vector of shunt admittances
        Ysh = (bus[:, idx.GS] + 1j * bus[:, idx.BS]) / baseMVA

        ## build connection matrices
        f = branch[:, idx.F_BUS] - 1                           ## list of "from" buses
        t = branch[:, idx.T_BUS] - 1                           ## list of "to" buses

        ## connection matrix for line & from buses
        Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))

        ## connection matrix for line & to buses
        Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))

        ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
        ## at each branch's "from" bus, and Yt is the same for the "to" bus end
        i = np.r_[range(nl), range(nl)]  # double set of row indices

        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))

        ## build Ybus
        Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

        ## Add Y bus into attribute
        self.Ybus = Ybus

        ## split Y bus into real and imaginary parts
        for i in range(self.number_bus):
            for j in range(self.number_bus):
                if i != j:
                    Ybus[i, j] = Ybus[i, j]*(-1)
                else:
                    pass
        self.G = np.real(Ybus)
        self.B = np.imag(Ybus)

