# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Runs an optimal power flow.
"""

from pypower.api import case39, ppoption, runpf, printpf
ppc = case39()
ppopt = ppoption(PF_ALG=1)
r = runpf(ppc, ppopt)
# printpf(r)

