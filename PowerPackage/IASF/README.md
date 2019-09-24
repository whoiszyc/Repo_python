# Impact Assessment on Scheduling Functions (IASF)
A python package with power system generation scheduling and control functions, including unit commitment,
economic dispatch, and load frequency control. The package provides application programming interface (API) for
usage and integration with other toolkits.

* The optimization is formulated using [Pyomo](http://www.pyomo.org/), a python open source optimization modeling interface.

* The load frequency control is solved using build-in [Euler's method](https://en.wikipedia.org/wiki/Euler_method) as the numerical integration method.

* Data format is aligned with [Matpower](https://matpower.org/) and [PyPower](https://github.com/rwl/PYPOWER)

* Some code from [PyPower](https://github.com/rwl/PYPOWER) are reused.

## Repository

* Case data is stored in the directory `cases`
* Main functions are stored in the directory `src`
* Two tests with necessary data are stored in the directory `tests`

In `src`:
* `general` defines a power system problem object with general-purpose data processing
* `scheduling` define a power system scheduling problem object, which necessary functions to
define unit commitment and economic dispatch 
* `control` define a power system scheduling problem object, including System splitting and equivalent balancing 
authority modeling, numerical integration
* `utils` define a solution dictionary object to automatically retrieve optimization results and easy plotting

## Installation
Before using the package, please add the parent directory into system path by
```
import sys
sys.path.append('parent directory path')
```

## Dependency

Optimization model interface [Pyomo](http://www.pyomo.org/)
```
pip install Pyomo
```

Mixed integer programming solver: either open-source [GLPK](https://www.gnu.org/software/glpk/) 
```
pip install glpk
```
or commercial [Gurobi](http://www.gurobi.com/). Installation of [Gurobi](http://www.gurobi.com/) is detailed on their website.


## Basic usage

In these three functions, load frequency control will depend on the result of economic dispatch.
 Economic dispatch will depend on the result of unit commitment. To run one function, one can either call 
 the dependent function to obtain the data, or previous load data.
 
### Load modules and data

Load test case data
```
from IASF.cases.ieee_39bus import *
from IASF.cases.index import *
network_data = case39()
```
Load module and functions
```
from IASF.src.scheduling import SchedulingProblem
from IASF.src.utils import SolutionDict
 ```

### Unit commitment

Define an instance of a UC problem class with starting and ending time, time step in minutes and type 'UC'
```
u = SchedulingProblem('00:00', '24:00', 60, 'UC')
```
Pass network data, load forecasting data
Note that the original data load data are not aligned with the actual forecasting data in term of magnitude
```
u.get_network_data(network_data, idx_matpower)
u.get_system_reserve_data(RV)
u.get_system_load_data(PL)
```
Set bus load mode, 1 means distributing forecasting data based on the original network load ratio
```
u.set_bus_load(1)
u.convert_cost()
```
Define optimization model, power flow model and solver name
```
u.define_pyomo_model('Concrete')
u.define_problem('DC')
u.solve_problem('glpk', 1)
```
Define the solution dictionary to retrieve and plot result
```
voltage_normal = SolutionDict(u.time_series, u.iter_bus_name, 'Time (h)', 'Voltage (PU)', 'Bus Voltage', (12, 5))
voltage_normal = u.get_solution_2d('V', u.iter_bus, u.iter_time, voltage_normal)
voltage_normal.plot_step_2d()
```


### Economic dispatch

Economic dispatch can be formed in the same way except the dependency data (generator commitment result) is mandatory

```
ed = SchedulingProblem('00:00', '24:00', 60, 'ED')
```
Pass generator on/off status to the attribute
```
ed.dependency = gen_power_status
```

### Load frequency control
Read dependent data from economic dispatch in excel format
```
time_series_df = pd.read_excel('time_series.xlsx')
ed_df = pd.read_excel('generator_dispatch.xlsx')
pf_df = pd.read_excel('power_flow.xlsx')
voltage_df = pd.read_excel('voltage.xlsx')
angle_df = pd.read_excel('angle.xlsx')
```
Change them to dictionary type
```
time_series = time_series_df.to_dict()
ed = ed_df.to_dict()
pf = pf_df.to_dict()
voltage = voltage_df.to_dict()
angle = angle_df.to_dict()
```


Define load frequency control instance and pass dynamic data and desired starting time instant
```
lfc = FrequencyControlProblem(LFC_39bus, '13:45')
```
Pass time series of economic dispatch to locate the instant index 
```
lfc.time_series = time_series[0]
lfc.locate_index()
```
Pass network data to perform the equivalent balancing authority modeling. The instant index is used to get power system
 operating condition at that instant to compute the center of inertia, tie-line coefficents and area headroom
```
lfc.get_network_data(network_data, idx_matpower)
lfc.compute_area_generation(ed)
lfc.compute_tieline_syn(angle, voltage)
```
Define symbolic model and compute state-space model
```
lfc.symbolic_model_define(0, 1)
lfc.symbolic_model_define(1, 1)
lfc.symbolic_model_define(2, 1)
lfc.get_vector()
lfc.get_state_space()
```
Define saturation based the area headroom
```
lfc.saturation_default(-100000, 100000)
# set saturation from computation results of economic dispatch
lfc.saturation_set('Pm_0', -1, lfc.area_headroom[0])
lfc.saturation_set('Pm_1', -1, lfc.area_headroom[1])
lfc.saturation_set('Pm_2', -1, lfc.area_headroom[2])
```
Give initial condition
```
lfc.x0 = [0] * lfc.number_state
```
Define time series
```
lfc.t = np.linspace(0, 50, 2000)
```
Define disturbance variable (area) starting time, ending time and magnitude in term of MW
```
lfc.input_default()
lfc.input_set('Pl_0', 5, 50, 500)
```
Solve problem using [Euler's method](https://en.wikipedia.org/wiki/Euler_method)
```
lfc.ode_euler()
```

## Key discussion
* ISO-level load distribution to individual buses
* System splitting and equivalent balancing authority modeling

## References

Linear power flow model reference
```
[1] P. A. Trodden, W. A. Bukhsh, A. Grothey, and K. I. M. McKinnon, “Optimization-based Islanding of power networks using piecewise linear AC power flow,” IEEE Trans. Power Syst., vol. 29, no. 3, pp. 1212–1220, 2014.
```

Load frequency control reference
```
[1] L. Jiang, W. Yao, Q. H. Wu, J. Y. Wen, and S. J. Cheng, “Delay-dependent stability for load frequency control with constant and time-varying delays,” IEEE Trans. Power Syst., vol. 27, no. 2, pp. 932–941, 2012.
[2] D. Apostolopoulou, P. W. Sauer, and A. D. Dominguez-Garcia, “Balancing authority area model and its application to the design of adaptive AGC systems,” IEEE Trans. Power Syst., vol. 31, no. 5, pp. 3756–3764, 2016.
[3] B. Polajžer, D. Dolinar, and J. Ritonja, “Estimation of Area’s Frequency Response Characteristic During Large Frequency Changes Using Local Correlation,” IEEE Trans. Power Syst., vol. 31, no. 4, pp. 3160–3168, 2016.
```

IEEE 39-bus system geographical feature reference
```
[1] F. Wilches-Bernal, J. H. Chow, and J. J. Sanchez-Gasca, “A Fundamental Study of Applying Wind Turbines for Power System Frequency Control,” IEEE Trans. Power Syst., vol. 31, no. 2, pp. 1496–1505, 2016.
```


## Related Resources

* [PyPsa](https://pypsa.org/)
* [PandaPower](https://pandapower.readthedocs.io/en/v2.0.1/)
* [PyPower](https://github.com/rwl/PYPOWER) and [PyPower-Dynamics](https://github.com/susantoj/PYPOWER-Dynamics)
* [Andes](https://github.com/cuihantao/andes)


## Future work
* Visualization of scheduling results and dynamic response 
* Locational marginal price function