Bus.con = [ ... 
  1  24  1  0  1  1;
  2  400  1  0  1  1;
  3  400  1  0  1  1;
  4  400  1  0  1  1;
 ];

Line.con = [ ... 
  2  4  2220  400  60  0  0  0  0.5  0  0  0  0  0  0  1;
  2  3  2220  400  60  0  0  0.01  0.8  0  0  0  0  0  0  1;
  3  4  2220  400  60  0  0  0  0.13  0  0  0  0  0  0  1;
  1  2  2220  24  60  0  0.06  0  0.15  0  0  0  0  0  0  1;
 ];

SW.con = [ ... 
  4  2220  400  0.90081  0  1.5  -1.5  1.1  0.9  0.8  1  1  1;
 ];

PV.con = [ ... 
  1  2220  24  0.9  1  0.8  -0.2  1.1  0.9  1  1;
 ];

Syn.con = [ ... 
  1  2220  24  60  6  0.15  0.003  1.81  0.3  0.23  8  0.03  1.76  0.65  0.25  1  0.07  7  0  0  0  1  1  0  0.05  0.1;
 ];

Exc.con = [ ... 
  1  3  7  -6.4  200  1  1  1  0.01  1  0.015  0.0006  0.9;
 ];

Tcsc.con = [ ... 
  2  2  2  1  2220  400  60  53.75  0.01  3.1416  -3.1416  5  1  0.2  0.1  10  1;
 ];

Bus.names = {... 
  'Bus1'; 'Bus2'; 'Bus3'; 'Bus4'};
