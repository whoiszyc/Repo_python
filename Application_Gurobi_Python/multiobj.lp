\ Model multiobj
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  Set0: Priority=3 Weight=1 AbsTol=1 RelTol=0.01
   El[0] + El[1] + El[2] + El[3] + El[4] + El[5] + El[6] + El[7] + El[8]
   + El[9]
  Set1: Priority=2 Weight=0.25 AbsTol=2 RelTol=0.01
   El[5] + El[6] + El[7] + El[8] + El[9] + El[15] + El[16] + El[17] + El[18]
   + El[19]
  Set2: Priority=2 Weight=1.25 AbsTol=3 RelTol=0.01
   El[3] + El[4] + El[6] + El[7] + El[13] + El[14] + El[16] + El[17]
  Set3: Priority=1 Weight=1 AbsTol=4 RelTol=0.01
   El[3] + El[4] + El[5] + El[9] + El[10] + El[11] + El[15] + El[16] + El[17]
Subject To
 Budget: El[0] + El[1] + El[2] + El[3] + El[4] + El[5] + El[6] + El[7]
   + El[8] + El[9] + El[10] + El[11] + El[12] + El[13] + El[14] + El[15]
   + El[16] + El[17] + El[18] + El[19] <= 12
Bounds
Binaries
 El[0] El[1] El[2] El[3] El[4] El[5] El[6] El[7] El[8] El[9] El[10] El[11]
 El[12] El[13] El[14] El[15] El[16] El[17] El[18] El[19]
End
