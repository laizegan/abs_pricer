# -*- coding: gbk -*-
import numpy as np
import matplotlib.pyplot as plt
from abs_pricer import ABS_Pricer

jyabs = ABS_Pricer(plan_cashflows_file='underlying_assets_df.csv',
                   oas=0.0132,
                   maturity=30,
                   timestep=1/12)
kwargs = {'r0': 0.023247,
          'kappa':1.679592,
          'theta':0.020505,
          'sigma': 0.015734,
          'n_paths': 1000}
psa_speed_array = np.linspace(0, 0.01, 100)
jyabsprice_array = np.zeros(100)
for i in range(len(psa_speed_array)):
    jyabsprice_array[i] = jyabs.calculate_absprice(**kwargs, psa_speed=psa_speed_array[i])
    print(f"When PSA speed equals {psa_speed_array[i]}, the price of the ABS product is: {jyabsprice_array[i]}")

plt.plot(psa_speed_array, jyabsprice_array)
plt.xlabel('PSA Speed')
plt.ylabel('ABS Price (RMB)')
plt.savefig('PSASPEEDandABSPRICE.png')
plt.show()