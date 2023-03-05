# -*- coding: gbk -*-
import numpy as np
import pandas as pd
from scipy.stats import norm

class ABS_Pricer(object):
    def __init__(self, plan_cashflows_file, oas, maturity, timestep):
        # Read planned cashflow of the ABS product from a csv file
        plan_cashflow_df = pd.read_csv(plan_cashflows_file,
                                       encoding='gbk',
                                       converters={'期初本金余额': float,
                                                   '本期应收本金': float,
                                                   '本期应收利息': float,
                                                   '期末本金余额': float
                                                   })
        self.plan_start_principal = np.array(plan_cashflow_df.loc[:, '期初本金余额'])
        self.plan_receivable_principal = np.array(plan_cashflow_df.loc[:, '本期应收本金'])
        self.plan_receivable_interest = np.array(plan_cashflow_df.loc[:, '本期应收利息'])
        self.plan_end_principal = np.array(plan_cashflow_df.loc[:, '期末本金余额'])
        self.plan_cashflow = self.plan_receivable_principal + self.plan_receivable_interest # Planned cash flow in a period
        self.oas = oas
        self.maturity = maturity
        self.timestep = timestep
        self.n_steps = int(maturity / timestep + 1)

    def simulate_cir_paths(self, r0, kappa, theta, sigma, n_paths):
        interest_rate_paths = np.zeros((n_paths, self.n_steps))   # Simulated paths of interest rates
        discount_factor_paths = np.zeros((n_paths, self.n_steps)) # Discount rate paths
        for t in range(self.n_steps):
            if t==0:
                # Set the original interest rate
                interest_rate_paths[:, t] = r0
                discount_factor_paths[:, t] = 1
            else:
                # Generate CIR process
                interest_rate_paths[:, t] = interest_rate_paths[:, t-1] \
                    + kappa * (theta - np.maximum(interest_rate_paths[:, t-1], 0)) * self.timestep \
                    + sigma * np.sqrt(np.maximum(interest_rate_paths[:, t-1], 0)) \
                    * np.sqrt(self.timestep) * np.random.standard_normal(n_paths)
                discount_factor_paths[:, t] = 1 / ((1 + discount_factor_paths[:, t-1]) \
                    * (1 + interest_rate_paths[:, t] * self.timestep + self.oas * self.timestep))

        return interest_rate_paths, discount_factor_paths

    def generate_prepayment_series(self, psa_speed=0.006):
        cpr = np.zeros(self.n_steps)                    # Yearly conditional prepayment rates
        prepayment_rates = np.zeros(self.n_steps)       # Monthly conditional prepayment rates
        for t in range(self.n_steps):
            cpr[t] = np.minimum(psa_speed * t / 30, psa_speed)
            prepayment_rates[t] = 1 - (1 - cpr[t]) ** (self.timestep)
        
        return prepayment_rates

    def generate_default_series(self):
        default_rates = np.zeros(self.n_steps) # Preditced default rate series
        for t in range(self.n_steps):
            if t < 30:
                default_rates[t] = 0.006 * t / 30
            elif t < 60:
                default_rates[t] = 0.006
            elif t < 120:
                default_rates[t] = (-0.00285) * t / 30 + 0.0117
            else:
                default_rates[t] = 0.0003

        return default_rates

    def get_cashflow(self, psa_speed):
        cashflow = self.plan_cashflow               # Cash flow during a period
        start_principal = self.plan_start_principal # Principals at the beginning of a period
        prepayments = np.zeros(self.n_steps)        # Prepayments at the end of a period
        defaults = np.zeros(self.n_steps)           # Defaults at the end of a period
        prepayment_rates = self.generate_prepayment_series(psa_speed)
        default_rates = self.generate_default_series()

        for t in range(self.n_steps-1):
            prepayments[t] = start_principal[t] * prepayment_rates[t]
            defaults[t] = cashflow[t] * default_rates[t]
            # Consider cash flow-in due to prepayments and cash-out due to defaults in the current period 
            cashflow[t] = cashflow[t] + prepayments[t] - defaults[t]
            # Consider cash flow-out due to prepayments in the future
            cashflow[t+1:] = cashflow[t+1:] * (1 - prepayment_rates[t])

        return cashflow

    def calculate_absprice(self, r0, kappa, theta, sigma, n_paths, psa_speed=0.006):
        interest_rate_paths, discount_factor_paths = self.simulate_cir_paths(r0, kappa, theta, sigma, n_paths)
        cashflow = self.get_cashflow(psa_speed)
        absprice = np.mean(np.sum(cashflow * discount_factor_paths, axis=1))

        return absprice



