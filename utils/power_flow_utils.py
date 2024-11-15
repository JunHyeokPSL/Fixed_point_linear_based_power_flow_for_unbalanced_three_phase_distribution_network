# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:24:17 2024

@author: user
"""

import numpy as np
import pandas as pd

def calculate_pq_load(V0, VL, pf_data_dict):
    
    load_dict = pf_data_dict['load_dict']
    s_wye = load_dict['s_wye']
    s_delta= load_dict['s_delta']
    s_wye_non_zero_indices = np.nonzero(s_wye)[0]
    
    H = pf_data_dict['H']
    
    YL0 = pf_data_dict['YL0']
    YLL = pf_data_dict['YLL']
    
    
    # Calculate updated currents for wye and delta using the updated voltage VL_new
    I = YLL @ VL + YL0 @ V0
    
    I_wye = np.zeros_like(I)
    I_wye[s_wye_non_zero_indices] = I[s_wye_non_zero_indices]
    # Calculate impedance for delta loads using |V|^2 / S^*
    V_delta = H @ VL  # Voltage across delta loads (difference between phases)
    S_delta = s_delta  # Complex power for delta loads
    Z_delta = np.abs(V_delta)**2 / np.conjugate(S_delta)  # Calculate impedance
    
    # Calculate I_delta using V / Z
    I_delta = V_delta / Z_delta  # Current for delta load buses
    
    
    # Calculate updated s_wye and s_delta using the new voltage VL_new and currents
    calc_s_wye = VL * np.conjugate(I_wye)  # Calculated complex power for wye loads
    calc_s_delta = (H @ VL) * np.conjugate(I_delta)  # Calculated complex power for delta loads
    
    # Split real and imaginary parts for comparison with initial load values
    calc_p_wye = np.round(np.real(calc_s_wye),4)
    calc_q_wye = np.round(np.imag(calc_s_wye),4)
    calc_p_delta = np.round(np.real(calc_s_delta),4)
    calc_q_delta = np.round(np.imag(calc_s_delta),4)
    calc_pq_wye = np.concatenate((calc_p_wye, calc_q_wye))
    calc_pq_delta = np.concatenate((calc_p_delta, calc_q_delta))
    
    return calc_pq_wye, calc_pq_delta


def generate_Hmatrix(delta_load, pf_data_dict):
    
    Ndelta = pf_data_dict['Ndelta']
    Nphases = pf_data_dict['Nphases']
    bus_phases = pf_data_dict['bus_phases']
    slack_nphases = pf_data_dict['slack_nphases']
    
    H = np.zeros((Ndelta, Nphases), dtype=float)

    current_delta_index = 0
    for _, row in delta_load.iterrows():
        bus = row['Bus']
        if '.' in bus:
            parts = bus.split('.')
            base_bus = parts[0]
            phase_1 = int(parts[1]) - 1
            phase_2 = int(parts[2]) - 1
            if base_bus in bus_phases:
                phase_indices = bus_phases[base_bus]
                phase_1_idx = [idx for phase, idx in phase_indices if phase == phase_1][0] - slack_nphases
                phase_2_idx = [idx for phase, idx in phase_indices if phase == phase_2][0] - slack_nphases

                # Update H matrix
                H[current_delta_index, phase_1_idx] = 1
                H[current_delta_index, phase_2_idx] = -1
                current_delta_index += 1
                
    return H
                
def setup_powerflow(voltage_df, line_df, load_df):
    
    Vbase = 14372.80
    
    v_df = voltage_df[['Bus']].iloc[0:] #Except source at substation
    v_df['realV'] = voltage_df['VM_real'] + 1j*voltage_df['VM_imag']
    # Initialize v_df['V'] based on Bus names
    v_df['V'] = v_df['Bus'].apply(lambda bus: 
        np.complex128(Vbase + 0.4607j) if bus.endswith('.1') else 
        np.complex128(Vbase * np.exp(-1j * 2 / 3 * np.pi) + 0.4607j) if bus.endswith('.2') else 
        np.complex128(Vbase * np.exp(1j * 2 / 3 * np.pi) + 0.4607j)
    )

    load_data = load_df[['Bus', 'Conn', 'kW', 'kVAR']]
    line_data = line_df[['From', 'To', 'R', 'X', 'C']]

    # Number of buses
    slack_name = v_df['Bus'].iloc[0][:-2] 

    bus_phases, current_index = generate_bus_phase_set(v_df)
    slack_nphases = len(bus_phases[slack_name])
    Nphases = current_index - slack_nphases
    
    pf_data_dict = {'bus_phases': bus_phases,
                    'Nphases': Nphases, 'slack_nphases': slack_nphases }
    

    Y00, Y0L, YL0, YLL = generate_ymatrix( line_data, v_df, slack_name)
    pf_data_dict['Y00'] = Y00
    pf_data_dict['Y0L'] = Y0L
    pf_data_dict['YL0'] = YL0
    pf_data_dict['YLL'] = YLL
    
    load_dict = generate_load_df( load_data, v_df, bus_phases, slack_name)
    pf_data_dict['load_dict'] = load_dict

    delta_load = load_dict['delta_load']
    # Construct H matrix based on delta load data
    Ndelta = delta_load.shape[0]
    pf_data_dict['Ndelta'] = Ndelta
    H = generate_Hmatrix(delta_load, pf_data_dict)
    pf_data_dict['H'] = H

    # Extract voltages from v_df
    V = v_df['V'].values  # Assuming V is stored in a column named 'V'

    # Splitting voltage vector into slack and load components
    V0 = V[:Y00.shape[0]]  # Slack bus voltages
    VL = V[Y00.shape[0]:]  # Load bus voltages
    VM = np.abs(VL)

    # Calculate w = - YLL_inv * YL0 * V0 (zero-load voltage)
    YLL_inv = np.linalg.inv(YLL)
    w = -YLL_inv @ YL0 @ V0
    
    pf_data_dict['YLL_inv'] = YLL_inv
    
    pf_state_dict = {'VL': VL, 'VM': VM, 'V0': V0, 'w': w, 'v_df': v_df.copy()}
    
    return pf_data_dict, pf_state_dict 

def update_matrix(pf_data_dict, V0, VL):
    
    H = pf_data_dict['H']
    YLL_inv = pf_data_dict['YLL_inv'] 
    
    VL_diag_inv = np.diag(1 / np.conjugate(VL))      

    
    # Calculate H^T * diag(H * VL)^-1 * s_delta
    H_VL = H @ np.conjugate(VL)
    H_VL_diag_inv = np.diag(1 / H_VL)
    
    M_Y = np.hstack((YLL_inv @ VL_diag_inv, -1j* YLL_inv @ VL_diag_inv))
    M_D = np.hstack((YLL_inv @ H.T @ H_VL_diag_inv, -1j* YLL_inv @ H.T @ H_VL_diag_inv))   
    
    
    # Calculate updated currents for wye and delta using the updated voltage VL_new
    calc_pq_wye, calc_pq_delta = calculate_pq_load(V0, VL, pf_data_dict)
    
    # Update voltage using the fixed-point equation
    K_Y = np.diag(1 / np.abs(VL)) @ np.real(np.diag(np.conjugate(VL)) @ M_Y)
    K_D = np.diag(1 / np.abs(VL)) @ np.real(np.diag(np.conjugate(VL)) @ M_D)
    
    b = np.abs(VL) - K_Y @ calc_pq_wye - K_D @ calc_pq_delta
    
    matrix_dict = {'M_Y': M_Y, 'M_D': M_D, 'K_Y': K_Y, 'K_D': K_D, 'b': b}
    
    return matrix_dict
    
    
    
def run_powerflow_vm(voltage_df, line_df, load_df, der_sol_dict = None):  
    
    pf_data_dict, pf_state_dict = setup_powerflow(voltage_df, line_df, load_df)

    H = pf_data_dict['H']
    YLL_inv = pf_data_dict['YLL_inv']
    load_dict = pf_data_dict['load_dict']
    pq_wye = load_dict['pq_wye']
    pq_delta = load_dict['pq_delta']
    
    if der_sol_dict:
        p_DER_wye = der_sol_dict['p_DER_wye']
        q_DER_wye = der_sol_dict['q_DER_wye']
        p_DER_delta = der_sol_dict['p_DER_delta']
        q_DER_delta = der_sol_dict['q_DER_delta']
        pq_DER_wye = np.concatenate((p_DER_wye, q_DER_wye))
        pq_DER_delta = np.concatenate((p_DER_delta, q_DER_delta))
        
        s_DER_wye = p_DER_wye + 1j* q_DER_wye
        s_DER_delta = p_DER_delta + 1j* q_DER_delta
        s_wye = load_dict['s_wye']
        s_delta = load_dict['s_delta']
        s_wye += s_DER_wye
        s_delta += s_DER_delta
        load_dict['s_wye'] = s_wye
        load_dict['s_delta'] = s_delta
        
    else:
        pq_DER_wye = np.zeros(len(pq_wye))
        pq_DER_delta = np.zeros(len(pq_delta))
        
    
    pq_wye += pq_DER_wye
    pq_delta += pq_DER_delta
        
    load_dict['pq_wye'] = pq_wye
    load_dict['pq_delta'] = pq_delta   
    
    
    VM, VL, V0 = pf_state_dict['VM'], pf_state_dict['VL'], pf_state_dict['V0']
    v_df, w = pf_state_dict['v_df'], pf_state_dict['w']
    
    # Iteratively update VL until convergence
    tolerance = 1e-6
    max_iterations = 100
    iteration = 0
    converged = False

    V_list = []
    converge_gap = []    
    while not converged and iteration < max_iterations:
        # Calculate diag(VL)^-1 * s_wye
        V_list.append(VM)  

        matrix_dict = update_matrix(pf_data_dict, V0, VL)
        M_Y, M_D = matrix_dict['M_Y'], matrix_dict['M_D']
        K_Y, K_D, b = matrix_dict['K_Y'], matrix_dict['K_D'], matrix_dict['b']
 
        # Calculate updated currents for wye and delta using the updated voltage VL_new
        calc_pq_wye, calc_pq_delta = calculate_pq_load(V0, VL, pf_data_dict)
        

        VL_new = w + M_Y @ pq_wye  + M_D @ pq_delta 
        
        # Calculate the magnitude of voltage |V|
        VM_new = K_Y @ pq_wye + K_D @ pq_delta + b   
  
        # Update voltage using the fixed-point equation
        converge_gap.append(np.linalg.norm(VM_new - VM))
        
        # Check for convergence
        if np.linalg.norm(VM_new - VM) < tolerance:
            converged = True
        else:
            VL = VL_new
            VM = VM_new
            iteration += 1

    # Printing results for verification
    print(f"Converged in {iteration} iterations" if converged else "Did not converge")

    v_df = v_df.copy()
    v_df['calVM'] = 0
    v_df.loc[:2, 'calVM'] = np.abs(V0)
    v_df.loc[3:, 'calVM'] = VM
    
    v_df['calV'] = 0
    v_df.loc[:2, 'calV'] = V0
    v_df.loc[3:, 'calV'] = VL
    
    pf_state_dict['VM'], pf_state_dict['VL'], pf_state_dict['V0'] =  VM, VL, V0
    
    flow_iter_result = [V_list, converge_gap]
    result_dict = {'v_df': v_df, 'H': H, 'iter_result': flow_iter_result, 
                   'w': w, 'pq_wye': pq_wye, 'pq_delta': pq_delta, 
                   'pf_data_dict': pf_data_dict, 'pf_state_dict': pf_state_dict,
                   'VM': VM_new, 'VL_new': VL_new
                   }
        
    return result_dict

def run_powerflow_pq(voltage_df, line_df, load_df):
    
    Vbase = 14372.80
    
    v_df = voltage_df[['Bus']].iloc[0:] #Except source at substation
    v_df['realV'] = voltage_df['VM_real'] + 1j*voltage_df['VM_imag']
    # Initialize v_df['V'] based on Bus names
    v_df['V'] = v_df['Bus'].apply(lambda bus: 
        np.complex128(Vbase + 0.4607j) if bus.endswith('.1') else 
        np.complex128(Vbase * np.exp(-1j * 2 / 3 * np.pi) + 0.4607j) if bus.endswith('.2') else 
        np.complex128(Vbase * np.exp(1j * 2 / 3 * np.pi) + 0.4607j)
    )

    load_data = load_df[['Bus', 'Conn', 'kW', 'kVAR']]
    line_data = line_df[['From', 'To', 'R', 'X', 'C']]

    # Number of buses
    num_buses = len(v_df)
    slack_name = v_df['Bus'].iloc[0][:-2] 

    bus_phases, current_index = generate_bus_phase_set(v_df)
    slack_nphases = len(bus_phases[slack_name])
    n_all_phase_wo_slack = current_index - slack_nphases

    Y00, Y0L, YL0, YLL = generate_ymatrix( line_data, v_df, slack_name)

    wye_load, delta_load = generate_load_df( load_data, v_df, bus_phases, slack_name)

    # Construct H matrix based on delta load data
    Ndelta = delta_load.shape[0]
    H = np.zeros((Ndelta, n_all_phase_wo_slack), dtype=float)

    current_delta_index = 0
    for _, row in delta_load.iterrows():
        bus = row['Bus']
        if '.' in bus:
            parts = bus.split('.')
            base_bus = parts[0]
            phase_1 = int(parts[1]) - 1
            phase_2 = int(parts[2]) - 1
            if base_bus in bus_phases:
                phase_indices = bus_phases[base_bus]
                phase_1_idx = [idx for phase, idx in phase_indices if phase == phase_1][0] - slack_nphases
                phase_2_idx = [idx for phase, idx in phase_indices if phase == phase_2][0] - slack_nphases

                # Update H matrix
                H[current_delta_index, phase_1_idx] = 1
                H[current_delta_index, phase_2_idx] = -1
                current_delta_index += 1
    
    # Extract voltages from v_df
    V = v_df['V'].values  # Assuming V is stored in a column named 'V'

    # Splitting voltage vector into slack and load components
    V0 = V[:Y00.shape[0]]  # Slack bus voltages
    VL = V[Y00.shape[0]:]  # Load bus voltages
    init_VL = VL
    # Calculate w = - YLL_inv * YL0 * V0 (zero-load voltage)
    YLL_inv = np.linalg.inv(YLL)
    w = -YLL_inv @ YL0 @ V0
    #w = VL

    # Extract load data (assuming they are represented as complex power S = P + jQ)
    # For wye loads
    s_wye = (wye_load['Wye_kW'] + 1j * wye_load['Wye_kVAR']).values
    # For delta loads
    s_delta = (delta_load['Delta_kW'] + 1j * delta_load['Delta_kVAR']).values 

    s_wye = -s_wye * 1e3
    s_delta = -s_delta * 1e3
    
    p_wye = - wye_load['Wye_kW'].values * 1e3
    q_wye = - wye_load['Wye_kVAR'].values * 1e3
    
    p_delta = - delta_load['Delta_kW'].values  * 1e3
    q_delta = - delta_load['Delta_kVAR'].values * 1e3
    
    pq_wye = np.concatenate((p_wye, q_wye))
    pq_delta = np.concatenate((p_delta, q_delta))
    
    # Iteratively update VL until convergence
    tolerance = 1e-6
    max_iterations = 100
    iteration = 0
    converged = False

    V_list = []
    converge_gap = [] 
    s_wye_non_zero_indices = np.nonzero(s_wye)[0]
    while not converged and iteration < max_iterations:
        # Calculate diag(VL)^-1 * s_wye
        V_list.append(VL)
        VL_diag_inv = np.diag(1 / np.conjugate(VL))      

        
        # Calculate H^T * diag(H * VL)^-1 * s_delta
        H_VL = H @ np.conjugate(VL)
        H_VL_diag_inv = np.diag(1 / H_VL)
        
        M_Y = np.hstack((YLL_inv @ VL_diag_inv, -1j* YLL_inv @ VL_diag_inv))
        M_D = np.hstack((YLL_inv @ H.T @ H_VL_diag_inv, -1j* YLL_inv @ H.T @ H_VL_diag_inv))
        
        # Calculate updated currents for wye and delta using the updated voltage VL_new
        I = YLL @ VL + YL0 @ V0
        
        I_wye = np.zeros_like(I)
        I_wye[s_wye_non_zero_indices] = I[s_wye_non_zero_indices]
        # Calculate impedance for delta loads using |V|^2 / S^*
        V_delta = H @ VL  # Voltage across delta loads (difference between phases)
        S_delta = s_delta  # Complex power for delta loads
        Z_delta = np.abs(V_delta)**2 / np.conjugate(S_delta)  # Calculate impedance
        
        # Calculate I_delta using V / Z
        I_delta = V_delta / Z_delta  # Current for delta load buses

        
        # Calculate updated s_wye and s_delta using the new voltage VL_new and currents
        calc_s_wye = VL * np.conjugate(I_wye)  # Calculated complex power for wye loads
        calc_s_delta = (H @ VL) * np.conjugate(I_delta)  # Calculated complex power for delta loads
        
        # Split real and imaginary parts for comparison with initial load values
        calc_p_wye = np.round(np.real(calc_s_wye),4)
        calc_q_wye = np.round(np.imag(calc_s_wye),4)
        calc_p_delta = np.round(np.real(calc_s_delta),4)
        calc_q_delta = np.round(np.imag(calc_s_delta),4)
        calc_pq_wye = np.concatenate((calc_p_wye, calc_q_wye))
        calc_pq_delta = np.concatenate((calc_p_delta, calc_q_delta))
        
        s_wye_current = M_Y @ pq_wye
        s_delta_current = M_D @ pq_delta        
        
        K_Y = np.diag(1 / np.abs(VL)) @ np.real(np.diag(np.conjugate(VL)) @ M_Y)
        K_D = np.diag(1 / np.abs(VL)) @ np.real(np.diag(np.conjugate(VL)) @ M_D)
        b = np.abs(VL) - K_Y @ calc_pq_wye - K_D @ calc_pq_delta
        
        # Calculate the magnitude of voltage |V|
        VM_new = K_Y @ pq_wye + K_D @ pq_delta + b 
        
        
        # Update voltage using the fixed-point equation
        VL_new = w + s_wye_current  + s_delta_current
        converge_gap.append(np.linalg.norm(VL_new - VL))
        # Check for convergence
        if np.linalg.norm(VL_new - VL) < tolerance:
            converged = True
        else:
            VL = VL_new
            iteration += 1

    # Printing results for verification
    print(f"Converged in {iteration} iterations" if converged else "Did not converge")

    v_df = v_df.copy()
    v_df['calV'] = 0
    v_df.loc[:2, 'calV'] = V0
    v_df.loc[3:, 'calV'] = VL
    
    flow_iter_result = [V_list, converge_gap]
    result_dict = {'v_df': v_df, 'H': H, 'iter_result': flow_iter_result, 
                   'w': w, 's_wye': 's_wye', 's_delta': s_delta, 
                   'Y00': Y00, 'Y0L': Y0L, 'YL0': YL0, 'YLL': YLL,
                   'VM': VM_new
                   }
    
    
    return result_dict


def run_powerflow(voltage_df, line_df, load_df):
    
    Vbase = 14372.80
    
    v_df = voltage_df[['Bus']].iloc[0:] #Except source at substation
    v_df['realV'] = voltage_df['VM_real'] + 1j*voltage_df['VM_imag']
    # Initialize v_df['V'] based on Bus names
    v_df['V'] = v_df['Bus'].apply(lambda bus: 
        np.complex128(Vbase + 0.4607j) if bus.endswith('.1') else 
        np.complex128(Vbase * np.exp(-1j * 2 / 3 * np.pi) + 0.4607j) if bus.endswith('.2') else 
        np.complex128(Vbase * np.exp(1j * 2 / 3 * np.pi) + 0.4607j)
    )

    load_data = load_df[['Bus', 'Conn', 'kW', 'kVAR']]
    line_data = line_df[['From', 'To', 'R', 'X', 'C']]

    # Number of buses
    num_buses = len(v_df)
    slack_name = v_df['Bus'].iloc[0][:-2] 

    bus_phases, current_index = generate_bus_phase_set(v_df)
    slack_nphases = len(bus_phases[slack_name])
    n_all_phase_wo_slack = current_index - slack_nphases

    Y00, Y0L, YL0, YLL = generate_ymatrix( line_data, v_df, slack_name)

    wye_load, delta_load = generate_load_df( load_data, v_df, bus_phases, slack_name)

    # Construct H matrix based on delta load data
    Ndelta = delta_load.shape[0]
    H = np.zeros((Ndelta, n_all_phase_wo_slack), dtype=float)

    current_delta_index = 0
    for _, row in delta_load.iterrows():
        bus = row['Bus']
        if '.' in bus:
            parts = bus.split('.')
            base_bus = parts[0]
            phase_1 = int(parts[1]) - 1
            phase_2 = int(parts[2]) - 1
            if base_bus in bus_phases:
                phase_indices = bus_phases[base_bus]
                phase_1_idx = [idx for phase, idx in phase_indices if phase == phase_1][0] - slack_nphases
                phase_2_idx = [idx for phase, idx in phase_indices if phase == phase_2][0] - slack_nphases

                # Update H matrix
                H[current_delta_index, phase_1_idx] = 1
                H[current_delta_index, phase_2_idx] = -1
                current_delta_index += 1
    
    # Extract voltages from v_df
    V = v_df['V'].values  # Assuming V is stored in a column named 'V'

    # Splitting voltage vector into slack and load components
    V0 = V[:Y00.shape[0]]  # Slack bus voltages
    VL = V[Y00.shape[0]:]  # Load bus voltages

    # Calculate w = - YLL_inv * YL0 * V0 (zero-load voltage)
    YLL_inv = np.linalg.inv(YLL)
    w = -YLL_inv @ YL0 @ V0
    #w = VL

    # Extract load data (assuming they are represented as complex power S = P + jQ)
    # For wye loads
    s_wye = (wye_load['Wye_kW'] + 1j * wye_load['Wye_kVAR']).values
    # For delta loads
    s_delta = (delta_load['Delta_kW'] + 1j * delta_load['Delta_kVAR']).values 

    s_wye = -s_wye * 1e3
    s_delta = -s_delta * 1e3

    # Iteratively update VL until convergence
    tolerance = 1e-6
    max_iterations = 100
    iteration = 0
    converged = False

    V_list = []
    converge_gap = [] 
    while not converged and iteration < max_iterations:
        # Calculate diag(VL)^-1 * s_wye
        V_list.append(VL)
        VL_diag_inv = np.diag(1 / np.conjugate(VL))
        s_wye_current = VL_diag_inv @ np.conjugate(s_wye)

        # Calculate H^T * diag(H * VL)^-1 * s_delta
        H_VL = H @ np.conjugate(VL)
        H_VL_diag_inv = np.diag(1 / H_VL)
        s_delta_current = H.T @ (H_VL_diag_inv @ np.conjugate(s_delta))

        # Update voltage using the fixed-point equation
        VL_new = w + YLL_inv @ (s_wye_current + s_delta_current)
        converge_gap.append(np.linalg.norm(VL_new - VL))
        # Check for convergence
        if np.linalg.norm(VL_new - VL) < tolerance:
            converged = True
        else:
            VL = VL_new
            iteration += 1

    # Printing results for verification
    print(f"Converged in {iteration} iterations" if converged else "Did not converge")

    v_df = v_df.copy()
    v_df['calV'] = 0
    v_df.loc[:2, 'calV'] = V0
    v_df.loc[3:, 'calV'] = VL
    
    flow_iter_result = [V_list, converge_gap]
    result_dict = {'v_df': v_df, 'H': H, 'iter_result': flow_iter_result, 
                   'w': w, 's_wye': 's_wye', 's_delta': s_delta, 
                   'Y00': Y00, 'Y0L': Y0L, 'YL0': YL0, 'YLL': YLL,
                   }
    
    
    return result_dict
    

def generate_bus_phase_set(v_df):
    
    # Determine the phases for each bus and assign indices accordingly
    bus_phases = {}
    current_index = 0
    for _, row in v_df.iterrows():
        bus_full = row['Bus']
        bus, phase = bus_full.split('.')
        phase = int(phase) - 1  # Convert to zero-based index
        if bus not in bus_phases:
            bus_phases[bus] = []
        bus_phases[bus].append((phase, current_index))
        current_index += 1
        
    return bus_phases, current_index
    
def generate_ymatrix(line_data, v_df, slack_name, frequency=60):
    # 주파수에 따라 커패시턴스 계산
    omega = 2 * np.pi * frequency
    
    bus_phases, current_index = generate_bus_phase_set(v_df)

    # Initialize admittance matrix Y
    slack_nphases = len(bus_phases[slack_name])
    all_phase_wo_slack = current_index - slack_nphases
    Y00 = np.zeros((slack_nphases, slack_nphases), dtype=complex)  # Slack bus to slack bus
    Y0L = np.zeros((slack_nphases, all_phase_wo_slack), dtype=complex)  # Slack bus to load buses
    YL0 = np.zeros((all_phase_wo_slack, slack_nphases), dtype=complex)  # Load buses to slack bus
    YLL = np.zeros((all_phase_wo_slack, all_phase_wo_slack), dtype=complex)  # Load buses to load buses

    for _, row in line_data.iterrows():
        from_bus = row['From']
        to_bus = row['To']
        R = np.array(row['R']) # 3x3 array for resistance
        X = np.array(row['X']) # 3x3 array for reactance
        C = np.array(row['C'])  # 3x3 array for capacitance
        C_matrix = 1j * omega * C /2 /1e9 # 커패시턴스의 효과를 허수부에 추가
        Z = R + 1j * X # Impedance matrix (3x3)
        

        # Determine the phases that are present for both from_bus and to_bus
        if from_bus in bus_phases and to_bus in bus_phases:
            from_phases = [phase for phase, _ in bus_phases[from_bus] if R[phase, phase] != 0]
            to_phases = [phase for phase, _ in bus_phases[to_bus] if R[phase, phase] != 0]
            
            if len(from_phases) == 1 or len(to_phases) == 1:
                if len(from_phases) == 1:
                    single_phase = from_phases[0]
                    multi_phases = to_phases
                    from_idx = [idx - slack_nphases for phase, idx in bus_phases[from_bus] if phase == single_phase][0]
                    to_idx = [idx - slack_nphases for phase, idx in bus_phases[to_bus] if phase in multi_phases][0] 
                else:
                    single_phase = to_phases[0]
                    multi_phases = from_phases
                    to_idx = [idx - slack_nphases for phase, idx in bus_phases[to_bus] if phase == single_phase][0]
                    from_idx = [idx - slack_nphases for phase, idx in bus_phases[from_bus] if phase in multi_phases][0] 
                    
                Y_value = 1 / (Z[single_phase, single_phase])
                
                if from_bus == slack_name:
                    Y00[single_phase, single_phase] += Y_value + C_matrix[single_phase, single_phase]
                    Y0L[single_phase, to_idx] -= Y_value
                    YL0[to_idx, single_phase] -= Y_value
                    YLL[to_idx, to_idx] += Y_value + C_matrix[single_phase, single_phase]
                elif to_bus == slack_name:
                    Y00[single_phase, single_phase] += Y_value + C_matrix[single_phase, single_phase]
                    Y0L[single_phase, from_idx] -= Y_value
                    YL0[from_idx, single_phase] -= Y_value
                    YLL[from_idx, from_idx] += Y_value + C_matrix[single_phase, single_phase]
                else:
                    YLL[from_idx, from_idx] += Y_value + C_matrix[single_phase, single_phase]
                    YLL[to_idx, to_idx] += Y_value + C_matrix[single_phase, single_phase]
                    YLL[from_idx, to_idx] -= Y_value
                    YLL[to_idx, from_idx] -= Y_value
                    
            elif len(from_phases) == 3 and len(to_phases) == 3:
                # 커패시턴스를 포함한 임피던스 행렬로부터 어드미턴스 계산
                Y_reduced = np.linalg.inv(Z)
                #Y_reduced = 1/Z

                from_base_indices = [idx - slack_nphases for _, idx in bus_phases[from_bus]] 
                to_base_indices = [idx - slack_nphases for _, idx in bus_phases[to_bus]] 
                  
                for i in range(3):
                    for j in range(3):
                        if from_bus == slack_name:
                            Y00[i, j] += Y_reduced[i, j] + C_matrix[i, j]
                            Y0L[i, to_base_indices[j]] -= Y_reduced[i, j]
                            YL0[to_base_indices[j], i] -= Y_reduced[i, j]
                            YLL[to_base_indices[i], to_base_indices[j]] += Y_reduced[i, j] + C_matrix[i, j]
                        elif to_bus == slack_name:
                            Y00[i, j] += Y_reduced[i, j] + C_matrix[i, j]
                            Y0L[j, from_base_indices[i]] -= Y_reduced[i, j]
                            YL0[from_base_indices[i], j] -= Y_reduced[i, j]
                            YLL[from_base_indices[i], from_base_indices[j]] += Y_reduced[i, j] + C_matrix[i, j]
                        else:
                            YLL[from_base_indices[i], from_base_indices[j]] += Y_reduced[i, j] + C_matrix[i, j]
                            YLL[to_base_indices[i], to_base_indices[j]] += Y_reduced[i, j]  + C_matrix[i, j]
                            YLL[from_base_indices[i], to_base_indices[j]] -= Y_reduced[i, j]
                            YLL[to_base_indices[i], from_base_indices[j]] -= Y_reduced[i, j]
                            
    return Y00, Y0L, YL0, YLL

def generate_load_df( load_data, v_df, bus_phases, slack_name):
    
    slack_nphases = len(bus_phases[slack_name])
    
    wye_load_df = v_df.iloc[slack_nphases:].copy()
    # Add load data to v_df based on matching conditions
    wye_load_df['Wye_kW'] = 0.0
    wye_load_df['Wye_kVAR'] = 0.0

    for _, row in load_data.iterrows():
        bus = row['Bus']
        conn = row['Conn']
        kW = row['kW']
        kVAR = row['kVAR']

        if conn == 'Wye':
            if bus in wye_load_df['Bus'].values:
                # Single-phase bus (e.g., mid820.1)
                wye_load_df.loc[wye_load_df['Bus'] == bus, ['Wye_kW', 'Wye_kVAR']] = kW, kVAR
            elif bus in bus_phases and len(bus_phases[bus]) == 3:
                # Three-phase bus (e.g., 860, 840)
                for phase, _ in bus_phases[bus]:
                    bus_phase = f"{bus}.{phase + 1}"
                    if bus_phase in wye_load_df['Bus'].values:
                        wye_load_df.loc[wye_load_df['Bus'] == bus_phase, ['Wye_kW', 'Wye_kVAR']] = kW / 3, kVAR / 3

    # Determine delta-connected loads from load data
    delta_load_data = []
    for _, row in load_data.iterrows():
        bus = row['Bus']
        conn = row['Conn']
        kW = row['kW']
        kVAR = row['kVAR']

        if conn == 'Delta':
            if '.' in bus:
                # Delta load with explicit phase (e.g., 830.1.2)
                delta_load_data.append({'Bus': bus, 'Delta_kW': kW, 'Delta_kVAR': kVAR})
            else:
                # Delta load without explicit phase (e.g., 848, 890)
                # delta_load_data.append({'Bus': f"{bus}.1.2", 'Delta_kW': kW , 'Delta_kVAR': kVAR })
                # delta_load_data.append({'Bus': f"{bus}.2.3", 'Delta_kW': kW , 'Delta_kVAR': kVAR })
                # delta_load_data.append({'Bus': f"{bus}.3.1", 'Delta_kW': kW , 'Delta_kVAR': kVAR })
                delta_load_data.append({'Bus': f"{bus}.1.2", 'Delta_kW': kW / 3, 'Delta_kVAR': kVAR / 3})
                delta_load_data.append({'Bus': f"{bus}.2.3", 'Delta_kW': kW / 3, 'Delta_kVAR': kVAR / 3})
                delta_load_data.append({'Bus': f"{bus}.3.1", 'Delta_kW': kW / 3, 'Delta_kVAR': kVAR / 3})

    # Convert delta load data to a DataFrame
    delta_load_df = pd.DataFrame(delta_load_data)
    
    s_wye = (wye_load_df['Wye_kW'] + 1j * wye_load_df['Wye_kVAR']).values
    # For delta loads
    s_delta = (delta_load_df['Delta_kW'] + 1j * delta_load_df['Delta_kVAR']).values 
    
    s_wye = -s_wye * 1e3
    s_delta = -s_delta * 1e3
    
    p_wye = - wye_load_df['Wye_kW'].values * 1e3
    q_wye = - wye_load_df['Wye_kVAR'].values * 1e3

    p_delta = - delta_load_df['Delta_kW'].values  * 1e3
    q_delta = - delta_load_df['Delta_kVAR'].values * 1e3
    
    pq_wye = np.concatenate((p_wye, q_wye))
    pq_delta = np.concatenate((p_delta, q_delta))
    
    load_dict = {'wye_load': wye_load_df, 'delta_load': delta_load_df,
                 'pq_wye': pq_wye, 'pq_delta': pq_delta,
                 's_wye': s_wye, 's_delta': s_delta}
    return load_dict

def calculate_line_df(line_data, bus_phases, v_dict):
    
    # 복소전력 계산 결과 저장
    complex_power_list = []
    current_list = []
    for _, row in line_data.iterrows():
        from_bus = row['From']
        to_bus = row['To']
        
        # 'from_bus'와 'to_bus'에 대해 사용 가능한 상을 bus_phases로부터 가져오기
        from_bus_phases = bus_phases.get(from_bus, [])
        to_bus_phases = bus_phases.get(to_bus, [])
        
        # 공통 상 구하기 (from_bus와 to_bus의 공통된 상에 대해서만 계산)
        common_phases = set([x[0] for x in from_bus_phases]) & set([x[0] for x in to_bus_phases])
        
        # 공통 상이 없으면 계산을 건너뜁니다.
        if len(common_phases) == 0:
            print(f'Line from {from_bus} to {to_bus} has no common phases, skipping...')
            continue

        # 전압 벡터 정의 (공통 상만 고려)
        V_from = []
        V_to = []
        
        for phase in common_phases:
            # from_bus와 to_bus에 대해 해당 상의 위치 찾기
            from_pos = next(pos for idx, pos in from_bus_phases if idx == phase)
            to_pos = next(pos for idx, pos in to_bus_phases if idx == phase)

            # 전압 벡터에 해당 상의 값을 추가
            V_from.append(v_dict.get(f'{from_bus}.{phase + 1}'))
            V_to.append(v_dict.get(f'{to_bus}.{phase + 1}'))
        
        V_from = np.array(V_from)
        V_to = np.array(V_to)
        
        # 전압 차이 계산
        delta_V = V_from - V_to
        
        # 선로 임피던스 Z = R + jX
        R = row['R']
        X = row['X']
        Z = R + 1j * X

        # 임피던스 행렬을 공통 상 크기에 맞게 조정
        indices = [idx for idx, _ in from_bus_phases if idx in common_phases]
        Z_reduced = Z[np.ix_(indices, indices)]
        I = np.linalg.inv(Z_reduced).dot(delta_V)
        
        # 복소전력 S = V_from * conj(I)
        S = V_from * np.conj(I)
        
        # 결과 저장
        complex_power_list.append(S)
        current_list.append(I)

    # 새로운 DataFrame 생성
    line_cal_df = pd.DataFrame({
        'Line': line_data.index,
        'From': line_data['From'],
        'To': line_data['To']
    })

    # 각 선로에 대한 전류와 전력 추가 (전류는 크기와 위상으로 나눔)
    for i, (S, I) in enumerate(zip(complex_power_list, current_list)):
        # 'from_bus'와 'to_bus'에 대해 사용 가능한 상을 다시 bus_phases로부터 가져오기
        from_bus_phases = bus_phases.get(line_data.at[i, 'From'], [])
        to_bus_phases = bus_phases.get(line_data.at[i, 'To'], [])
        common_phases = set([x[0] for x in from_bus_phases]) & set([x[0] for x in to_bus_phases])
        
        for phase in common_phases:
            idx = next(idx for idx, pos in from_bus_phases if idx == phase)
            line_cal_df.at[i, f'Icm_{phase + 1}'] = np.abs(I[list(common_phases).index(phase)])  # 전류 크기
            line_cal_df.at[i, f'Ica_{phase + 1}'] = np.angle(I[list(common_phases).index(phase)], deg=True)  # 전류 위상 (deg)
            line_cal_df.at[i, f'Pc_{phase + 1}'] = S[list(common_phases).index(phase)].real  # 유효전력
            line_cal_df.at[i, f'Qc_{phase + 1}'] = S[list(common_phases).index(phase)].imag  # 무효전력

                
    return line_cal_df


def verify_powerflow_result(line_df, voltage_df, bus_prefix, to_prefix):
    # From이 'sourcebus'이고 To가 '800'인 경우 필터링
    line_condition = (line_df['From'] == bus_prefix) & (line_df['To'] == to_prefix)
    filtered_line_df = line_df[line_condition]

    # sourcebus와 800의 각 상에 대한 전압 복소수 계산
    voltage_from = voltage_df[voltage_df['Bus'].str.startswith(bus_prefix)]
    voltage_to = voltage_df[voltage_df['Bus'].str.startswith(to_prefix)]

    V_from = voltage_from['VM_real'].values + 1j * voltage_from['VM_imag'].values
    V_to = voltage_to['VM_real'].values + 1j * voltage_to['VM_imag'].values

    # 각 상의 전류 크기 및 위상 계산
    I_magnitudes = filtered_line_df[['Im_1', 'Im_2', 'Im_3']].values.flatten()
    I_angles = filtered_line_df[['Ia_1', 'Ia_2', 'Ia_3']].values.flatten()
    I = I_magnitudes * np.exp(1j * np.radians(I_angles))

    # 3x3 임피던스 행렬 생성 (R, X, C 값 사용)
    R = filtered_line_df['R'].iloc[0]
    X = filtered_line_df['X'].iloc[0]
    C = filtered_line_df['C'].iloc[0]

    Z_matrix = np.zeros((3, 3), dtype=complex)

    for i in range(3):
        for j in range(3):
            Z_matrix[i, j] = R[i,j] + 1j * X[i,j]

    # 구한 임피던스와 전류를 이용해 To 버스의 전압 계산 (V_to = V_from - Z * I)
    V_to_calculated = V_from - np.dot(Z_matrix, I)

    # 선로에 흐르는 전력 계산 (S = V * I* , P = Re(S), Q = Im(S))
    S = V_from * np.conj(I)
    P = S.real
    Q = S.imag

    # 계산된 V_to와 원래 V_to의 차이 확인
    voltage_diff = V_to_calculated - (voltage_to['VM_real'].values + 1j * voltage_to['VM_imag'].values)

    # V_from, V_to, Z를 이용해 전류 계산 (I = (V_from - V_to) / Z)
    cal_I = np.zeros(3, dtype=complex)
    for i in range(3):
        if Z_matrix[i, i] != 0:
            cal_I[i] = (V_from[i] - V_to[i]) / Z_matrix[i, i]


    # 결과 출력
    print("3x3 Impedance Matrix (Z):")
    print(Z_matrix)

    for phase in range(1, 4):
        print(f"Phase {phase}:")
        print(f"  Impedance (Z): {Z_matrix[phase-1, phase-1]:.7f} ohms")
        print(f"  Current (I): {I[phase-1]:.4f} A")
        print(f"  Active Power (P): {P[phase-1]:.4f} W")
        print(f"  Reactive Power (Q): {Q[phase-1]:.4f} VAR")
        print(f"  Calculated Voltage at To Bus (V_to): {V_to_calculated[phase-1]:.4f} V")
        print(f"  Original Voltage at To Bus (V_to): {voltage_to['VM_real'].values[phase-1] + 1j * voltage_to['VM_imag'].values[phase-1]:.4f} V")
        print(f"  Voltage Difference: {voltage_diff[phase-1]:.4f} V")
