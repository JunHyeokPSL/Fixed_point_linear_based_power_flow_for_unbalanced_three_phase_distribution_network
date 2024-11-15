# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:55:12 2024

@author: HOME
"""

import win32com.client
import os, sys
import pandas as pd
import numpy as np

NOW_DIR = os.getcwd()
data_path = os.path.join(NOW_DIR, 'data')
sys.path.append(os.path.join(NOW_DIR, 'data'))
sys.path.append(os.path.join(NOW_DIR, 'utils'))
import utils as ut
import power_flow_utils as pf
case_name = 'ieee34' #IEEE13, IEEE34, IEEE37, IEEE123

case_dir = os.path.join(data_path, case_name)

case_dss = {'ieee13': 'IEEE13Nodeckt.dss', 
            'ieee34': 'ieee34Mod1.dss',
            'ieee37': 'ieee37.dss',
            'ieee123': 'IEEE123Master.dss'
    }


factor_dict = {}
factor_dict['line_length'] = 0.2
Vbase = 14372.80

case_path = os.path.join(data_path, case_name, f'{case_name}_data.xlsx')
case_file_flag = True
if case_file_flag:
    ut.modified_generate_ieee34_file2(case_path)

# OpenDSS 개체 초기화
dss = win32com.client.Dispatch('OpenDSSEngine.DSS')
if not dss.Start(0):
    raise Exception("Failed to start OpenDSS engine")

# # OpenDSS 명령어로 IEEE 34 버스 시스템 불러오기
os.chdir(case_dir)

df = pd.read_excel(case_path, sheet_name = None)
dss = ut.run_opendss(dss, df, factor_dict)
dss_text = dss.Text
dssCircuit = dss.ActiveCircuit
dssSolution = dssCircuit.Solution
dssElem = dssCircuit.ActiveCktElement
dssBus = dssCircuit.ActiveBus

# dss_text.Command = f'Compile {case_dss[case_name]}'
# dss_text.Command = "Edit Vsource.Source pu=1.0"
# #dss_text.Command = "Edit Vsource.Source angle=0.0"
# #dss_text.Command = "New Energymeter.M1  Line.L1  1"
# dss_text.Command = "solve"

# 결과 출력: 모든 버스의 전압 정보 가져오기
dss_circuit = dss.ActiveCircuit

if dss_circuit.Solution.Converged:
    
    voltage_df= ut.opendss_voltage_result(dss_circuit)
    voltage_3ph_df, voltage_pu_3ph_df = ut.create_voltage_3ph_df(voltage_df)

    line_df = ut.opendss_line_result(dss_circuit)
    load_df = ut.opendss_load_result(dss_circuit)

else:
    print("Solution did not converge. Please check the system configuration.")
    