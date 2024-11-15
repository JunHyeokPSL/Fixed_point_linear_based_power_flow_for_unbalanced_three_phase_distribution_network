# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:52:06 2024

@author: HOME
"""
import pandas as pd
from collections import defaultdict
import numpy as np

                       

def opendss_load_result(dss_circuit):
    element_names = dss_circuit.AllElementNames
    load_data = []

    for element_name in element_names:
        if "load" in element_name.lower():  # 부하 요소만 필터링
            dss_circuit.SetActiveElement(element_name)
            load_elem = dss_circuit.ActiveCktElement

            # 부하가 연결된 첫 번째 버스 이름 가져오기
            load_bus = load_elem.BusNames[0]

            # 부하 파라미터 가져오기
            powers = load_elem.Powers  # 모든 상의 유효 전력과 무효 전력 (순서대로: P1, Q1, P2, Q2, ...)
            total_kw = sum(powers[::2])  # 짝수 인덱스는 kW 값
            total_kvar = sum(powers[1::2])  # 홀수 인덱스는 kVAR 값

            # 결선 정보 가져오기 (delta인지 wye인지)
            connection_type = load_elem.Properties("conn").Val
            connection_type = "Delta" if connection_type.lower() == "delta" else "Wye"

            load_data.append([element_name, load_bus, connection_type , total_kw, total_kvar])

    load_df = pd.DataFrame(load_data, columns=["Load Name", "Bus", "Conn" , "kW", "kVAR"])

    return load_df

def opendss_voltage_result(dss_circuit):
    all_bus_voltages = dss_circuit.AllBusVmagPu  # 모든 버스의 전압 크기 (퍼유닛)
    all_bus_names = dss_circuit.ALLNodeNames      # 모든 버스의 이름
    
    # 각 bus 이름이 몇 번째로 등장했는지 추적하기 위한 카운터
    bus_count = defaultdict(int)
    
    # 버스 이름과 상 정보를 분리하고 각 상에 대한 전압 크기 및 위상각 정보 저장
    bus_voltage_dict = {}
    
    for i, bus_name in enumerate(all_bus_names):
        bus, phase = bus_name.split('.')  # 버스 이름과 상 번호(1, 2, 3) 분리
        bus_count[bus] += 1  # 동일한 bus 이름의 등장 횟수 추적
        
        # 각 bus가 처음 등장한 것인지, 두 번째 등장인지 확인
        bus_key = bus_name # bus 이름과 등장 횟수를 조합하여 고유 키 생성
        
        if bus_key not in bus_voltage_dict:
            bus_voltage_dict[bus_key] = {}  # 각 버스마다 상 정보를 저장할 딕셔너리
        
        # 각 버스의 전압 위상각을 개별적으로 구하기
        dss_circuit.SetActiveBus(bus)
        voltage_angles = dss_circuit.ActiveBus.VMagAngle  # [Vmag1, angle1, Vmag2, angle2, ...]

        # 현재 상에 해당하는 전압 크기 및 위상각 저장
        try:
            index = 2 * (int(bus_count[bus]) - 1)  # 각 상에 맞는 인덱스 계산
            bus_voltage_dict[bus_key][phase] = {
                'magnitude_pu': all_bus_voltages[i],
                'magnitude': voltage_angles[index] ,         # 전압 크기 (퍼유닛)
                'angle': voltage_angles[index + 1]        # 해당 상의 위상각
            }
        except IndexError:
            # 상 정보가 부족한 경우 None으로 처리
            bus_voltage_dict[bus_key][phase] = {
                'magnitude_pu': all_bus_voltages[i],  # 전압 크기는 있지만 위상각이 없을 경우 처리
                'magnitude' : None,
                'angle': None
            }

    # 데이터프레임으로 변환
    bus_data = []
    for bus_key, phases in bus_voltage_dict.items():
        row = {'Bus': bus_key}
        for phase in ['1', '2', '3']:  # 각 상에 대해 반복
            if phase in phases:
                mag = phases[phase]['magnitude']
                ang = phases[phase]['angle']
                row['VM'] = phases[phase]['magnitude_pu']
                row['VM_real'] = mag * np.cos(np.radians(ang))
                row['VM_imag'] = mag * np.sin(np.radians(ang)) 
                row[f'VM_{phase}'] = mag # 해당 상의 전압 크기
                row[f'VA_{phase}'] = ang       # 해당 상의 전압 위상각
                 
            else:
                row[f'VM_{phase}'] = None  # 해당 상이 없으면 None
                row[f'VA_{phase}'] = None  # 해당 상이 없으면 None
        bus_data.append(row)
    
    voltage_df = pd.DataFrame(bus_data)
    
    return voltage_df

def create_voltage_3ph_df(voltage_df):
    # 빈 딕셔너리 생성 (Bus 이름을 키로 사용하여 3상 정보를 저장)
    voltage_3ph_data = {}
    voltage_pu_3ph_data = {}

    # voltage_df에서 각 row에 대해 처리
    for _, row in voltage_df.iterrows():
        bus_full_name = row['Bus']  # 예: '800.1'
        bus, phase = bus_full_name.split('.')  # Bus 이름과 상 번호 분리 (예: '800', '1')
        
        # Bus 이름이 처음 등장하면 초기화
        if bus not in voltage_3ph_data:
            voltage_3ph_data[bus] = {
                'Bus': bus,  # Bus 이름
                'VM_1': None, 'VA_1': None,
                'VM_2': None, 'VA_2': None,
                'VM_3': None, 'VA_3': None
            }
            
            voltage_pu_3ph_data[bus] = {
                'Bus': bus,  # Bus 이름
                'VM_1': None, 'VA_1': None,
                'VM_2': None, 'VA_2': None,
                'VM_3': None, 'VA_3': None
            }
        
        # 상 번호에 따라 데이터를 채우기
        phase_num = int(phase)  # 상 번호를 정수로 변환
        voltage_3ph_data[bus][f'VM_{phase_num}'] = row[f'VM_{phase_num}']
        voltage_3ph_data[bus][f'VA_{phase_num}'] = row[f'VA_{phase_num}']
        
        voltage_pu_3ph_data[bus][f'VM_{phase_num}'] = row['VM']
        voltage_pu_3ph_data[bus][f'VA_{phase_num}'] = row[f'VA_{phase_num}']
    
    # 딕셔너리를 데이터프레임으로 변환
    voltage_3ph_df = pd.DataFrame(voltage_3ph_data.values())
    voltage_pu_3ph_df = pd.DataFrame(voltage_pu_3ph_data.values())
    return voltage_3ph_df, voltage_pu_3ph_df


def opendss_line_result(dss_circuit):
    
    # 선로 전류 및 유효/무효전력 정보
    line_currents = {}
    line_powers = {}
    
    # 선로 정보 가져오기
    for line_name in dss_circuit.Lines.AllNames:
        dss_circuit.Lines.Name = line_name
        dssElem = dss_circuit.ActiveCktElement
        
        # 선로가 가지고 있는 상 수 확인
        num_phases = dssElem.NumPhases
        
        # 선로의 각 상에 대한 전류 정보 (크기와 위상각 포함)
        currents = dssElem.CurrentsMagAng  # [Magnitude, Angle, Magnitude, Angle, ...]
        current_per_phase = currents[:num_phases * 2]  # 상별 전류 데이터 (각 상당 2개의 값: 크기와 각도)

        # 선로의 유효/무효전력 (P, Q)
        powers = dssElem.Powers  # [P1, Q1, P2, Q2, ...]
        real_power_per_phase = powers[::2][:num_phases]  # 유효전력 (P)
        reactive_power_per_phase = powers[1::2][:num_phases]  # 무효전력 (Q)
        
        size = 3  # 행렬의 크기 (3x3)
        rmatrix = np.zeros((size, size))
        xmatrix = np.zeros((size, size))
        cmatrix = np.zeros((size, size))
        
        line_code = int(dssElem.Properties('linecode').Val)
        rmatrix_str = dssElem.Properties('rmatrix').Val
        rmatrix_list = [float(x) for x in rmatrix_str.replace('[', '').replace(']', '').replace('|', ' ').split()]
        
        xmatrix_str = dssElem.Properties('xmatrix').Val
        xmatrix_list = [float(x) for x in xmatrix_str.replace('[', '').replace(']', '').replace('|', ' ').split()]
        
        cmatrix_str = dssElem.Properties('cmatrix').Val
        cmatrix_list = [float(x) for x in cmatrix_str.replace('[', '').replace(']', '').replace('|', ' ').split()]
        
        line_length = float(dssElem.Properties('length').Val)
                
        if line_code in [300,301]:
            index = 0
            for i in range(size):
                for j in range(i + 1):
                    rmatrix[i, j] = rmatrix_list[index] * line_length 
                    xmatrix[i, j] = xmatrix_list[index] * line_length 
                    cmatrix[i, j] = cmatrix_list[index] * line_length 
                    if i != j:  # 대각이 아닌 경우 대칭으로 상단도 채우기
                        rmatrix[j, i] = rmatrix_list[index] * line_length 
                        xmatrix[j, i] = xmatrix_list[index] * line_length 
                        cmatrix[j, i] = cmatrix_list[index] * line_length 
                    index += 1
                    
        elif line_code in [302]:
            rmatrix[0, 0] = rmatrix_list[0] * line_length 
            xmatrix[0, 0] = xmatrix_list[0] * line_length 
            cmatrix[0, 0] = cmatrix_list[0] * line_length 
            
        elif line_code in [303, 304]:
            rmatrix[1, 1] = rmatrix_list[0] * line_length 
            xmatrix[1, 1] = xmatrix_list[0] * line_length 
            cmatrix[1, 1] = cmatrix_list[0] * line_length 
            
        else:
            ValueError
            
        
        # 각 상의 정보를 상 번호에 맞게 저장하기 위해 상 번호 가져오기
        line_phases = dssElem.BusNames[0].split('.')[1:]  # Bus 이름에서 상 번호 가져오기
        line_from_bus =  dssElem.BusNames[0].split('.')[0]
        line_to_bus =  dssElem.BusNames[1].split('.')[0]
        
        # 전류와 전력을 딕셔너리에 저장
        line_currents[line_name] = current_per_phase
        line_powers[line_name] = {
            'real': real_power_per_phase,
            'reactive': reactive_power_per_phase,
            'phases': line_phases,  # 상 번호 저장
            'from_bus': line_from_bus,
            'to_bus': line_to_bus,
            'resistance': rmatrix,
            'reactance': xmatrix,
            'susceptance': cmatrix
        }
        
    # 선로 전류 및 유효/무효전력 데이터프레임 생성
    line_data = []
    for line, currents in line_currents.items():
        row = {'Line': line}
        
        row['From'] = line_powers[line]['from_bus']
        row['To'] = line_powers[line]['to_bus']
        
        num_phases = len(line_powers[line]['phases'])
        # 각 선로의 상 번호에 맞게 데이터를 처리
        for i in range(num_phases):
            phase = line_powers[line]['phases'][i]  # 상 번호 가져오기 (1, 2, 3 등)
            row[f'Im_{phase}'] = currents[i * 2]  # 전류 크기
            row[f'Ia_{phase}'] = currents[i * 2 + 1]  # 전류 위상각
            row[f'Pbr_{phase}'] = line_powers[line]['real'][i]  # 유효전력 (P)
            row[f'Qbr_{phase}'] = line_powers[line]['reactive'][i]  # 무효전력 (Q)
        
        row['R'] = line_powers[line]['resistance']
        row['X'] = line_powers[line]['reactance']
        row['C'] = line_powers[line]['susceptance']    
        
        line_data.append(row)
    
    # 데이터프레임 출력
    line_df = pd.DataFrame(line_data)
    
    return line_df

def run_opendss(dss, df, factor_dict):
    # 시스템 초기화 및 기본 설정
    line_length_factor = factor_dict['line_length']
    
    dss_text = dss.Text
    
    dss_text.Command = "Clear"
    #dss_text.Command = "Edit Vsource.Source angle=0.0"
    #dss_text.Command = "New Energymeter.M1  Line.L1  1"

    dss_text.Command = "Set DefaultBaseFrequency=60"

    # 회로 초기화 및 설정
    dss_text.Command = "New Circuit.ieee34-1 basekv=24.9 pu=1.0 angle=0 mvasc3=200000"
    # dss_text.Command = "~ basekv=69 pu=1.05 angle=30 mvasc3=200000"

    # # Substation Transformer
    # dss_text.Command = "New Transformer.SubXF Phases=3 Windings=2 Xhl=0.001 ppm=0"
    # dss_text.Command = "~ wdg=1 bus=sourcebus conn=Delta kv=69 kva=25000 %r=0.0005"
    # dss_text.Command = "~ wdg=2 bus=800 conn=wye kv=24.9 kva=25000 %r=0.0005"

    # 전압 소스 설정
    #dss_text.Command = "New Vsource.Source Bus1=sourcebus Phases=3 BasekV=69 pu=1.05 Angle=30"
    dss_text.Command = "Edit Vsource.Source pu=1.0"

    # 라인 코드 가져오기
    dss_text.Command = "Redirect IEEELineCodes.dss"

    # 라인 및 중간 버스 정의
    for _, row in df['Lines'].iterrows():
        command = (f"New Line.{row['Name']} "
                    f"Phases={row['Phases']} "
                    f"Bus1={row['Bus1']} "
                    f"Bus2={row['Bus2']} "
                    f"LineCode={row['LineCode']} "
                    f"Length={row['Length'] * line_length_factor} "
                    f"units={row['units']}")    
        # OpenDSS 명령어 실행
        dss_text.Command = command
    
    #! 24.9/4.16 kV  Transformer
    if df['Transformer'].shape[0] >= 1:
        print("Transformer On")
        dss_text.Command = "New Transformer.XFM1  Phases=3 Windings=2 Xhl=4.08"
        dss_text.Command = "~ wdg=1 bus=832       conn=wye   kv=24.9  kva=500    %r=0.95"
        dss_text.Command = "~ wdg=2 bus=888       conn=Wye   kv=4.16  kva=500    %r=0.95"



    # Capacitors 정의
    for _, row in df['Capacitors'].iterrows():
        command = (f"New Capacitor.{row['Name']} "
                    f"Bus1={row['Bus1']} "
                    f"Phases={row['Phases']} "
                    f"kVAR={row['kVAR']} "
                    f"kV={row['kV']} ")    
        # OpenDSS 명령어 실행
        dss_text.Command = command
        

    # Regulators 정의 (Regulator 1, Regulator 2 등)
    #! Regulator 1
    for _, row in df['Regulator'].iterrows():
        command = (f"New transformer.{row['Name']} "
                    f"Phases={row['Phases']} "
                    "winding=2 "
                    f"bank={row['Bank']} "
                    f"buses=({row['Buses']}) "
                    f"conns=\'{row['Conns']}\' "
                    f"kvs =\"{row['kVs']} {row['kVs']}\" "
                    f"kvas =\"{row['kVAs']} {row['kVAs']}\" "
                    f"XHL={row['XHL']} ")    
    
        # OpenDSS 명령어 실행
        dss_text.Command = command
        
        command1 = (f"New regcontrol.c{row['Name']} "
                   f"transformer={row['Name']} " 
                   "winding=2 "        
                   f"vreg={row['vreg']} "
                   f"band={row['band']} "
                   f"ptratio={row['ptratio']} "
                   f"ctprim={row['ctprim']} "
                   f"R={row['R']} "
                   f"X={row['X']} ") 
        
        dss_text.Command = command1
    
    
    # Load 정의 (Spot Loads, Distributed Loads 등)
    for _, row in df['Spot Loads'].iterrows():
        command = (f"New Load.{row['Name']} "
                   f"Bus1={row['Bus1']} "
                   f"Phases={row['Phases']} "
                   f"Conn={row['Conn']} "               
                   f"Model={row['Model']} "
                   f"kV={row['kV']} "
                   f"kW={row['kW']} "
                   f"kVAR={row['kVAR']}")    
        # OpenDSS 명령어 실행
        dss_text.Command = command
        
        command1 = (f"Load.{row['Name']}.vminpu=0.85 daily=default")
        dss_text.Command = command1
         
    # ! distributed loads connected to line mid points
    for _, row in df['Distributed Loads'].iterrows():
        command = (f"New Load.{row['Name']} "
                   f"Bus1={row['Bus1']} "
                   f"Phases={row['Phases']} "
                   f"Conn={row['Conn']} "
                   f"Model={row['Model']} "
                   f"kV={row['kV']} "
                   f"kW={row['kW']} "
                   f"kVAR={row['kVAR']}")    
        # OpenDSS 명령어 실행
        dss_text.Command = command
        
        command1 = (f"Load.{row['Name']}.vminpu=0.85 daily=default")
        dss_text.Command = command1
        
    # 전압 베이스 설정 및 계산
    dss_text.Command = "Set VoltageBases=[69, 24.9, 4.16, 0.48]"
    dss_text.Command = "CalcVoltageBases"
    dss_text.Command = "solve"
    
    return dss


def modified_generate_ieee34_file(case_path):
    
    import pandas as pd
    from openpyxl import Workbook

    # 데이터 입력
    
    line_data = [
        ["L1", 3, "800.1.2.3", "802.1.2.3", 300, 2.58, "kft"],
        ["L2", 3, "802.1.2.3", "806.1.2.3", 300, 1.73, "kft"],
        ["L3", 3, "806.1.2.3", "808.1.2.3", 300, 32.23, "kft"],
        ["L4", 1, "808.2", "810.2", 303, 5.804, "kft"],
        ["L5", 3, "808.1.2.3", "812.1.2.3", 300, 37.5, "kft"],
        ["L6", 3, "812.1.2.3", "814.1.2.3", 300, 29.73, "kft"],
        ["L7", 3, "814.1.2.3", "850.1.2.3", 301, 0.01, "kft"],
        ["L8", 1, "816.1", "818.1", 302, 1.71, "kft"],
        ["L9", 3, "816.1.2.3", "824.1.2.3", 301, 10.21, "kft"],
        ["L10", 1, "818.1", "820.1", 302, 48.15, "kft"],
        ["L11", 1, "820.1", "822.1", 302, 13.74, "kft"],
        ["L12", 1, "824.2", "826.2", 303, 3.03, "kft"],
        ["L13", 3, "824.1.2.3", "828.1.2.3", 301, 0.84, "kft"],
        ["L14", 3, "828.1.2.3", "830.1.2.3", 301, 20.44, "kft"],
        ["L15", 3, "830.1.2.3", "854.1.2.3", 301, 0.52, "kft"],
        ["L16", 3, "832.1.2.3", "858.1.2.3", 301, 4.9, "kft"],
        ["L17", 3, "834.1.2.3", "860.1.2.3", 301, 2.02, "kft"],
        ["L18", 3, "834.1.2.3", "842.1.2.3", 301, 0.28, "kft"],
        ["L19", 3, "836.1.2.3", "840.1.2.3", 301, 0.86, "kft"],
        ["L20", 3, "836.1.2.3", "862.1.2.3", 301, 0.28, "kft"],
        ["L21", 3, "842.1.2.3", "844.1.2.3", 301, 1.35, "kft"],
        ["L22", 3, "844.1.2.3", "846.1.2.3", 301, 3.64, "kft"],
        ["L23", 3, "846.1.2.3", "848.1.2.3", 301, 0.53, "kft"],
        ["L24", 3, "850.1.2.3", "816.1.2.3", 301, 0.31, "kft"],
        ["L25", 3, "852.1.2.3", "832.1.2.3", 301, 0.01, "kft"],
        ["L26", 1, "854.2", "856.2", 303, 23.33, "kft"],
        ["L27", 3, "854.1.2.3", "852.1.2.3", 301, 36.83, "kft"],
        ["L28", 1, "858.1", "864.1", 302, 1.62, "kft"],
        ["L29", 3, "858.1.2.3", "834.1.2.3", 301, 5.83, "kft"],
        ["L30", 3, "860.1.2.3", "836.1.2.3", 301, 2.68, "kft"],
        ["L31", 1, "862.2", "838.2", 304, 4.86, "kft"],
        ["L32", 3, "888.1.2.3", "890.1.2.3", 300, 10.56, "kft"],
        ["L33", 3, "832.1.2.3", "888.1.2.3", 300, 0.01, "kft"]       
     
    ]
    
    # 컬럼 이름 설정
    columns = ["Name", "Phases", "Bus1", "Bus2", "LineCode", "Length", "units"]

    # Pandas DataFrame 생성
    line_df = pd.DataFrame(line_data, columns=columns)
    
    # Transformer 데이터    
    transformer_data = []
    # transformer_data = [
    #     [1, "832", "wye", 24.9, 500, 0.95],
    #     [2, "888", "wye", 4.16, 500, 0.95]
    # ]    
    transformer_columns = ["wdg", "bus", "conn", "kv", "kva", "%r"]
    transformer_df = pd.DataFrame(transformer_data, columns=transformer_columns)
    
    # # Capacitor 데이터
    capacitor_data = []
    # capacitor_data = [
    #     ["C844", "844", 3, 300, 24.9],
    #     ["C848", "848", 3, 450, 24.9]
    # ]
    capacitor_columns = ["Name", "Bus1", "Phases", "kVAR", "kV"]
    capacitor_df = pd.DataFrame(capacitor_data, columns=capacitor_columns)
    
    # # Regulator 데이터
    regulator_data = []
    # regulator_data = [
    #     ["reg1a", 1, "reg1", "814.1 814r.1", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg1b", 1, "reg1", "814.2 814r.2", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg1c", 1, "reg1", "814.3 814r.3", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg2a", 1, "reg2", "852.1 852r.1", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5],
    #     ["reg2b", 1, "reg2", "852.2 852r.2", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5],
    #     ["reg2c", 1, "reg2", "852.3 852r.3", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5]
    # ]
    regulator_columns = ["Name", "Phases", "Bank", "Buses", "Conns", "kVs", "kVAs", "XHL", "vreg", "band", "ptratio", "ctprim", "R", "X"]
    regulator_df = pd.DataFrame(regulator_data, columns=regulator_columns)
    
    # Spot Load 데이터
    spot_load_data = [
        ["S860", "860", 3, "Wye", 1, 24.9, 60.0, 48.0],
        ["S840", "840", 3, "Wye", 1, 24.9, 27.0, 21.0],
        ["S844", "844", 3, "Wye", 1, 24.9, 405.0, 315.0],
        ["S848", "848", 3, "Delta", 1, 24.9, 60.0, 48.0],
        ["S830a", "830.1.2", 1, "Delta", 1, 24.9, 10.0, 5.0],
        ["S830b", "830.2.3", 1, "Delta", 1, 24.9, 10.0, 5.0],
        ["S830c", "830.3.1", 1, "Delta", 1, 24.9, 25.0, 10.0],
        ["S890", "890", 3, "Delta", 1, 24.9, 450.0, 225.0]
        #["S890", "890", 3, "Delta", 5, 4.16, 450.0, 225.0]
    ]
    spot_load_columns = ["Name", "Bus1", "Phases", "Conn", "Model", "kV", "kW", "kVAR"]
    spot_load_df = pd.DataFrame(spot_load_data, columns=spot_load_columns)

    # Distributed Load 데이터
    distributed_load_data = [
        ["D802_806sb", "802.2", 1, "Wye", 1, 14.376, 15.0, 7.5],
        ["D802_806rb", "806.2", 1, "Wye", 1, 14.376, 15.0, 7.5],
        ["D802_806sc", "802.3", 1, "Wye", 1, 14.376, 12.5, 7.0],
        ["D802_806rc", "806.3", 1, "Wye", 1, 14.376, 12.5, 7.0],
        ["D808_810sb", "808.2", 1, "Wye", 1, 14.376, 8.0, 4.0],
        ["D808_810rb", "810.2", 1, "Wye", 1, 14.376, 8.0, 4.0],
        ["D818_820sa", "818.1", 1, "Wye", 1, 14.376, 17.0, 8.5],
        ["D818_820ra", "820.1", 1, "Wye", 1, 14.376, 17.0, 8.5],
        ["D820_822sa", "820.1", 1, "Wye", 1, 14.376, 67.5, 35.0],
        ["D820_822ra", "822.1", 1, "Wye", 1, 14.376, 67.5, 35.0],
        ["D816_824sb", "816.2.3", 1, "Delta", 1, 24.900, 2.5, 1.0],
        ["D816_824rb", "824.2.3", 1, "Delta", 1, 24.900, 2.5, 1.0],
        ["D824_826sb", "824.2", 1, "Wye", 1, 14.376, 20.0, 10.0],
        ["D824_826rb", "826.2", 1, "Wye", 1, 14.376, 20.0, 10.0],
        ["D824_828sc", "824.3", 1, "Wye", 1, 14.376, 2.0, 1.0],
        ["D824_828rc", "828.3", 1, "Wye", 1, 14.376, 2.0, 1.0],
        ["D828_830sa", "828.1", 1, "Wye", 1, 14.376, 3.5, 1.5],
        ["D828_830ra", "830.1", 1, "Wye", 1, 14.376, 3.5, 1.5],
        ["D854_856sb", "854.2", 1, "Wye", 1, 14.376, 2.0, 1.0],
        ["D854_856rb", "856.2", 1, "Wye", 1, 14.376, 2.0, 1.0],
        ["D832_858sa", "832.1", 1, "Delta", 1, 24.900, 3.5, 1.5],
        ["D832_858ra", "858.1", 1, "Delta", 1, 24.900, 3.5, 1.5],
        ["D832_858sb", "832.2", 1, "Delta", 1, 24.900, 1.0, 0.5],
        ["D832_858rb", "858.2", 1, "Delta", 1, 24.900, 1.0, 0.5],
        ["D832_858sc", "832.3", 1, "Delta", 1, 24.900, 3.0, 1.5],
        ["D832_858rc", "858.3", 1, "Delta", 1, 24.900, 3.0, 1.5],
        ["D858_864sb", "858.1", 1, "Wye", 1, 14.376, 1.0, 0.5],
        ["D858_864rb", "864.1", 1, "Wye", 1, 14.376, 1.0, 0.5],
        ["D858_834sa", "858.1.2", 1, "Delta", 1, 24.900, 2.0, 1.0],
        ["D858_834ra", "834.1.2", 1, "Delta", 1, 24.900, 2.0, 1.0],
        ["D858_834sb", "858.2.3", 1, "Delta", 1, 24.900, 7.5, 4.0],
        ["D858_834rb", "834.2.3", 1, "Delta", 1, 24.900, 7.5, 4.0],
        ["D858_834sc", "858.3.1", 1, "Delta", 1, 24.900, 6.5, 3.5],
        ["D858_834rc", "834.3.1", 1, "Delta", 1, 24.900, 6.5, 3.5],
        ["D834_860sa", "834.1.2", 1, "Delta", 1, 24.900, 8.0, 4.0],
        ["D834_860ra", "860.1.2", 1, "Delta", 1, 24.900, 8.0, 4.0],
        ["D834_860sb", "834.2.3", 1, "Delta", 1, 24.900, 10.0, 5.0],
        ["D834_860rb", "860.2.3", 1, "Delta", 1, 24.900, 10.0, 5.0],
        ["D834_860sc", "834.3.1", 1, "Delta", 1, 24.900, 55.0, 27.5],
        ["D834_860rc", "860.3.1", 1, "Delta", 1, 24.900, 55.0, 27.5],
        ["D860_836sa", "860.1.2", 1, "Delta", 1, 24.900, 15.0, 7.5],
        ["D860_836ra", "836.1.2", 1, "Delta", 1, 24.900, 15.0, 7.5],
        ["D860_836sb", "860.2.3", 1, "Delta", 1, 24.900, 5.0, 3.0],
        ["D860_836rb", "836.2.3", 1, "Delta", 1, 24.900, 5.0, 3.0],
        ["D860_836sc", "860.3.1", 1, "Delta", 1, 24.900, 21.0, 11.0],
        ["D860_836rc", "836.3.1", 1, "Delta", 1, 24.900, 21.0, 11.0],
        ["D836_840sa", "836.1.2", 1, "Delta", 1, 24.900, 9.0, 4.5],
        ["D836_840ra", "840.1.2", 1, "Delta", 1, 24.900, 9.0, 4.5],
        ["D836_840sb", "836.2.3", 1, "Delta", 1, 24.900, 11.0, 5.5],
        ["D836_840rb", "840.2.3", 1, "Delta", 1, 24.900, 11.0, 5.5],
        ["D862_838sb", "862.2", 1, "Wye", 1, 14.376, 14.0, 7.0],
        ["D862_838rb", "838.2", 1, "Wye", 1, 14.376, 14.0, 7.0],
        ["D842_844sa", "842.1", 1, "Wye", 1, 14.376, 4.5, 2.5],
        ["D842_844ra", "844.1", 1, "Wye", 1, 14.376, 4.5, 2.5],
        ["D844_846sb", "844.2", 1, "Wye", 1, 14.376, 12.5, 6.0],
        ["D844_846rb", "846.2", 1, "Wye", 1, 14.376, 12.5, 6.0],
        ["D844_846sc", "844.3", 1, "Wye", 1, 14.376, 10.0, 5.5],
        ["D844_846rc", "846.3", 1, "Wye", 1, 14.376, 10.0, 5.5],
        ["D846_848sb", "846.2", 1, "Wye", 1, 14.376, 11.5, 5.5],
        ["D846_848rb", "848.2", 1, "Wye", 1, 14.376, 11.5, 5.5]
    ]
    distributed_load_columns = ["Name", "Bus1", "Phases", "Conn", "Model", "kV", "kW", "kVAR"]
    distributed_load_df = pd.DataFrame(distributed_load_data, columns=distributed_load_columns)

    # 엑셀 파일 생성 및 저장
    with pd.ExcelWriter(case_path) as writer:
        line_df.to_excel(writer, sheet_name="Lines", index=False)
        transformer_df.to_excel(writer, sheet_name="Transformer", index=False)
        capacitor_df.to_excel(writer, sheet_name="Capacitors", index=False)
        regulator_df.to_excel(writer, sheet_name="Regulator", index=False)
        spot_load_df.to_excel(writer, sheet_name="Spot Loads", index=False)
        distributed_load_df.to_excel(writer, sheet_name="Distributed Loads", index=False)
    
    print("ieee34_data.xlsx 파일이 생성되었습니다.")


def modified_generate_ieee34_file2(case_path):
    
    import pandas as pd
    from openpyxl import Workbook

    # 데이터 입력
    line_data = [
        ["L0", 3, "sourcebus.1.2.3", "800.1.2.3", 300, 0.001, "kft"],
        ["L1", 3, "800.1.2.3", "802.1.2.3", 300, 2.58, "kft"],
        ["L2a", 3, "802.1.2.3", "mid806.1.2.3", 300, 1.73 / 2, "kft"],
        ["L2b", 3, "mid806.1.2.3", "806.1.2.3", 300, 1.73 / 2, "kft"],
        ["L3", 3, "806.1.2.3", "808.1.2.3", 300, 32.23, "kft"],
        ["L4a", 1, "808.2", "Mid810.2", 303, 5.804 / 2, "kft"],
        ["L4b", 1, "Mid810.2", "810.2", 303, 5.804 / 2, "kft"],
        ["L5", 3, "808.1.2.3", "812.1.2.3", 300, 37.5, "kft"],
        ["L6", 3, "812.1.2.3", "814.1.2.3", 300, 29.73, "kft"],
        ["L7", 3, "814.1.2.3", "850.1.2.3", 301, 0.01, "kft"],
        ["L24", 3, "850.1.2.3", "816.1.2.3", 301, 0.31, "kft"],
        ["L8", 1, "816.1", "818.1", 302, 1.71, "kft"],
        ["L9a", 3, "816.1.2.3", "mid824.1.2.3", 301, 10.21 / 2, "kft"],
        ["L9b", 3, "mid824.1.2.3", "824.1.2.3", 301, 10.21 / 2, "kft"],
        ["L10a", 1, "818.1", "mid820.1", 302, 48.15 / 2, "kft"],
        ["L10b", 1, "mid820.1", "820.1", 302, 48.15 / 2, "kft"],
        ["L11a", 1, "820.1", "mid822.1", 302, 13.74 / 2, "kft"],
        ["L11b", 1, "mid822.1", "822.1", 302, 13.74 / 2, "kft"],
        ["L12a", 1, "824.2", "mid826.2", 303, 3.03 / 2, "kft"],
        ["L12b", 1, "mid826.2", "826.2", 303, 3.03 / 2, "kft"],
        ["L13a", 3, "824.1.2.3", "mid828.1.2.3", 301, 0.84 / 2, "kft"],
        ["L13b", 3, "mid828.1.2.3", "828.1.2.3", 301, 0.84 / 2, "kft"],
        ["L14a", 3, "828.1.2.3", "mid830.1.2.3", 301, 20.44 / 2, "kft"],
        ["L14b", 3, "mid830.1.2.3", "830.1.2.3", 301, 20.44 / 2, "kft"],
        ["L15", 3, "830.1.2.3", "854.1.2.3", 301, 0.52, "kft"],
        ["L16a", 3, "832.1.2.3", "mid858.1.2.3", 301, 4.9 / 2, "kft"],
        ["L16b", 3, "mid858.1.2.3", "858.1.2.3", 301, 4.9 / 2, "kft"],
        ["L29a", 3, "858.1.2.3", "mid834.1.2.3", 301, 5.83 / 2, "kft"],
        ["L29b", 3, "mid834.1.2.3", "834.1.2.3", 301, 5.83 / 2, "kft"],
        ["L18", 3, "834.1.2.3", "842.1.2.3", 301, 0.28, "kft"],
        ["L19a", 3, "836.1.2.3", "mid840.1.2.3", 301, 0.86 / 2, "kft"],
        ["L19b", 3, "mid840.1.2.3", "840.1.2.3", 301, 0.86 / 2, "kft"],
        ["L21a", 3, "842.1.2.3", "mid844.1.2.3", 301, 1.35 / 2, "kft"],
        ["L21b", 3, "mid844.1.2.3", "844.1.2.3", 301, 1.35 / 2, "kft"],
        ["L22a", 3, "844.1.2.3", "mid846.1.2.3", 301, 3.64 / 2, "kft"],
        ["L22b", 3, "mid846.1.2.3", "846.1.2.3", 301, 3.64 / 2, "kft"],
        ["L23a", 3, "846.1.2.3", "mid848.1.2.3", 301, 0.53 / 2, "kft"],
        ["L23b", 3, "mid848.1.2.3", "848.1.2.3", 301, 0.53 / 2, "kft"],
        ["L26a", 1, "854.2", "mid856.2", 303, 23.33 / 2, "kft"],
        ["L26b", 1, "mid856.2", "856.2", 303, 23.33 / 2, "kft"],
        ["L27", 3, "854.1.2.3", "852.1.2.3", 301, 36.83, "kft"],
        ["L25", 3, "852.1.2.3", "832.1.2.3", 301, 0.01, "kft"],
        ["L28a", 1, "858.1", "mid864.1", 302, 1.62 / 2, "kft"],
        ["L28b", 1, "mid864.1", "864.1", 302, 1.62 / 2, "kft"],
        ["L17a", 3, "834.1.2.3", "mid860.1.2.3", 301, 2.02 / 2, "kft"],
        ["L17b", 3, "mid860.1.2.3", "860.1.2.3", 301, 2.02 / 2, "kft"],
        ["L30a", 3, "860.1.2.3", "mid836.1.2.3", 301, 2.68 / 2, "kft"],
        ["L30b", 3, "mid836.1.2.3", "836.1.2.3", 301, 2.68 / 2, "kft"],
        ["L20", 3, "836.1.2.3", "862.1.2.3", 301, 0.28, "kft"],
        ["L31a", 1, "862.2", "mid838.2", 304, 4.86 / 2, "kft"],
        ["L31b", 1, "mid838.2", "838.2", 304, 4.86 / 2, "kft"],
        ["L32", 3, "888.1.2.3", "890.1.2.3", 300, 10.56, "kft"],
        ["L33", 3, "832.1.2.3", "888.1.2.3", 300, 0.01, "kft"]  
    ]
        
    # 컬럼 이름 설정
    columns = ["Name", "Phases", "Bus1", "Bus2", "LineCode", "Length", "units"]
    
    # Pandas DataFrame 생성
    line_df = pd.DataFrame(line_data, columns=columns)
    
    # Transformer 데이터    
    transformer_data = []
    # transformer_data = [
    #     [1, "832", "wye", 24.9, 500, 0.95],
    #     [2, "888", "wye", 4.16, 500, 0.95]
    # ]    
    transformer_columns = ["wdg", "bus", "conn", "kv", "kva", "%r"]
    transformer_df = pd.DataFrame(transformer_data, columns=transformer_columns)
    
    # # Capacitor 데이터
    capacitor_data = []
    # capacitor_data = [
    #     ["C844", "844", 3, 300, 24.9],
    #     ["C848", "848", 3, 450, 24.9]
    # ]
    capacitor_columns = ["Name", "Bus1", "Phases", "kVAR", "kV"]
    capacitor_df = pd.DataFrame(capacitor_data, columns=capacitor_columns)
    
    # # Regulator 데이터
    regulator_data = []
    # regulator_data = [
    #     ["reg1a", 1, "reg1", "814.1 814r.1", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg1b", 1, "reg1", "814.2 814r.2", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg1c", 1, "reg1", "814.3 814r.3", "wye wye", 14.376, 20000, 1, 122, 2, 120, 100, 2.7, 1.6],
    #     ["reg2a", 1, "reg2", "852.1 852r.1", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5],
    #     ["reg2b", 1, "reg2", "852.2 852r.2", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5],
    #     ["reg2c", 1, "reg2", "852.3 852r.3", "wye wye", 14.376, 20000, 1, 124, 2, 120, 100, 2.5, 1.5]
    # ]
    regulator_columns = ["Name", "Phases", "Bank", "Buses", "Conns", "kVs", "kVAs", "XHL", "vreg", "band", "ptratio", "ctprim", "R", "X"]
    regulator_df = pd.DataFrame(regulator_data, columns=regulator_columns)
    
    # Spot Load 데이터
    spot_load_data = [
        ["S860", "860", 3, "Wye", 1, 24.9, 60.0, 48.0],
        ["S840", "840", 3, "Wye", 1, 24.9, 27.0, 21.0],
        ["S844", "844", 3, "Wye", 1, 24.9, 405.0, 315.0],
        ["S848", "848", 3, "Delta", 1, 24.9, 60.0, 48.0],
        ["S830a", "830.1.2", 1, "Delta", 1, 24.9, 10.0, 5.0],
        ["S830b", "830.2.3", 1, "Delta", 1, 24.9, 10.0, 5.0],
        ["S830c", "830.3.1", 1, "Delta", 1, 24.9, 25.0, 10.0],
        ["S890", "890", 3, "Delta", 1, 24.9, 450.0, 225.0]
        #["S890", "890", 3, "Delta", 5, 4.16, 450.0, 225.0]
    ]
    
    # spot_load_data = [
    #     ["S860", "860", 3, "Wye", 1, 24.9, 0, 0],
    #     ["S840", "840", 3, "Wye", 1, 24.9, 0, 0],
    #     ["S844", "844", 3, "Wye", 1, 24.9, 0, 0],
    #     ["S848", "848", 3, "Delta", 1, 24.9, 0, 0],
    #     ["S830a", "830.1.2", 1, "Delta", 1, 24.9, 0, 0],
    #     ["S830b", "830.2.3", 1, "Delta", 1, 24.9, 0, 0],
    #     ["S830c", "830.3.1", 1, "Delta", 1, 24.9, 0, 0],
    #     ["S890", "890", 3, "Delta", 1, 24.9, 0, 0]
    #     #["S890", "890", 3, "Delta", 5, 4.16, 450.0, 225.0]
    # ]    
    
    spot_load_columns = ["Name", "Bus1", "Phases", "Conn", "Model", "kV", "kW", "kVAR"]
    spot_load_df = pd.DataFrame(spot_load_data, columns=spot_load_columns)
    
    # 데이터 입력
    
    distributed_load_data = [
        ["D802_806sb", "802.2", 1, "Wye", 1, 14.376, 15.0, 7.5], 
        ["D802_806b", "Mid806.2", 1, "Wye", 1, 14.376, 30, 15], 
        ["D802_806c", "Mid806.3", 1, "Wye", 1, 14.376, 25, 14], 
        ["D808_810b", "Mid810.2", 1, "Wye", 5, 14.376, 16, 8], 
        ["D818_820a", "mid820.1", 1, "Wye", 2, 14.376, 34, 17], 
        ["D820_822a", "mid822.1", 1, "Wye", 1, 14.376, 135, 70], 
        ["D816_824b", "mid824.2.3", 1, "Delta", 5, 24.900, 5, 2], 
        ["D824_826b", "mid826.2", 1, "Wye", 5, 14.376, 40.0, 20.0], 
        ["D824_828c", "mid828.3", 1, "Wye", 1, 14.376, 4.0, 2.0], 
        ["D828_830a", "mid830.1", 1, "Wye", 1, 14.376, 7, 3], 
        ["D854_856b", "mid856.2", 1, "Wye", 1, 14.376, 4, 2], 
        ["D832_858a", "mid858.1.2", 1, "Delta", 2, 24.900, 7, 3], 
        ["D832_858b", "mid858.2.3", 1, "Delta", 2, 24.900, 2, 1], 
        ["D832_858c", "mid858.3.1", 1, "Delta", 2, 24.900, 6, 3], 
        ["D858_864a", "mid864.1", 1, "Wye", 1, 14.376, 2, 1], 
        ["D858_834a", "mid834.1.2", 1, "Delta", 1, 24.900, 4.0, 2.0], 
        ["D858_834b", "mid834.2.3", 1, "Delta", 1, 24.900, 15, 8], 
        ["D858_834c", "mid834.3.1", 1, "Delta", 1, 24.900, 13, 7], 
        ["D834_860a", "mid860.1.2", 1, "Delta", 2, 24.900, 16, 8], 
        ["D834_860b", "mid860.2.3", 1, "Delta", 2, 24.900, 20.0, 10], 
        ["D834_860c", "mid860.3.1", 1, "Delta", 2, 24.900, 110, 55], 
        ["D860_836a", "mid836.1.2", 1, "Delta", 1, 24.900, 30, 15], 
        ["D860_836b", "mid836.2.3", 1, "Delta", 1, 24.900, 10, 6], 
        ["D860_836c", "mid836.3.1", 1, "Delta", 1, 24.900, 42, 22], 
        ["D836_840a", "mid840.1.2", 1, "Delta", 5, 24.900, 18, 9], 
        ["D836_840b", "mid840.2.3", 1, "Delta", 5, 24.900, 22, 11], 
        ["D862_838b", "mid838.2", 1, "Wye", 1, 14.376, 28.0, 14], 
        ["D842_844a", "mid844.1", 1, "Wye", 1, 14.376, 9, 5], 
        ["D844_846b", "mid846.2", 1, "Wye", 1, 14.376, 25, 12], 
        ["D844_846c", "mid846.3", 1, "Wye", 1, 14.376, 20, 11], 
        ["D846_848b", "mid848.2", 1, "Wye", 1, 14.376, 23, 11]
    ]
    
    # distributed_load_data = [
    #     ["D802_806sb", "802.2", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D802_806b", "Mid806.2", 1, "Wye", 1, 14.376,  0, 0], 
    #     ["D802_806c", "Mid806.3", 1, "Wye", 1, 14.376,  0, 0], 
    #     ["D808_810b", "Mid810.2", 1, "Wye", 5, 14.376,  0, 0], 
    #     ["D818_820a", "mid820.1", 1, "Wye", 2, 14.376,  0, 0], 
    #     ["D820_822a", "mid822.1", 1, "Wye", 1, 14.376,  0, 0], 
    #     ["D816_824b", "mid824.2.3", 1, "Delta", 5, 24.900,  0, 0], 
    #     ["D824_826b", "mid826.2", 1, "Wye", 5, 14.376,  0, 0], 
    #     ["D824_828c", "mid828.3", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D828_830a", "mid830.1", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D854_856b", "mid856.2", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D832_858a", "mid858.1.2", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D832_858b", "mid858.2.3", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D832_858c", "mid858.3.1", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D858_864a", "mid864.1", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D858_834a", "mid834.1.2", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D858_834b", "mid834.2.3", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D858_834c", "mid834.3.1", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D834_860a", "mid860.1.2", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D834_860b", "mid860.2.3", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D834_860c", "mid860.3.1", 1, "Delta", 2, 24.900, 0, 0], 
    #     ["D860_836a", "mid836.1.2", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D860_836b", "mid836.2.3", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D860_836c", "mid836.3.1", 1, "Delta", 1, 24.900, 0, 0], 
    #     ["D836_840a", "mid840.1.2", 1, "Delta", 5, 24.900, 0, 0], 
    #     ["D836_840b", "mid840.2.3", 1, "Delta", 5, 24.900, 0, 0], 
    #     ["D862_838b", "mid838.2", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D842_844a", "mid844.1", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D844_846b", "mid846.2", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D844_846c", "mid846.3", 1, "Wye", 1, 14.376, 0, 0], 
    #     ["D846_848b", "mid848.2", 1, "Wye", 1, 14.376, 0, 0]
    # ]
    
    
    distributed_load_columns = ["Name", "Bus1", "Phases", "Conn", "Model", "kV", "kW", "kVAR"]
    distributed_load_df = pd.DataFrame(distributed_load_data, columns=distributed_load_columns)

    # 엑셀 파일 생성 및 저장
    with pd.ExcelWriter(case_path) as writer:
        line_df.to_excel(writer, sheet_name="Lines", index=False)
        transformer_df.to_excel(writer, sheet_name="Transformer", index=False)
        capacitor_df.to_excel(writer, sheet_name="Capacitors", index=False)
        regulator_df.to_excel(writer, sheet_name="Regulator", index=False)
        spot_load_df.to_excel(writer, sheet_name="Spot Loads", index=False)
        distributed_load_df.to_excel(writer, sheet_name="Distributed Loads", index=False)
    
    print("ieee34_data.xlsx 파일이 생성되었습니다.")

