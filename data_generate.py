#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:56:03 2021

Generation of Data Set
- generation of inp files and rainfall files 
- rainfall is based on Huff method

@author: drminor
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

np.random.seed(1234)

# Quartile weights of Huff method
weights = np.array([[0.546292, 0.141443, -0.00515813, 7.94825e-5, -5.77417e-7, 1.61502e-09],
           [0.421947, -0.0380057, 0.00434001, -0.000104192, 9.7863e-7, -3.26939e-9],
           [-0.18446, 0.0813196, -0.00423723, 0.000104273, -1.08264e-6, 3.94163e-9],
           [0.473655, -0.0409675, 0.00278422, -6.97028e-5, 7.68968e-7, -3.04136e-9]])



def plot_cum_rainfall(cum_rainfalls:'np.array'):
    '''
    Huff 수식에 의해 생성된 누적 강우량 그래프

    Parameters
    ----------
    cum_rainfalls : 'np.array'
        cumulative rainfalls

    Returns
    -------
    None.

    '''
    if isinstance(cum_rainfalls, list):
        cum_rainfalls = np.asarray(cum_rainfalls)
    else:
        np.expand_dims(cum_rainfalls, axis=0)

    plot_args = ['ro-', 'bs-', 'gv-', 'k^-']
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('Cumulative Rainfall')
    ax.set_xlabel('Time(minutes)')
    ax.set_ylabel('Rainfall(mm)')
    for i, rain in enumerate(cum_rainfalls):
        ax.plot(np.arange(rain.shape[0])+1, rain, plot_args[i], label=str(i+1)+'-'+'quantile')
    ax.legend()
    
    fig.show()


# def write_inpfile(cum_rainfalls:'np.array', in_filepath:str, out_filepath:str):
#     contents = []
    
#     with open(in_filepath, 'r') as in_fd:
#         with open(out_filepath, 'w') as out_fd:
#             while True:
#                 line = in_fd.readline()
#                 if not line:
#                     break
#                 print(line.strip())
#                 out_fd.write(line)


def compose_inpfile(cum_rainfalls:'np.array', 
                  sample_inp:str, 
                  year=10, 
                  duration=60)->list:
    '''
    sample.inp 파일과 누적 강우로부터 inp file 생성 

    Parameters
    ----------
    cum_rainfalls : 'np.array'
        누적 강우량
    sample_inp : str
        sample inp file, 지역별 inp 파일을 만들기위한 frame file
    year : TYPE, optional
        강우 빈도 The default is 10 year.
    duration : TYPE, optional
        강우 기간. The default is 60 minutes.

    Returns
    -------
    list
        list of inpfile contents 
    
    ;
    OutfallSeries    8/14/2006  0:1        4.7  

    '''
    contents = []
    
    with open(sample_inp, 'r') as in_fd:
        while True:
            line = in_fd.readline()
            if not line:
                break
            if 'END_TIME' in line.strip():
                end_time = f'{int(duration/60 + 1):02d}:{int(duration%60):02d}:00\n'
                line = line[:-9] + end_time
                contents.append(line)    
            elif '[TIMESERIES]' in line.strip():
                contents.append(line) # don't miss appending  [TIMESERIES] line
                contents.append(in_fd.readline())
                contents.append(in_fd.readline())
                for t, value in enumerate(cum_rainfalls):
                    string = f'Hyeto            8/01/2023  {(t+1)//60}:{(t+1)%60:02d}       {value}\n'
                    contents.append(string)
            else:
                contents.append(line)
    
    return contents
            

def compose_gasan_inpfile(cum_rainfalls:'np.array', 
                  sample_inp:str, 
                  year=10, 
                  duration=60)->list:
    '''
    sample.inp 파일과 누적 강우로부터 inp file 생성 

    Parameters
    ----------
    cum_rainfalls : 'np.array'
        누적 강우량
    sample_inp : str
        sample inp file, 지역별 inp 파일을 만들기위한 frame file
    year : TYPE, optional
        강우 빈도 The default is 10 year.
    duration : TYPE, optional
        강우 기간. The default is 60 minutes.

    Returns
    -------
    list
        list of inpfile contents 
    
    ;
    OutfallSeries    8/14/2006  0:1        4.7  

    '''
    contents = []
    
    with open(sample_inp, 'r') as in_fd:
        while True:
            line = in_fd.readline()
            if not line:
                break
            if 'END_TIME' in line.strip():
                end_time = f'{int(duration/60 + 1):02d}:{int(duration%60):02d}:00\n'
                line = line[:-9] + end_time
                contents.append(line)    
            elif '[TIMESERIES]' in line.strip():
                contents.append(line) # don't miss appending  [TIMESERIES] line
                contents.append(in_fd.readline())
                contents.append(in_fd.readline())
                for t, value in enumerate(cum_rainfalls):
                    string = f'Hyeto            8/01/2023  {(t+1)//60}:{(t+1)%60:02d}       {value}\n'
                    contents.append(string)
                contents.append(';\n')
                for t in range(360):
                    string = f'OutfallSeries    8/01/2023  {(t+1)//60}:{(t+1)%60:02d}        4.7\n'
                    contents.append(string)
            else:
                contents.append(line)
    
    return contents
            
            
def generate_rainfall(rainfall:int, 
                      duration:int, 
                      raintype=1) -> 'np.array':
    '''
    generate rainfall using huff zero dimension function

    Parameters
    ----------
    rainfall : int 'mm'
        total quantity of rainfall
    duration : int 'minutes'
        분단위 강우 지속 시간
    raintype : 'quartile:(1, 2, 3, 4)', optional
        4가지 강우 패턴. The default is 1.

    Returns
    -------
    TYPE numpy.array
        cumulative rainfalls with minutes

    '''
    cum_rainfalls = [] # cumulative rainfalls
    for i in range(duration):
        x = (i+1) / duration * 100.
        value = np.array([x, x**2, x**3, x**4, x**5, x**6])
        cum_rainfalls.append(round((weights[raintype-1] @ value) * rainfall / 100., 2))
    
    cv1, cv2, _ = confirm_valid(np.array(cum_rainfalls))
    if cv1 == False: # if minus rainfall, set it to zero value 
        for i in range(len(cum_rainfalls)):
            if cum_rainfalls[i] < 0.0:
                cum_rainfalls[i] = 0.0
    if cv2 == False: # if decrementaion phase, set it to the greastet value
        for i in range(len(cum_rainfalls)-1):
            if cum_rainfalls[i+1] < cum_rainfalls[i]:
                cum_rainfalls[i+1] = cum_rainfalls[i]
    if cv1 == False or cv2 == False:
        cv1, cv2, info = confirm_valid(np.array(cum_rainfalls))
        print(info)
    
    return np.array(cum_rainfalls)


def confirm_valid(cr:np.array):
    cv1 = all(cr >= 0.0)
    cv2 = True
    for i in range(len(cr)-1):
        cv2 = cr[i+1] - cr[i] >= 0.0
        if cv2 == False:
            break
    info = ('minus, decrement' if cv2 == False else 'minus') if cv1 == False else ('decrement' if cv2 == False else 'no problem')
    
    return cv1, cv2, info


def clear_rainfolder(regions):
    '''
    Clear all files in every ./rainfall/years/ folder

    Parameters
    ----------
    regions : list

    Returns
    -------
    None.

    '''
    #year_freq = [10, 20, 30, 50, 80] # year  
    dir_year_list = glob.glob(os.path.join(Path(__file__).parents[1], 'data/rainfall/*'))
    for dirpath in dir_year_list: # 년도 빈도
        flist = glob.glob(dirpath+'/*')
        for fpath in flist:
            os.remove(fpath)  


def clear_inpfolder(regions):
    '''
    Clear all files in every ../data/region/years/ folder

    Parameters
    ----------
    regions : list
    
    Returns
    -------
    None.

    '''
    
    year_dir = os.listdir(os.path.join(Path(__file__).parents[1], 'data/rainfall')) # year 
    
    for region in regions: 
        for y in year_dir: # 년도 빈도
            dirpath = os.path.join(Path(__file__).parents[1], 
                                   'data/'+region, y)
            print(f'Remove {dirpath}')
            flist = glob.glob(dirpath+'/*')
            for fpath in flist:
                os.remove(fpath)  


def write_datafiles(regions,
                    year_freq:list=[10, 20, 30, 50, 80],
                    duration:list=[10, 60, 120, 180, 240],
                    more:bool=False,
                    n_vars:int=30) -> 'rainfall.txt file generation':
    '''
    rainfall.txt file and inp file generation
    전주관측소 빈도강우량 참조

    Parameters
    ----------
    year_freq : list
        total quantity of rainfall
        [10, 20, 30, 50, 80] # year 
    duration : list 
        분단위 강우 지속 시간 (단위: 'minutes')
        [10, 60, 120, 180, 240, 360, 540, 720, 1080, 1440] # minutes
    more : bool
        build more rainfall data by using normal distribution with std rain * 0.1 (10%)
    n_vars : the number of more data per rainfall case
        default 30
        
    Returns
    -------
    None.

    '''
    
    # rainfall shape -> (year x minutes); 단위: mm
    #            10, 60, 120, 180, 240, 360, 540, 720, 1080, 1440(m)
    rainfall = [[22.5, 64.8, 87.1, 101.4, 114.8, 137, 158.4, 169.1, 186.6, 198.8],  #10years
                [24.9, 73.6, 99.3, 115.6, 131.1, 157, 181.9, 193.3, 213.1, 226.9],  #20years
                [26.3, 78.6, 106.3, 123.8, 140.5, 168.5, 195.5, 207.2, 228.4, 243], #30years
                [28, 84.9, 115.1, 134, 152.2, 183, 212.4, 224.6, 247.4, 263.2],     #50years
                [29.5, 90.6, 123.1, 143.3, 163, 196.1, 227.8, 240.5, 264.8, 281.6], #80years
                [30.3, 93.3, 126.9, 147.8, 168.1, 202.4, 235.2, 248, 273.1, 290.4]] #100years
    raintype = [1, 2, 3, 4]
    
    for i, y in enumerate(year_freq): # 년도 빈도
        n = 0
        
        # rainfall directory generation
        rain_dirpath = os.path.join(Path(__file__).parents[1], 
                                    'data/rainfall', f'{y}year')
        if not os.path.exists(rain_dirpath):
            print(f'building folder: {rain_dirpath}')
            os.makedirs(rain_dirpath)
        
        for region in regions:
            inp_dirpath = os.path.join(Path(__file__).parents[1], 
                                       'data/' + region, f'{y}year')
            if not os.path.exists(inp_dirpath):
                print(f'building folder: {inp_dirpath}')
                os.makedirs(inp_dirpath)
                
        for j, d in enumerate(duration): #강우 지속시간
            for type_ in raintype: #강우 유형
                cum_rainfalls = generate_rainfall(rainfall[i][j], d, type_)
                filename = f'{y}yr_{d:04}m_h{n:03}.txt'
                filepath = os.path.join(rain_dirpath, filename)
                #print(filepath)
                with open(filepath, 'w') as fd:
                    print('writing: ' + filepath)
                    cum_rainfalls.tofile(fd, sep='\n')
                
                for region in regions:
                    sample_inp = os.path.join(Path(__file__).parents[1], 
                                              'data/sample_inp', 
                                              'sample_'+region+'.inp')
                    #print(sample_inp)
                    inp_fname = f'{y}yr_{d:04}m_h{n:03}.inp'
                    inp_dirpath = os.path.join(Path(__file__).parents[1], 
                                               'data/' + region, f'{y}year')
                    inp_fpath = os.path.join(inp_dirpath, inp_fname)
                    #print(inp_fpath)
                    contents = compose_gasan_inpfile(cum_rainfalls, sample_inp, y, d)
                    with open(inp_fpath, 'w') as fd:
                        fd.writelines(contents)     
                n += 1
                if more is True: # Generate additional rainfall data
                    rain_list = more_rains(rainfall[i][j], n_vars=n_vars)
                    for k in rain_list:  # for each sample generated by normal distribution
                        cum_rainfalls = generate_rainfall(k, d, type_)
                        filename = f'{y}yr_{d:04}m_h{n:03}.txt'
                        filepath = os.path.join(rain_dirpath, filename)
                        #print(filepath)
                        with open(filepath, 'w') as fd:
                            #print('writing: ' + filepath)
                            cum_rainfalls.tofile(fd, sep='\n')
                        
                        for region in regions:
                            sample_inp = os.path.join(Path(__file__).parents[1], 
                                                      'data/sample_inp', 
                                                      'sample_'+region+'.inp')
                            #print(sample_inp)
                            inp_fname = f'{y}yr_{d:04}m_h{n:03}.inp'
                            inp_dirpath = os.path.join(Path(__file__).parents[1], 
                                                       'data/' + region, f'{y}year')
                            inp_fpath = os.path.join(inp_dirpath, inp_fname)
                            #print(inp_fpath)
                            contents = compose_gasan_inpfile(cum_rainfalls, sample_inp, y, d)
                            with open(inp_fpath, 'w') as fd:
                                fd.writelines(contents)
                        n += 1


def more_rains(rainfall:float, n_vars:int, std_r:float=0.5)->list:
    """
    build more rains data samples by using normal distribution

    Parameters
    ----------
    rainfall : float
        mean value for rainfall data
    n_vars : int, optional
        The number of additional samples. The default is 30.
    std_r : float
        standard deviation => rainfall * std_r
        
    Returns
    -------
    Type: list
    num rainfall samples

    """
    
    rain_list = []
    n = 0
    while(True):
        for i in range(n_vars):
            rain_list.append(round(np.random.normal(rainfall, std_r), 1))
        rain_arr = np.unique(np.array(rain_list))
        if len(rain_arr) < 30:
            rain_list = rain_arr.tolist()
        else:
            np.random.shuffle(rain_arr)
            return rain_arr[:n_vars].tolist()
        
        # to avoid too many iterations
        n += 1
        assert n < 100, 'Too many iteration to build rainfall samples in function more_rains'
            
    
if __name__ == '__main__':
    # Clear data folder
    regions = ['gasan'] #['bongdong', 'dooyeo', 'naechon']
    
    print(f'Experiments for {regions}')
    clear_rainfolder(regions)
    clear_inpfolder(regions)
    
    write_datafiles(regions, year_freq=[10, 20, 30, 50, 80, 100],
                    duration=[10, 60, 120, 180, 240, 360, 540, 720, 1080, 1440], more=True)
    
    #### For Test ####
    # rainfall = [[22.5, 64.8, 87.1, 101.4, 114.8, 137, 158.4, 169.1, 186.6, 198.8],  #10years
    #             [24.9, 73.6, 99.3, 115.6, 131.1, 157, 181.9, 193.3, 213.1, 226.9],  #20years
    #             [26.3, 78.6, 106.3, 123.8, 140.5, 168.5, 195.5, 207.2, 228.4, 243], #30years
    #             [28, 84.9, 115.1, 134, 152.2, 183, 212.4, 224.6, 247.4, 263.2],     #50years
    #             [29.5, 90.6, 123.1, 143.3, 163, 196.1, 227.8, 240.5, 264.8, 281.6], #80years
    #             [30.3, 93.3, 126.9, 147.8, 168.1, 202.4, 235.2, 248, 273.1, 290.4]] #100years
    # raintype = [1, 2, 3, 4]
    
    # year_freq=[10, 20, 30, 50, 80, 100],
    # duration=[10, 60, 120, 180, 240, 360, 540, 720, 1080, 1440]
    
    # for i in range(len(rainfall)):
    #     for j in range(len(rainfall[0])):
    #         first = generate_rainfall(rainfall[i][j], duration[j], 4)
    # contents = compose_inpfile(first, './data/WORK_211102/sample_naechon.inp', year=10, duration=60)
    # second = generate_rainfall(90, 12, 2)
    # third = generate_rainfall(90, 12, 3)
    # fourth = generate_rainfall(90, 12, 4)
    # plot_cum_rainfall([first, second, third, fourth])
