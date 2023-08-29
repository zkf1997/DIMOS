import yaml
import os, sys
import socket

def get_host_name():
    return hostname


# smplx body model
def get_body_model_path():
    bmpath = 'data/models_smplx_v1_1/models/'
    return bmpath

# marker placement data
def get_body_marker_path():
    mkpath = 'data/models_smplx_v1_1/models/markers'
    return mkpath

def get_amass_canonicalized_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-nfs/kaizhao/datasets/amass/AMASS-Canonicalized-MP/data'
    elif 'wks' in hostname:
        mkpath = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MP/data'
    else:
        raise ValueError('not stored here')
    return mkpath

def get_amass_canonicalizedx10_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/vlg-nfs/kaizhao/datasets/amass/AMASS-Canonicalized-MPx10/data'
    elif 'wks' in hostname:
        mkpath = '/home/kaizhao/dataset/amass/AMASS-Canonicalized-locomotion-MPx10/data'
    else:
        raise ValueError('not stored here')
    return mkpath




hostname = socket.gethostname()
print('host name:', hostname)



















