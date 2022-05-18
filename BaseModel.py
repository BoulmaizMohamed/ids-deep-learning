"""
Created on Sat May 14 13:54:05 2022

@author: bauhaus
"""

from pydantic import BaseModel

class Model(BaseModel):
    id:                    int
    dur:                   float
    proto:                 int
    service:               int
    state:                 int
    spkts:                 int
    dpkts:                 int
    sbytes:                int
    dbytes:                int
    rate:                 float
    sttl:                  int
    dttl:                  int
    sload:                float
    dload:                float
    sloss:                 int
    dloss:                 int
    sinpkt:               float
    dinpkt:               float
    sjit:                 float
    djit:                 float
    swin:                   int
    stcpb:                  int
    dtcpb:                 int
    dwin:                   int
    tcprtt:               float
    synack:               float
    ackdat:               float
    smean:                  int
    dmean:                  int
    trans_depth:            int
    response_body_len:      int
    ct_srv_src:             int
    ct_state_ttl:           int
    ct_dst_ltm:             int
    ct_src_dport_ltm:       int
    ct_dst_sport_ltm:       int
    ct_dst_src_ltm:         int
    is_ftp_login:           int
    ct_ftp_cmd:             int
    ct_flw_http_mthd:       int
    ct_src_ltm:             int
    ct_srv_dst:             int
    is_sm_ips_ports:        int
    
