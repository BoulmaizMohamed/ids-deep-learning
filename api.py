"""
Created on Wed May 18 08:50:16 2022

@author: bauhaus
"""

#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
from keras.models import load_model
from BaseModel import Model
from sklearn.preprocessing import LabelEncoder
# 2. Create the app object
app = FastAPI()
encoder = LabelEncoder()
model = load_model("C:/Users/bauhaus/Desktop/pfe/Model.h5")

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name():
    return {'Welcome To my project'}

@app.post('/predict')
def predict_banknote(data:Model):
    data = data.dict()
    
    id=data['id']
    dur=data['dur']
    proto=encoder.fit_transform(data['proto'])
    service=encoder.fit_transform(data['service'])
    state=encoder.fit_transform(data['state'])
    spkts=data['spkts']
    dpkts=data['dpkts']
    sbytes=data['sbytes']
    dbytes=data['dbytes']
    rate=data['rate']
    sttl=data['sttl']
    dttl=data['dttl']
    sload=data['sload']
    dload=data['dload']
    sloss=data['sloss']
    dloss=data['dloss']
    sinpkt=data['sinpkt']
    dinpkt=data['dinpkt']
    sjit=data['sjit']
    djit=data['djit']
    swin=data['swin']
    stcpb=data['stcpb']
    dtcpb=data['dtcpb']
    dwin=data['dwin']
    tcprtt=data['tcprtt']
    synack=data['synack']
    ackdat=data['ackdat']
    smean=data['smean']
    dmean=data['dmean']
    trans_depth=data['trans_depth']
    response_body_len=data['response_body_len']
    ct_srv_src=data['ct_srv_src']
    ct_state_ttl=data['ct_state_ttl']
    ct_dst_ltm=data['ct_dst_ltm']
    ct_src_dport_ltm=data['ct_src_dport_ltm']
    ct_dst_sport_ltm=data['ct_dst_sport_ltm']
    ct_dst_src_ltm=data['ct_dst_src_ltm']
    is_ftp_login=data['is_ftp_login']
    ct_ftp_cmd=data['ct_ftp_cmd']
    ct_flw_http_mthd=data['ct_flw_http_mthd']
    ct_src_ltm=data['ct_src_ltm']
    ct_srv_dst=data['ct_srv_dst']
    is_sm_ips_ports=data['is_sm_ips_ports']
    
    
   
    prediction = model.predict([[id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes
           ,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt
           ,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat
           ,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,
           ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login
           ,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports]])
    
    if(prediction[0]>0.5):
        prediction="attack"
    else:
        prediction="not an attack"
    return {
        'prediction': prediction
    }
    
    
    
    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
