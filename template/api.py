"""
Created on Wed May 18 08:50:16 2022

@author: bauhaus
"""

#pip install fastapi uvicorn

# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI,Request
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# 2. Create the app object
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = LabelEncoder()
model = load_model("C:/Users/bauhaus/Desktop/pfe/Model.h5")

templates = Jinja2Templates(directory="templates")


#  Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


#    Located at: http://127.0.0.1:8000/predict

 

@app.post('/predict')
async def predict_attack(dataRequest:Request):


    data = await  dataRequest.json()

    id=data['id']
    dur=data['dur']
    proto=data['proto']
    service=data['service']
    state=data['state']
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
    
    """
    proto1=encoder.fit_transform(proto)
    service1=encoder.fit_transform(service)
    state1=encoder.fit_transform(state)
    """
   
   
    prediction = model.predict([[id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes
           ,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt
           ,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat
           ,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,
           ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login
           ,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports]])
    
    #prediction = prediction.round()
    print(prediction)
    
    if(prediction>0.5):
        prediction="attack"
    else:
        prediction="normal"
        
    return {'prediction': prediction}
    
    
    
    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
  #  uvicorn api2:app --reload
