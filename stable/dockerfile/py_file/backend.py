from flask import Flask, request
from flask_cors import CORS
import os

from kubernetes import client
#k8s
import k8s_object as k8s



app = Flask(__name__)
CORS(app)
@app.route("/")
def hello():
    return "welcome!"

#submit ray job
@app.route("/submit_job",methods=['POST'])
def submit():
    #reload ray_job.py from test/backup/
    os.system("cp backup/ray_job.py test/ray_job.py")
    name = str(request.args.get("username", "default"))
    worker_num = str(request.args.get("worker", 1))
    #worker num
    os.system("sed -i 's\\num_workers=1\\num_workers={worker}\g' test/ray_job.py".format(worker=worker_num))
    os.system("sed -i 's\\username\\{name}\g' test/ray_job.py".format(name=name))
    def job_submit(namespace, type, model=None):
        cmd = "RAY_ADDRESS=ray-head-svc.{name}:6379 ray job submit --working-dir test/ -- python3 ray_job.py".format(name=namespace)
        
        if type=="custom":
            os.system("sed -i 's\import model.model as model_fw\import model.custom_model as model_fw\g' test/ray_job.py")
            os.system(cmd)
            
        else:
            model = "model_" + model
            os.system("sed -i 's\import model.model as model_fw\import model.{model_name} as model_fw\g' test/ray_job.py".format(model_name=model))
            os.system(cmd)

    try:
        file = request.files["file"]
        if file.filename != "":
            file.save("test/custom_model/"+file.filename)
        job_submit(name, "custom")
        
        return "custom"
    except:
        model = request.args.get("modelName", "model")
        print(model)
        job_submit(name, "default", model)

        return "default"

    

#check namespace exist
@app.route("/check_namespace")
def check():
    namespace = request.args.get("username", "default")
    v1 = client.CoreV1Api()
    
    try:
        v1.read_namespace(namespace)
        return "old"
    except:
        return "new"

#create ray env
@app.route("/create_ray_env")
def creat_ray_env():
    user_name = request.args.get("username", "default")
    cpu_num = int(request.args.get("cpu", 6))
    memory_num = int(request.args.get("memory", 6))
    worker_num = int(request.args.get("worker", 3))


    k8s_cmd = k8s.kubernetes_control(user_name)
    #create ray head
    k8s_cmd.create_namespace()
    k8s_cmd.create_pod()
    k8s_cmd.create_service_dashboard()
    k8s_cmd.create_service_head_register()
    #create ray worker
    k8s_cmd.create_deployment(cpu=cpu_num, memory=memory_num, replicas_num=worker_num)
    k8s_cmd.create_ingress()

    return "ok"

import socket

# print(socket.gethostname())
app.run(host=socket.gethostname(), port=5000, debug=True)
