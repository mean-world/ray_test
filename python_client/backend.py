from flask import Flask, request

from kubernetes import client
#k8s
import k8s_object as k8s


app = Flask(__name__)
@app.route("/")
def hello():
    return "welcome!"

#submit ray job
@app.route("/submit_job")
def submit():
    namespace = request.args.get("namespace", "default")
    cmd = "RAY_ADDRESS=ray-head-svc:6379 ray job submit --working-dir /test/ -- python3 test.py"
    cmd = 'hello {name}, hello {text}'.format(name=name, text=text)
    os.system(cmd)

#check namespace exist
@app.route("/check_namespace")
def check():
    namespace = request.args.get("namespace", "default")
    v1 = client.CoreV1Api()
    
    try:
        result = v1.read_namespace(namespace)
        return "ok"
    except:
        return "not ok"

#create ray env
@app.route("/create_ray_env")
def creat_ray_env():
    user_name = request.args.get("username", "default")
    cpu_num = request.args.get("cpu", 2)
    memory_num = request.args.get("memory", 2)
    worker_num = request.args.get("worker", 3)


    k8s_cmd = k8s.kubernetes_control(user_name)
    #create ray head
    k8s_cmd.create_namespace()
    k8s_cmd.create_pod()
    k8s_cmd.create_service_dashboard()
    k8s_cmd.create_service_head_register()
    #create ray worker
    k8s_cmd.create_deployment(cpu=cpu_num, memory=memory_num, replicas_num=worker_num)
    # return "ok"
    return (user_name)+str(cpu_num)+str(memory_num)

import socket

print(socket.gethostname())
app.run(host=socket.gethostname(), port=5000)
