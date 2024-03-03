from flask import Flask, request

#k8s
import k8s_object as k8s


app = Flask(__name__)
@app.route("/")
def hello():
    return "welcome!"

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