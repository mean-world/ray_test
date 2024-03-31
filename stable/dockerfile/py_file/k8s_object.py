
import datetime
from kubernetes import client, config
#get kubectl 
config.load_incluster_config()


#create ray heads

class kubernetes_control():
    def __init__(self, namespace="default", ray_image="pear1798/ray-test:cpuV4", driver_type="cpu"):
        self.namespace = namespace #user name
        self.image = ray_image
        self.driver_type = driver_type

    #create ray worker
    def create_deployment(self, cpu, memory, replicas_num):
        cpu_args = "--num-cpus=" + str(cpu)
        memory_args = "--memory=" + str(memory*1000000000)
        memory = str(memory) + "G"
        container = client.V1Container(
        name="ray-worker",
        image=self.image,
        resources=client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": memory},
            limits={"cpu": cpu, "memory": memory},
        ),
        command = ["ray"],
        args = ["start", "--address=ray-head-svc:6379", "--block", cpu_args, memory_args],
        env=[client.V1EnvVar(name="driver_type", value=self.driver_type)],
        )

        init_container = client.V1Container(
            name="test-head-status",
            image="busybox:1.28",
            command = ["sleep"],
            args = ["30"]
        )

        #set worker pod on gpu node & try one node one worker pod
        affinity = client.V1Affinity(
            pod_anti_affinity=client.V1PodAntiAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    client.V1WeightedPodAffinityTerm(
                        weight=100,
                        pod_affinity_term=client.V1LabelSelector(
                            match_expressions=[
                                client.V1LabelSelectorRequirement(
                                    key="type",
                                    operator="In",
                                    values=["worker"]
                                )
                            ]
                        )
                    )
                ]
            )
        )

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"type": "worker"}),
            spec=client.V1PodSpec(
                containers=[container], 
                init_containers=[init_container], 
                # node_selector={"type":"gpu"},
                # affinity=affinity
            ),
        )

        spec = client.V1DeploymentSpec(
            replicas=replicas_num, template=template, selector={
                "matchLabels":
                {"type": "worker"}})


        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name="ray-worker"),
            spec=spec,
        )

        apps_v1 = client.AppsV1Api()
        apps_v1.create_namespaced_deployment(body=deployment, namespace=self.namespace)

    #create ray head
    def create_pod(self):
        #container ray 
        container = client.V1Container(
            name="ray-head",
            image=self.image,
            ports=[client.V1ContainerPort(container_port=6379), client.V1ContainerPort(container_port=8265)],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "4", "memory": "4G"},
                limits={"cpu": "4", "memory": "4G"},
            ),
            env=[client.V1EnvVar(name="driver_type", value=self.driver_type)],
            command = ["ray"],
            args = ["start", "--block", "--head", "--dashboard-host=0.0.0.0", "--num-cpus=4", "--memory=4000000000"],
            volume_mounts=[client.V1VolumeMount(mount_path="/root/ray_results",name="cache-volume")],
        )

        #container ray 
        container_2 = client.V1Container(
            name="jupyter",
            working_dir="/root/ray_results",
            image="pear1798/jupyter_test:v2",
            ports=[client.V1ContainerPort(container_port=8888)],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "2", "memory": "2G"},
                limits={"cpu": "2", "memory": "2G"},
            ),
            env=[client.V1EnvVar(name="driver_type", value=self.driver_type), client.V1EnvVar(name="RAY_ADDRESS", value="127.0.0.1:6379")],
            command = ["jupyter"],
            args = ["lab", "--no-browser", "--port=8888", "--allow-root", "--autoreload", "--ip=0.0.0.0"],
            volume_mounts=[client.V1VolumeMount(mount_path="/root/ray_results",name="cache-volume")],
        )

        #spec
        spec = client.V1PodSpec(containers=[container, container_2], restart_policy="Always", volumes=[client.V1Volume(empty_dir=client.V1EmptyDirVolumeSource(),name="cache-volume")])

        #pod
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(name="ray-head", labels={"ray-type": "head"}),
            spec=spec,
        )

        v1 = client.CoreV1Api()
        v1.create_namespaced_pod(namespace=self.namespace, body=pod)

    #create ray service for ray head dashboard
    def create_service_dashboard(self):
        api_instance = client.CoreV1Api()

        service = client.V1Service()

        service.api_version = "v1"
        service.kind = "Service"
        service.metadata = client.V1ObjectMeta(name="ray-dashboard-svc")

        spec = client.V1ServiceSpec()
        spec.selector = {"ray-type": "head"}
        spec.ports = [client.V1ServicePort(protocol="TCP", port=8265, target_port=8265, name="dashboard")]
        service.spec = spec

        api_instance.create_namespaced_service(namespace=self.namespace, body=service)

    #create ray service to connect ray head
    def create_service_head_register(self):
        api_instance = client.CoreV1Api()

        service = client.V1Service()

        service.api_version = "v1"
        service.kind = "Service"
        service.metadata = client.V1ObjectMeta(name="ray-head-svc")

        spec = client.V1ServiceSpec()
        spec.selector = {"ray-type": "head"}
        spec.ports = [client.V1ServicePort(protocol="TCP", port=6379, target_port=6379, name="head")]
        service.spec = spec

        api_instance.create_namespaced_service(namespace=self.namespace, body=service)

    #create jupyter dashboard
    def create_service_jupyter_dashboard(self):
        api_instance = client.CoreV1Api()

        service = client.V1Service()

        service.api_version = "v1"
        service.kind = "Service"
        service.metadata = client.V1ObjectMeta(name="jupyter-dashboard-svc")

        spec = client.V1ServiceSpec()
        spec.selector = {"ray-type": "head"}
        spec.ports = [client.V1ServicePort(protocol="TCP", port=8888, target_port=8888, name="jupyter")]
        service.spec = spec

        api_instance.create_namespaced_service(namespace=self.namespace, body=service)

    #use user name to create namespace 
    def create_namespace(self):
        v1 = client.CoreV1Api()
        v1.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=self.namespace)))
        # namespace_body = client.V1Namespace(metadata=client.V1ObjectMeta(namespace=self.namespace))
        # v1 = client.CoreV1Api()
        # v1.create_namespace(body=namespace_body)

    #create ingress
    def create_ingress(self):
        host = self.namespace + "-dashboard.localdev.me"
        networking_v1_api = client.NetworkingV1Api()

        body = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(name="ray-ingress"),
            spec=client.V1IngressSpec(
                ingress_class_name="nginx",
                rules=[client.V1IngressRule(
                    host=host,
                    http=client.V1HTTPIngressRuleValue(
                        paths=[client.V1HTTPIngressPath(
                            path="/",
                            path_type="Prefix",
                            backend=client.V1IngressBackend(
                                service=client.V1IngressServiceBackend(
                                    port=client.V1ServiceBackendPort(
                                        number=8265,
                                    ),
                                    name="ray-dashboard-svc")
                                )
                            )
                        ]
                    )
                )]
            )
        )

        networking_v1_api.create_namespaced_ingress(
            namespace=self.namespace,
            body=body
        )

    #create jupyter ingress
    def create_jupyter_ingress(self):
        host = self.namespace + "-jupyter.localdev.me"
        networking_v1_api = client.NetworkingV1Api()

        body = client.V1Ingress(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(name="jupyter-ingress"),
            spec=client.V1IngressSpec(
                ingress_class_name="nginx",
                rules=[client.V1IngressRule(
                    host=host,
                    http=client.V1HTTPIngressRuleValue(
                        paths=[client.V1HTTPIngressPath(
                            path="/",
                            path_type="Prefix",
                            backend=client.V1IngressBackend(
                                service=client.V1IngressServiceBackend(
                                    port=client.V1ServiceBackendPort(
                                        number=8888,
                                    ),
                                    name="jupyter-dashboard-svc")
                                )
                            )
                        ]
                    )
                )]
            )
        )

        networking_v1_api.create_namespaced_ingress(
            namespace=self.namespace,
            body=body
        )

# k8s_cmd = kubernetes_control("AAAAA")
# #create ray head
# k8s_cmd.create_namespace()
# print("ok ns")
# k8s_cmd.create_pod()
# print("ok pod")
# k8s_cmd.create_service_dashboard()
# print("ok svc1")
# k8s_cmd.create_service_head_register()
# print("ok svc2")
# #create ray worker
# k8s_cmd.create_deployment(cpu=2, memory=2, replicas_num=3)
# print("ok deploy")
# k8s_cmd.create_ingress()
# print("ok ingress")


