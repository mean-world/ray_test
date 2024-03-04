# Distributed Training Platform on Kubernetes Cluster

This project aims to establish a distributed training platform running on a Kubernetes cluster, utilizing the technology stack of Ray, MLflow, React, and Flask to implement an end-to-end machine learning workflow.

## Technology Components

- **Ray:** Used for distributed training of PyTorch models, achieving efficient utilization of computational resources.
- **MLflow:** Employed for model saving and hyperparameter logging, providing model management and experiment tracking functionalities.
- **React:** Utilized for building a user-friendly frontend web interface, offering an intuitive user experience.
- **Flask:** Utilized for constructing the backend API, handling user requests and communicating with MLflow and Ray.

## Functionality

This project will allow users to submit training jobs through a simple frontend interface, with the system automatically allocating resources on the Kubernetes cluster, utilizing Ray for distributed training, and leveraging MLflow for experiment tracking and model saving. Through the user interface provided by React, users can monitor training progress, view experiment results.

## Purpose

This project aims to provide a comprehensive, scalable machine learning development and deployment solution, enabling users to conduct model training and management more easily.
## Table of Contents

1. [Preparation](#preparation)
2. [How to Run](#how-to-run)
3. [Execution Results](#execution-results)
4. [Demo Video](#demo-video)
5. [Future Directions](#future-directions)

## Preparation

Before running the project, ensure you have the necessary dependencies and Kubernetes components installed. Follow the steps below for installation:

### 1. Install kubeadm(Version 1.29.0-1.1)

Refer to the official Kubernetes documentation for detailed instructions on installing kubeadm:

[Install kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)

### 2. Install cri-o(Version 1.29.0)

Follow the installation guide provided by the cri-o project on GitHub:

[Install cri-o](https://github.com/cri-o/cri-o)

### 3. Install CNI (Calico)

Deploy Calico for networking by applying the manifest:

```
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.2/manifests/calico.yaml
```

### 4. Install Ingress Controller (nginx-ingress)

Deploy nginx-ingress controller using the provided manifest:

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/cloud/deploy.yaml
```

### 5. Rebuilding Images

[backend server & ray node image](./stable/dockerfile)
[frontend image](./test_react)


Once you have installed these components, you can proceed with running the project on your Kubernetes cluster.

## How to Run

Provide instructions on how to run the project locally or deploy it on a Kubernetes cluster.

## Execution Results

Describe the expected outcomes or any important details about the execution of the project.

## Demo Video

Link to a video demonstrating the project in action.

## Future Directions

Discuss potential future enhancements or directions for the project.## Note

Please note that the current setup is for CPU testing. Future iterations of the project will include support for GPU-based distributed training.

