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

1. [backend server & ray node image](./stable/dockerfile) 
2. [frontend image](./test_react)


Once you have installed these components, you can proceed with running the project on your Kubernetes cluster.

## How to Run

To run the project, follow the steps below:

### 1. Deploy Frontend
Run the following command to deploy the frontend:
```
kubectl apply -f stable/k8s_yaml/frontend.yaml
```

### 2. Deploy Backend Server
Use the following command to deploy the backend server:
```
kubectl apply -f stable/k8s_yaml/backend.yaml
```

### 3. Deploy MLflow Server
Deploy the MLflow server using the command:
```
kubectl apply -f stable/k8s_yaml/mlflow.yaml
```
### 4. Start Port-Forwarding
tart port-forwarding to access the application:
```
kubectl port-forward --namespace=ingress-nginx service/ingress-nginx-controller 8080:80
```

### 5. Connect to Frontend Webpage
To connect to the frontend webpage, use the following URL:
```
http://third-party-platform.localdev.me:8080
```

## Execution Results

Upon logging in, users are directed to the dashboard interface, where they can access different functionalities based on their user status:

### Login Interface
- **Functionality:** Users can enter their username and click the login button to access the dashboard page.
- **Image:** Image A depicts the login interface.
# <p align="center">Image A: Login Interface</p>
![Login Interface](https://github.com/mean-world/ray_test/assets/87417974/569c0654-b621-4dc1-80e1-8bfc4a0ac2a6)

### Dashboard Interface
- **Functionality:** The dashboard is divided into two sections, displaying different content based on the user's status:
- **New User:** The left section displays the "Create Distributed Training Environment" feature. This feature allows creating a Ray distributed environment in Kubernetes cluster specifically for the new user.
- **Existing User:** The right section displays the "Submit Training" feature and opens the Ray Dashboard (personal use) and MLflow Dashboard (shared). Users can choose to use default provided models or upload their own model data for training submission. The submitted training will be sent to Ray, and upon completion, the model and hyperparameters will be sent to MLflow for storage.
- **Images:** 
  - Image B shows the "Create Distributed Training Environment" feature on the left side of the dashboard interface.
  - Image C shows the "Submit Training" feature on the right side of the dashboard interface.

# <p align="center">Image B: Dashboard Interface - Left Section</p>
![Dashboard Interface - Left Section](https://github.com/mean-world/ray_test/assets/87417974/1ca4178a-3ac6-40bc-b7bf-480e685a6bcf)

# <p align="center">Image C: Dashboard Interface - Right Section</p>
![Dashboard Interface - Right Section](https://github.com/mean-world/ray_test/assets/87417974/11509c55-7368-4b8b-a1d1-93ad09d48023)


## Demo Video

Link to a video demonstrating the project in action.

## Future Directions

Discuss potential future enhancements or directions for the project.## Note

Please note that the current setup is for CPU testing. Future iterations of the project will include support for GPU-based distributed training.

