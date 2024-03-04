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

Describe any prerequisites or setup required before running the project.

## How to Run

Provide instructions on how to run the project locally or deploy it on a Kubernetes cluster.

## Execution Results

Describe the expected outcomes or any important details about the execution of the project.

## Demo Video

Link to a video demonstrating the project in action.

## Future Directions

Discuss potential future enhancements or directions for the project.## Note

Please note that the current setup is for CPU testing. Future iterations of the project will include support for GPU-based distributed training.

