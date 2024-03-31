# Build  ray node & backend server Image 
## ray node image build
```
docker build -t pear1798/ray_node:cpuV4 -f Dockerfile .
```
## backend server
```
docker build -t pear1798/backend:stableCpuV7 -f backend_Dockerfile .
```
## jupyter container image
```
docker build -t pear1798/jupyter_test:v2 -f jupyter_Dockerfile .
```

# note 
### ray version 2.9.3

