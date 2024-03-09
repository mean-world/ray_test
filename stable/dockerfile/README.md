# Build  ray node & backend server Image 
## ray node image build
```
docker build -t pear1798/ray_node:cpuV4 -f Dockerfile .
```
## backend server
```
docker build -t pear1798/backend:stableCpuV2 -f backend_Dockerfile .
```
# note 
### ray version 2.9.3

