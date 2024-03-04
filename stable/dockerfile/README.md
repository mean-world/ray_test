# Build  ray node & backend server Image 
## ray node image build
```
docker build -t pear1798/ray_node:cpuV3 -f Dockerfile .
```
## backend server
```
docker build -t pear1798/backend:your_test8 -f backend_Dockerfile .
```
