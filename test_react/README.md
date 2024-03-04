# Build  Frontend Image 
To build the frontend image, follow the steps below:

1. If you have npm environment:
```
npm i
npm run container
```
2. If you don't have npm environment:
```
docker run  -it -v your_path/test_react:/test --rm node:20 bash
cd test
npm i
npm run container
# exit container 
exit
docker build -t 
```
