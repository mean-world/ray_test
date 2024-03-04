# Build  Frontend Image 
To build the frontend image, follow the steps below:

1. If you have npm environment:
```
npm i
npm run container
```
2. If you don't have npm environment:
```
＃ Start a Docker container with Node.js environment:
docker run  -it -v your_path/test_react:/test --rm node:20 bash
cd test
＃ Install dependencies and build the container
npm i
npm run container
# Exit the container 
exit
docker build -t pear1798/test_react:stable .
```
