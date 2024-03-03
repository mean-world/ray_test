FROM node:20-alpine3.19 as builder

WORKDIR /app
COPY ./ ./

RUN npm install
RUN npm run build
#/app/dist

FROM nginx:stable
WORKDIR /app
COPY --from=builder /app/dist/* /usr/share/nginx/html 
COPY ./server.nginx.conf /etc/nginx/conf.d/default.conf
