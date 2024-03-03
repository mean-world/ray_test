import cookieParser from 'cookie-parser';
import cors from 'cors';
import express from 'express';
import logger from 'morgan';
import helmet from 'helmet';
import mustache from 'mustache-express';
import * as path from 'path';

import { app } from './app';
import { routers } from './routers';

app.disable('x-powered-by');

app.use(logger('dev'));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cookieParser(process.env.SECRET_COOKIE));
app.use(cors());

// app.use(helmet());
// app.use(
//   helmet.contentSecurityPolicy({
//     directives: {
//       imgSrc: ["'self'", 'blob:', 'data:'],
//     },
//   })
// );
// app.use(helmet.crossOriginResourcePolicy({ policy: 'cross-origin' }));
// app.use(helmet.crossOriginEmbedderPolicy({ policy: 'require-corp' }));

// view engine setup
app.engine('html', mustache());
app.set('view engine', 'html');
app.set('views', path.join(__dirname, 'app'));

app.use(express.static(path.join(__dirname, 'app')));
app.use(express.static(path.join(__dirname, 'assets')));

// mount all routers
routers.keys().forEach((key) => {
  const router = routers(key).default;
  app.use(router());
});

app.all('*', (req, res) => {
  res.render('index');
});

const port = process.env.PORT || 3333;
const server = app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Listening at http://localhost:${port}/api`);
});

// eslint-disable-next-line no-console
server.on('error', console.error);
