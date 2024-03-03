import { Routes, Route } from 'react-router-dom';
import ProtectRoute from './protect-route';
import Login from '../app/login/login';
import Dashboard from '../app/dashborad/dashboard';

/* eslint-disable-next-line */
export interface RootRouteProps { }

export function RootRoute(_props: RootRouteProps) {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route element={<ProtectRoute />}>
        <Route path="/dashboard" element={<Dashboard />} />
      </Route>
    </Routes>
  );
}

export default RootRoute;
