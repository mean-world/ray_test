import { useEffect, useState } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';

/* eslint-disable-next-line */
export interface ProtectRouteProps {}

export function ProtectRoute(_props: ProtectRouteProps) {
  const [isLogin, setIsLogin] = useState(false);
  const navigate = useNavigate();

  // useEffect(() => {
  //   if (!isLogin) navigate('/');
  // }, [isLogin, navigate]);

  return <Outlet />;
}

export default ProtectRoute;
