/* eslint-disable react-hooks/rules-of-hooks */
import { UserOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Button, Card, Flex, Input, message } from 'antd';
import { SetStateAction, useEffect, useState } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';

const Login: React.FC = () => {
  const [namespace, setnamespace] = useState('');
  const onChange_namespace = (event: React.ChangeEvent<HTMLInputElement>) => {
    // console.log(event.target.value);
    setnamespace(event.target.value);
  };
  const navigate = useNavigate();

  return (
    <div style={{ height: '100%' }}>
      <Flex justify="center" align="ceter">
        <Card bordered={false} style={{ background: '#ADD8E6' }}>
          <h2>input username</h2>
          <Input
            size="large"
            placeholder="username"
            prefix={<UserOutlined />}
            maxLength={10}
            style={{ width: 150 }}
            value={namespace}
            onChange={onChange_namespace}
          />
          <br />
          <Button
            type="primary"
            // onClick={() =>  navigate('/', { state: { test: true } })}
            onClick={() =>
              navigate('/dashboard', {
                state: { namespace: namespace },
              })
            }
            style={{ width: 150, marginTop: 10 }}
          >
            submit
          </Button>
        </Card>
      </Flex>
    </div>
  );
};

export default Login;
