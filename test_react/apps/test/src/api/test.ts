import { apiClient } from './base';

export const setTraining = async (
  username: string,
  cpu: number | null,
  memory: number | null,
  worker: number | null
) => {
  return apiClient.get('/create_ray_env', {
    params: { username, cpu, memory, worker },
  });
};

export const checkNamespace = async (username: string) => {
  return apiClient.get('/check_namespace', {
    params: { username },
  });
};



export const submitJob_api = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filename: any,
  modelName: string,
  username: string,
  worker: number | null
) => {
  const formData = new FormData();
  formData.append('file', filename);
  console.log(formData);
  return apiClient.post('/submit_job', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    params: { modelName, username, worker },
  });
};
