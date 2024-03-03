import axios from 'axios';
import { API } from '../../src/constants/com-const';

export const apiClient = axios.create({
  baseURL: API.baseURL,
  // timeout: 10000, // 10 seconds
});
