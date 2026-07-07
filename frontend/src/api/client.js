import axios from 'axios';

// When deployed behind NGINX proxy, the path is /ps2-api
// When running locally, it defaults to http://localhost:8002
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8002';

const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
