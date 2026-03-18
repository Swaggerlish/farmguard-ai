import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
});

export const predictDisease = async (file, language = "english") => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post(`/predict?language=${language}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};