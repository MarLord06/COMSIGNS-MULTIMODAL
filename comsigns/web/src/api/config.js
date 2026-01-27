/**
 * API configuration for ComSigns frontend.
 * Centralized configuration for all backend endpoints.
 */

// Base API URL - change for production
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
  // Video inference
  VIDEO_INFER: '/api/video/infer',
  VIDEO_INFO: '/api/video/info',
  VIDEO_CONFIG: '/api/video/config',
  
  // PKL inference
  PKL_INFER: '/infer',
  PKL_EVALUATE: '/infer/evaluate',
  PKL_BATCH_EVALUATE: '/infer/batch/evaluate',
  
  // Batch inference with sequence
  BATCH_INFER: '/api/inference/batch',
  SEQUENCE: '/api/inference/sequence',
  SEQUENCE_RESET: '/api/inference/sequence/reset',
  
  // System
  HEALTH: '/health',
  INFO: '/info',
};

// Video constraints (should match backend)
export const VIDEO_CONSTRAINTS = {
  MAX_SIZE_MB: 100,
  MIN_DURATION_SEC: 0.1,
  MAX_DURATION_SEC: 30,
  ALLOWED_EXTENSIONS: ['.mp4', '.mov', '.avi', '.webm', '.mkv'],
  ALLOWED_MIME_TYPES: [
    'video/mp4',
    'video/quicktime',
    'video/x-msvideo',
    'video/webm',
    'video/x-matroska',
  ],
};

/**
 * Build full API URL
 * @param {string} endpoint - Endpoint path
 * @returns {string} Full URL
 */
export function buildUrl(endpoint) {
  return `${API_BASE_URL}${endpoint}`;
}

export default {
  API_BASE_URL,
  API_ENDPOINTS,
  VIDEO_CONSTRAINTS,
  buildUrl,
};
