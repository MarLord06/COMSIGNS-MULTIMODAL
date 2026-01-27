/**
 * Video prediction API client.
 * Handles communication with the video inference backend.
 */

import { API_BASE_URL, API_ENDPOINTS, VIDEO_CONSTRAINTS, buildUrl } from './config';

/**
 * @typedef {Object} VideoPrediction
 * @property {string} video - Original video filename
 * @property {number} class_id - Model output class ID
 * @property {string} class_name - Internal class name (e.g., HEAD_259)
 * @property {string} gloss - Human-readable sign gloss
 * @property {number} score - Confidence score (0-1)
 * @property {boolean} accepted - Whether prediction was accepted
 * @property {string} reason - Reason for acceptance/rejection
 */

/**
 * @typedef {Object} VideoInferenceResponse
 * @property {VideoPrediction[]} results - Per-video prediction results
 * @property {Object[]} errors - Per-video errors
 */

/**
 * @typedef {Object} VideoInfo
 * @property {string} filename
 * @property {number} fps
 * @property {number} frame_count
 * @property {number} width
 * @property {number} height
 * @property {number} duration_sec
 */

/**
 * Validate video file before upload
 * @param {File} file - File to validate
 * @returns {{valid: boolean, error?: string}}
 */
export function validateVideoFile(file) {
  // Check MIME type
  if (!VIDEO_CONSTRAINTS.ALLOWED_MIME_TYPES.includes(file.type)) {
    const ext = file.name.split('.').pop()?.toLowerCase();
    const validExt = VIDEO_CONSTRAINTS.ALLOWED_EXTENSIONS.some(e => 
      e.toLowerCase() === `.${ext}`
    );
    
    if (!validExt) {
      return {
        valid: false,
        error: `Tipo de archivo no soportado: ${file.type || ext}. Formatos permitidos: ${VIDEO_CONSTRAINTS.ALLOWED_EXTENSIONS.join(', ')}`
      };
    }
  }
  
  // Check file size
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > VIDEO_CONSTRAINTS.MAX_SIZE_MB) {
    return {
      valid: false,
      error: `Archivo muy grande: ${sizeMB.toFixed(1)}MB (máximo: ${VIDEO_CONSTRAINTS.MAX_SIZE_MB}MB)`
    };
  }
  
  return { valid: true };
}

/**
 * Run inference on one or more video files
 * @param {File[]} files - Array of video files
 * @param {Object} options - Options
 * @param {number} [options.topk=5] - Number of top predictions
 * @param {AbortSignal} [options.signal] - Abort signal
 * @returns {Promise<VideoInferenceResponse>}
 */
export async function inferFromVideos(files, options = {}) {
  const { topk = 5, signal } = options;
  
  if (!files || files.length === 0) {
    throw new Error('No se proporcionaron archivos de video');
  }
  
  // Validate all files
  for (const file of files) {
    const validation = validateVideoFile(file);
    if (!validation.valid) {
      throw new Error(`${file.name}: ${validation.error}`);
    }
  }
  
  // Build form data
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  
  // Make request
  const url = `${buildUrl(API_ENDPOINTS.VIDEO_INFER)}?topk=${topk}`;
  
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    signal,
  });
  
  if (!response.ok) {
    let errorMessage = 'Error en la inferencia de video';
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      errorMessage = `Error ${response.status}: ${response.statusText}`;
    }
    throw new Error(errorMessage);
  }
  
  return response.json();
}

/**
 * Get video metadata without running inference
 * @param {File[]} files - Array of video files
 * @returns {Promise<VideoInfo[]>}
 */
export async function getVideoInfo(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  
  const response = await fetch(buildUrl(API_ENDPOINTS.VIDEO_INFO), {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Error al obtener información del video');
  }
  
  return response.json();
}

/**
 * Get video processing configuration from backend
 * @returns {Promise<Object>}
 */
export async function getVideoConfig() {
  const response = await fetch(buildUrl(API_ENDPOINTS.VIDEO_CONFIG));
  
  if (!response.ok) {
    throw new Error('Error al obtener configuración');
  }
  
  return response.json();
}

/**
 * Check backend health
 * @returns {Promise<{status: string, model_loaded: boolean, num_classes?: number}>}
 */
export async function checkHealth() {
  const response = await fetch(buildUrl(API_ENDPOINTS.HEALTH));
  return response.json();
}

export default {
  validateVideoFile,
  inferFromVideos,
  getVideoInfo,
  getVideoConfig,
  checkHealth,
};
