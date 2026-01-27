import React, { useRef, useState, useCallback } from 'react';
import { validateVideoFile } from '../api/videoApi';
import { VIDEO_CONSTRAINTS } from '../api/config';
import './VideoTranslator.css';

/**
 * Video file with metadata for display
 * @typedef {Object} VideoFileItem
 * @property {string} id - Unique ID
 * @property {File} file - Original file
 * @property {string} name - File name
 * @property {number} size - Size in bytes
 * @property {string|null} previewUrl - Object URL for preview
 * @property {string|null} error - Validation error if any
 */

/**
 * VideoTranslator - Multi-file video uploader for sign language translation
 * 
 * Features:
 * - Drag & drop support
 * - Multiple file selection
 * - File validation (type, size)
 * - Video preview thumbnails
 * - File list management
 */
function VideoTranslator({ onInfer, loading = false, disabled = false }) {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);

  // Generate unique ID
  const generateId = () => `video-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  // Process files and validate
  const processFiles = useCallback((fileList) => {
    const newFiles = Array.from(fileList).map(file => {
      const validation = validateVideoFile(file);
      return {
        id: generateId(),
        file,
        name: file.name,
        size: file.size,
        previewUrl: validation.valid ? URL.createObjectURL(file) : null,
        error: validation.valid ? null : validation.error,
      };
    });
    
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  // Handle input change
  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
    // Reset input to allow selecting same file again
    e.target.value = '';
  };

  // Remove a file from list
  const removeFile = (id) => {
    setFiles(prev => {
      const file = prev.find(f => f.id === id);
      if (file?.previewUrl) {
        URL.revokeObjectURL(file.previewUrl);
      }
      return prev.filter(f => f.id !== id);
    });
  };

  // Clear all files
  const clearAll = () => {
    files.forEach(f => {
      if (f.previewUrl) URL.revokeObjectURL(f.previewUrl);
    });
    setFiles([]);
  };

  // Submit files for inference
  const handleSubmit = () => {
    const validFiles = files.filter(f => !f.error).map(f => f.file);
    if (validFiles.length > 0 && onInfer) {
      onInfer(validFiles);
    }
  };

  // Format file size
  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validCount = files.filter(f => !f.error).length;
  const errorCount = files.filter(f => f.error).length;

  return (
    <div className="video-translator">
      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={VIDEO_CONSTRAINTS.ALLOWED_MIME_TYPES.join(',')}
          onChange={handleInputChange}
          multiple
          className="file-input"
          disabled={disabled}
        />
        
        <div className="drop-zone-content">
          <div className="drop-icon">📹</div>
          <h3>Arrastra videos aquí</h3>
          <p>o haz clic para seleccionar</p>
          <span className="hint">
            Formatos: {VIDEO_CONSTRAINTS.ALLOWED_EXTENSIONS.join(', ')} • 
            Máx: {VIDEO_CONSTRAINTS.MAX_SIZE_MB}MB por archivo
          </span>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="file-list">
          <div className="file-list-header">
            <h4>
              <span className="file-count">{files.length}</span> video(s) seleccionado(s)
              {errorCount > 0 && (
                <span className="error-count">({errorCount} con error)</span>
              )}
            </h4>
            <button 
              className="clear-all-btn"
              onClick={clearAll}
              disabled={loading}
            >
              Limpiar todo
            </button>
          </div>

          <div className="file-items">
            {files.map(item => (
              <div 
                key={item.id} 
                className={`file-item ${item.error ? 'has-error' : ''}`}
              >
                {/* Thumbnail */}
                <div className="file-thumbnail">
                  {item.previewUrl ? (
                    <video src={item.previewUrl} muted />
                  ) : (
                    <div className="no-preview">⚠️</div>
                  )}
                </div>

                {/* Info */}
                <div className="file-info">
                  <span className="file-name" title={item.name}>
                    {item.name}
                  </span>
                  <span className="file-size">{formatSize(item.size)}</span>
                  {item.error && (
                    <span className="file-error">{item.error}</span>
                  )}
                </div>

                {/* Remove button */}
                <button
                  className="remove-btn"
                  onClick={() => removeFile(item.id)}
                  disabled={loading}
                  title="Eliminar"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {files.length > 0 && (
        <div className="translator-actions">
          <button
            className="translate-btn"
            onClick={handleSubmit}
            disabled={loading || validCount === 0}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Procesando...
              </>
            ) : (
              <>
                <span className="btn-icon">🔮</span>
                Traducir {validCount} video{validCount !== 1 ? 's' : ''}
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export default VideoTranslator;
