import React, { useRef, useState } from 'react'
import './VideoUploader.css'

function VideoUploader({ onUpload, loading }) {
  const fileInputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file)
    } else {
      alert('Por favor selecciona un archivo de video válido')
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0])
    }
  }

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="video-uploader">
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleInputChange}
          className="file-input"
          id="video-input"
        />
        <label htmlFor="video-input" className="upload-label">
          {selectedFile ? (
            <>
              <span className="file-icon">📹</span>
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </>
          ) : (
            <>
              <span className="upload-icon">📤</span>
              <p className="upload-text">
                Arrastra un video aquí o haz clic para seleccionar
              </p>
              <p className="upload-hint">
                Formatos soportados: MP4, AVI, MOV, MKV
              </p>
            </>
          )}
        </label>
      </div>

      {selectedFile && (
        <div className="upload-actions">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? 'Procesando...' : 'Procesar Video'}
          </button>
          <button
            onClick={handleClear}
            disabled={loading}
            className="btn btn-secondary"
          >
            Limpiar
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Procesando video y extrayendo keypoints...</p>
        </div>
      )}
    </div>
  )
}

export default VideoUploader

