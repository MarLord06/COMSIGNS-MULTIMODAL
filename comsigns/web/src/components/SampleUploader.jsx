import React, { useRef, useState } from 'react'
import './SampleUploader.css'

/**
 * Component for uploading .pkl sample files for inference.
 */
function SampleUploader({ onUpload, loading }) {
  const fileInputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [topk, setTopk] = useState(5)

  const handleFileSelect = (file) => {
    if (file && file.name.endsWith('.pkl')) {
      setSelectedFile(file)
    } else {
      alert('Por favor selecciona un archivo .pkl válido')
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
      onUpload(selectedFile, topk)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="sample-uploader">
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
          accept=".pkl"
          onChange={handleInputChange}
          className="file-input"
          id="sample-input"
        />
        <label htmlFor="sample-input" className="upload-label">
          {selectedFile ? (
            <>
              <span className="file-icon">📦</span>
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </>
          ) : (
            <>
              <span className="upload-icon">📤</span>
              <p className="upload-text">
                Arrastra un sample aquí o haz clic para seleccionar
              </p>
              <p className="upload-hint">
                Formato soportado: .pkl (features extraídos)
              </p>
            </>
          )}
        </label>
      </div>

      {selectedFile && (
        <>
          <div className="topk-selector">
            <label htmlFor="topk">Top-K predicciones:</label>
            <select 
              id="topk" 
              value={topk} 
              onChange={(e) => setTopk(Number(e.target.value))}
            >
              <option value={3}>Top 3</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
            </select>
          </div>

          <div className="upload-actions">
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <span className="spinner-small"></span>
                  Procesando...
                </>
              ) : (
                '🔍 Inferir Seña'
              )}
            </button>
            <button
              onClick={handleClear}
              disabled={loading}
              className="btn btn-secondary"
            >
              Limpiar
            </button>
          </div>
        </>
      )}

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Ejecutando inferencia...</p>
        </div>
      )}
    </div>
  )
}

export default SampleUploader
