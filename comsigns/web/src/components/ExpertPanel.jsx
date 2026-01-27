import React, { useRef, useState, useEffect } from 'react'
import './ExpertPanel.css'

/**
 * Expert System Panel for sign language inference.
 * 
 * This component provides:
 * - Single or batch file input (multiple .pkl files)
 * - Per-file prediction display with acceptance decision
 * - Accumulated sequence visualization
 * - Sequence reset functionality
 * 
 * The frontend is purely passive - all decisions come from the backend.
 */

const API_URL = 'http://localhost:8000'

function ExpertPanel() {
  // State
  const [selectedFiles, setSelectedFiles] = useState([])  // Now supports multiple files
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [currentPrediction, setCurrentPrediction] = useState(null)  // Single file result
  const [batchResults, setBatchResults] = useState(null)  // Batch results
  const [sequence, setSequence] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  
  const fileInputRef = useRef(null)

  // Load initial sequence state on mount
  useEffect(() => {
    fetchSequence()
  }, [])

  // ============================================================
  // API Calls
  // ============================================================

  const fetchSequence = async () => {
    try {
      const response = await fetch(`${API_URL}/sequence`)
      if (response.ok) {
        const data = await response.json()
        setSequence(data.sequence)
      }
    } catch (err) {
      console.error('Failed to fetch sequence:', err)
    }
  }

  const submitForInference = async () => {
    if (selectedFiles.length === 0) return

    setLoading(true)
    setError(null)
    setBatchResults(null)
    setCurrentPrediction(null)

    try {
      const formData = new FormData()
      
      // Single file vs batch - use different endpoints
      if (selectedFiles.length === 1) {
        // Single file: use /infer/evaluate endpoint
        formData.append('file', selectedFiles[0])

        const response = await fetch(`${API_URL}/infer/evaluate`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Error en la inferencia')
        }

        const data = await response.json()
        
        // Update state from backend response
        setCurrentPrediction(data.prediction)
        setSequence(data.sequence)
        
      } else {
        // Multiple files: use /infer/batch/evaluate endpoint
        selectedFiles.forEach(file => {
          formData.append('files', file)
        })

        const response = await fetch(`${API_URL}/infer/batch/evaluate`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Error en la inferencia batch')
        }

        const data = await response.json()
        
        // Update state from backend response
        setBatchResults(data)
        setSequence(data.sequence)
      }
      
      // Clear selected files after successful submission
      setSelectedFiles([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

    } catch (err) {
      setError(err.message || 'Error de conexión con el servidor')
      console.error('Inference error:', err)
    } finally {
      setLoading(false)
    }
  }

  const resetSequence = async () => {
    try {
      const response = await fetch(`${API_URL}/sequence/reset`, {
        method: 'POST',
      })

      if (response.ok) {
        const data = await response.json()
        setSequence(data.sequence)
        setCurrentPrediction(null)
        setBatchResults(null)
        setError(null)
      }
    } catch (err) {
      setError('Error resetting sequence')
      console.error('Reset error:', err)
    }
  }

  // ============================================================
  // File Handling
  // ============================================================

  const handleFileSelect = (files) => {
    // Filter to only .pkl files
    const validFiles = Array.from(files).filter(file => file.name.endsWith('.pkl'))
    
    if (validFiles.length === 0) {
      setError('Por favor selecciona archivos .pkl válidos')
      return
    }
    
    if (validFiles.length < files.length) {
      // Some files were filtered out
      setError(`${files.length - validFiles.length} archivo(s) ignorados (solo .pkl permitido)`)
    } else {
      setError(null)
    }
    
    setSelectedFiles(validFiles)
  }

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const clearFiles = () => {
    setSelectedFiles([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
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
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files)
    }
  }

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files)
    }
  }

  // ============================================================
  // Helpers
  // ============================================================

  const getBucketClass = (bucket) => {
    switch (bucket) {
      case 'HEAD': return 'bucket-head'
      case 'MID': return 'bucket-mid'
      case 'OTHER': return 'bucket-other'
      default: return ''
    }
  }

  const formatConfidence = (conf) => {
    return `${(conf * 100).toFixed(1)}%`
  }

  // ============================================================
  // Render
  // ============================================================

  return (
    <div className="expert-panel">
      {/* Section A: Video Input Panel */}
      <section className="panel video-input-panel">
        <div className="panel-header">
          <h2>📹 Video Input</h2>
          <p>Sube uno o varios samples de señas (.pkl) para evaluación</p>
        </div>

        <div
          className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFiles.length > 0 ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pkl"
            multiple
            onChange={handleInputChange}
            className="file-input"
            id="expert-file-input"
          />
          <label htmlFor="expert-file-input" className="drop-zone-label">
            {selectedFiles.length > 0 ? (
              <div className="files-selected">
                <span className="file-icon">📦</span>
                <span className="files-count">
                  {selectedFiles.length} archivo{selectedFiles.length > 1 ? 's' : ''} seleccionado{selectedFiles.length > 1 ? 's' : ''}
                </span>
              </div>
            ) : (
              <div className="drop-zone-content">
                <span className="upload-icon">📤</span>
                <span className="drop-text">
                  Arrastra archivos aquí o haz clic para seleccionar
                </span>
                <span className="drop-hint">Formato: .pkl (uno o múltiples archivos)</span>
              </div>
            )}
          </label>
        </div>

        {/* Selected Files List */}
        {selectedFiles.length > 0 && (
          <div className="files-list">
            <div className="files-list-header">
              <span className="files-list-title">Archivos seleccionados:</span>
              <button onClick={clearFiles} className="btn-clear-files">
                Limpiar todo
              </button>
            </div>
            <div className="files-list-items">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-item-icon">📄</span>
                  <span className="file-item-name">{file.name}</span>
                  <span className="file-item-size">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                  <button 
                    onClick={() => removeFile(index)} 
                    className="file-item-remove"
                    title="Eliminar"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="input-actions">
          <button
            onClick={submitForInference}
            disabled={selectedFiles.length === 0 || loading}
            className="btn btn-primary btn-infer"
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Procesando{selectedFiles.length > 1 ? ` ${selectedFiles.length} archivos` : ''}...
              </>
            ) : (
              <>
                <span className="btn-icon">🔍</span>
                {selectedFiles.length > 1 
                  ? `Inferir ${selectedFiles.length} archivos` 
                  : 'Enviar para Inferencia'}
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
      </section>

      {/* Section B: Results Panel (Single or Batch) */}
      <section className="panel prediction-panel">
        <div className="panel-header">
          <h2>🎯 {batchResults ? 'Resultados del Batch' : 'Predicción Actual'}</h2>
          <p>{batchResults ? 'Resultados por archivo' : 'Resultado del modelo + decisión del experto'}</p>
        </div>

        {batchResults ? (
          // Batch Results Display
          <div className="batch-results">
            {/* Summary */}
            <div className="batch-summary">
              <div className="summary-stat">
                <span className="stat-value">{batchResults.summary.processed}</span>
                <span className="stat-label">Procesados</span>
              </div>
              <div className="summary-stat accepted">
                <span className="stat-value">{batchResults.summary.accepted}</span>
                <span className="stat-label">Aceptados</span>
              </div>
              <div className="summary-stat rejected">
                <span className="stat-value">{batchResults.summary.rejected}</span>
                <span className="stat-label">Rechazados</span>
              </div>
              {batchResults.summary.failed > 0 && (
                <div className="summary-stat failed">
                  <span className="stat-value">{batchResults.summary.failed}</span>
                  <span className="stat-label">Errores</span>
                </div>
              )}
            </div>

            {/* Per-file Results */}
            <div className="batch-results-list">
              {batchResults.results.map((item, index) => (
                <div 
                  key={index} 
                  className={`batch-result-card ${item.prediction.accepted ? 'accepted' : 'rejected'}`}
                >
                  <div className="result-card-header">
                    <span className="result-file-name">{item.file_name}</span>
                    <span className={`result-decision-badge ${item.prediction.accepted ? 'accepted' : 'rejected'}`}>
                      {item.prediction.accepted ? '✓ Aceptado' : '✗ Rechazado'}
                    </span>
                  </div>
                  <div className="result-card-body">
                    <div className="result-gloss">{item.prediction.gloss}</div>
                    <div className="result-meta">
                      <span className="result-confidence">
                        {formatConfidence(item.prediction.confidence)}
                      </span>
                      <span className={`result-bucket ${getBucketClass(item.prediction.bucket)}`}>
                        {item.prediction.bucket}
                      </span>
                    </div>
                    <div className="result-reason">{item.prediction.reason}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Batch Errors */}
            {batchResults.errors && batchResults.errors.length > 0 && (
              <div className="batch-errors">
                <h4>⚠️ Errores:</h4>
                {batchResults.errors.map((err, index) => (
                  <div key={index} className="batch-error-item">
                    <span className="error-file">{err.file_name}</span>
                    <span className="error-msg">{err.error}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : currentPrediction ? (
          <div className="prediction-content">
            {/* Decision Badge */}
            <div className={`decision-badge ${currentPrediction.accepted ? 'accepted' : 'rejected'}`}>
              <span className="decision-icon">
                {currentPrediction.accepted ? '✓' : '✗'}
              </span>
              <span className="decision-text">
                {currentPrediction.accepted ? 'ACEPTADO' : 'RECHAZADO'}
              </span>
            </div>

            {/* Main Prediction */}
            <div className="prediction-main">
              <div className="gloss-display">
                <span className="gloss-label">Gloss Predicho</span>
                <span className="gloss-value">{currentPrediction.gloss}</span>
              </div>

              {/* Confidence Bar */}
              <div className="confidence-section">
                <div className="confidence-header">
                  <span className="confidence-label">Confianza</span>
                  <span className="confidence-value">
                    {formatConfidence(currentPrediction.confidence)}
                  </span>
                </div>
                <div className="confidence-bar-track">
                  <div 
                    className="confidence-bar-fill"
                    style={{ width: `${currentPrediction.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Bucket */}
              <div className="bucket-section">
                <span className="bucket-label">Bucket</span>
                <span className={`bucket-tag ${getBucketClass(currentPrediction.bucket)}`}>
                  {currentPrediction.bucket}
                </span>
              </div>
            </div>

            {/* Reason */}
            <div className="reason-section">
              <span className="reason-label">Razón</span>
              <p className="reason-text">{currentPrediction.reason}</p>
            </div>

            {/* Rule Applied */}
            {currentPrediction.rule_applied && (
              <div className="rule-section">
                <span className="rule-label">Regla Aplicada</span>
                <code className="rule-code">{currentPrediction.rule_applied}</code>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-prediction">
            <span className="empty-icon">🤔</span>
            <p>Aún no hay predicción</p>
            <p className="empty-hint">Sube un sample para comenzar</p>
          </div>
        )}
      </section>

      {/* Section C: Sequence Panel */}
      <section className="panel sequence-panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h2>📝 Secuencia Aceptada</h2>
            <p>Palabras validadas por el sistema experto</p>
          </div>
          <div className="sequence-stats">
            <span className="stat-badge">
              {sequence?.length || 0} palabras
            </span>
          </div>
        </div>

        {sequence && sequence.length > 0 ? (
          <div className="sequence-content">
            {/* Sequence Flow */}
            <div className="sequence-flow">
              {sequence.accepted.map((item, index) => (
                <React.Fragment key={index}>
                  <div className="sequence-item">
                    <div className="item-header">
                      <span className={`item-bucket ${getBucketClass(item.bucket)}`}>
                        {item.bucket}
                      </span>
                      <span className="item-position">#{item.position + 1}</span>
                    </div>
                    <span className="item-gloss">{item.gloss}</span>
                    <span className="item-confidence">
                      {formatConfidence(item.confidence)}
                    </span>
                  </div>
                  {index < sequence.accepted.length - 1 && (
                    <span className="sequence-arrow">→</span>
                  )}
                </React.Fragment>
              ))}
            </div>

            {/* Glosses Summary */}
            <div className="glosses-summary">
              <span className="summary-label">Secuencia:</span>
              <span className="summary-text">
                {sequence.glosses.join(' → ')}
              </span>
            </div>
          </div>
        ) : (
          <div className="empty-sequence">
            <span className="empty-icon">📭</span>
            <p>La secuencia está vacía</p>
            <p className="empty-hint">Las palabras aceptadas aparecerán aquí</p>
          </div>
        )}

        {/* Rejected Items (Collapsible) */}
        {sequence?.rejected && sequence.rejected.length > 0 && (
          <details className="rejected-section">
            <summary className="rejected-summary">
              <span className="rejected-icon">🚫</span>
              {sequence.rejected.length} predicciones rechazadas
            </summary>
            <div className="rejected-list">
              {sequence.rejected.map((item, index) => (
                <div key={index} className="rejected-item">
                  <span className="rejected-gloss">{item.gloss}</span>
                  <span className="rejected-reason">{item.reason}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </section>

      {/* Section D: Controls */}
      <section className="panel controls-panel">
        <button
          onClick={resetSequence}
          className="btn btn-secondary btn-reset"
          disabled={loading}
        >
          <span className="btn-icon">🔄</span>
          Reiniciar Secuencia
        </button>
      </section>
    </div>
  )
}

export default ExpertPanel
