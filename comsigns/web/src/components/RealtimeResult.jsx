import React from 'react'
import './RealtimeResult.css'

function RealtimeResult({ prediction }) {
  if (!prediction) return null

  return (
    <div className="realtime-result">
      {/* Glosa Principal - Destacada */}
      <div className="gloss-display">
        <div className="gloss-label">Seña Detectada</div>
        <div className="gloss-value">
          {prediction.gloss || 'PROCESANDO...'}
        </div>
        <div className="confidence-container">
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ width: `${(prediction.confidence || 0) * 100}%` }}
            />
          </div>
          <span className="confidence-text">
            {((prediction.confidence || 0) * 100).toFixed(0)}% confianza
          </span>
        </div>
      </div>

      {/* Traducción */}
      <div className="translation-display">
        <div className="translation-label">
          <span className="icon">💬</span>
          Traducción
        </div>
        <div className="translation-value">
          {prediction.text || 'Esperando...'}
        </div>
      </div>

      {/* Texto Acumulado - Conversación */}
      <div className="conversation-display">
        <div className="conversation-label">
          <span className="icon">📝</span>
          Conversación Completa
        </div>
        <div className="conversation-text">
          {prediction.accumulated_text || 'Comienza a hacer señas para iniciar la conversación...'}
        </div>
      </div>

      {/* Estadísticas */}
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <div className="stat-label">Latencia</div>
            <div className="stat-value">{prediction.processing_time_ms || 0}ms</div>
          </div>
        </div>
        
        <div className="stat-item">
          <div className="stat-icon">🎞️</div>
          <div className="stat-content">
            <div className="stat-label">Buffer</div>
            <div className="stat-value">{prediction.frames_in_buffer || 0} frames</div>
          </div>
        </div>
        
        <div className="stat-item">
          <div className="stat-icon">#</div>
          <div className="stat-content">
            <div className="stat-label">Secuencia</div>
            <div className="stat-value">#{prediction.sequence || 0}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RealtimeResult
