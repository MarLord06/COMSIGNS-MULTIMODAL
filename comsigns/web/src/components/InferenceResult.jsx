import React from 'react'
import './InferenceResult.css'

function InferenceResult({ result }) {
  if (!result) return null

  return (
    <div className="inference-result">
      <h2>Resultados de Inferencia</h2>

      <div className="result-section">
        <h3>Información del Clip</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">ID del Clip:</span>
            <span className="info-value">{result.clip_id}</span>
          </div>
          <div className="info-item">
            <span className="info-label">FPS:</span>
            <span className="info-value">{result.fps}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Número de Frames:</span>
            <span className="info-value">{result.num_frames}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Forma del Embedding:</span>
            <span className="info-value">{result.embedding_shape.join(' × ')}</span>
          </div>
        </div>
      </div>

      {result.metadata && (
        <div className="result-section">
          <h3>Metadatos</h3>
          <div className="info-grid">
            {result.metadata.duration && (
              <div className="info-item">
                <span className="info-label">Duración:</span>
                <span className="info-value">{result.metadata.duration.toFixed(2)}s</span>
              </div>
            )}
            {result.metadata.resolution && (
              <div className="info-item">
                <span className="info-label">Resolución:</span>
                <span className="info-value">
                  {result.metadata.resolution.width} × {result.metadata.resolution.height}
                </span>
              </div>
            )}
            {result.metadata.source && (
              <div className="info-item">
                <span className="info-label">Fuente:</span>
                <span className="info-value">{result.metadata.source}</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="result-section">
        <h3>Embeddings Generados</h3>
        <p className="embedding-info">
          Se generaron {result.embeddings?.length || 0} embeddings de dimensión{' '}
          {result.embedding_shape?.[2] || 'N/A'}
        </p>
        <details className="embedding-details">
          <summary>Ver detalles técnicos</summary>
          <pre className="embedding-preview">
            {JSON.stringify(
              {
                clip_id: result.clip_id,
                embedding_shape: result.embedding_shape,
                num_embeddings: result.embeddings?.length,
                sample_embedding: result.embeddings?.[0]?.slice(0, 10) || []
              },
              null,
              2
            )}
          </pre>
        </details>
      </div>

      <div className="status-badge success">
        ✓ Inferencia completada exitosamente
      </div>
    </div>
  )
}

export default InferenceResult

