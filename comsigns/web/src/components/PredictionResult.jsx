import React from 'react'
import './PredictionResult.css'

/**
 * Component to display inference prediction results.
 * Shows top-1 prediction prominently and top-k list below.
 */
function PredictionResult({ result }) {
  if (!result) return null

  const { top1, topk, meta } = result

  // Get bucket badge color
  const getBucketColor = (bucket) => {
    switch (bucket) {
      case 'HEAD': return 'bucket-head'
      case 'MID': return 'bucket-mid'
      case 'OTHER': return 'bucket-other'
      default: return ''
    }
  }

  // Format confidence as percentage
  const formatConfidence = (conf) => {
    return `${(conf * 100).toFixed(1)}%`
  }

  return (
    <div className="prediction-result">
      {/* Top-1 Prediction */}
      <div className="top1-section">
        <h2 className="section-title">Predicción Principal</h2>
        <div className="top1-card">
          <div className="gloss-display">
            <span className="gloss-text">{top1.gloss}</span>
            {top1.is_other && (
              <span className="other-badge">⚠️ OTHER</span>
            )}
          </div>
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar"
              style={{ width: `${top1.confidence * 100}%` }}
            />
            <span className="confidence-text">
              {formatConfidence(top1.confidence)}
            </span>
          </div>
          <div className="top1-meta">
            <span className={`bucket-badge ${getBucketColor(top1.bucket)}`}>
              {top1.bucket}
            </span>
            {top1.old_class_id !== null && (
              <span className="class-id">ID: {top1.new_class_id}</span>
            )}
          </div>
        </div>
      </div>

      {/* Top-K Predictions */}
      <div className="topk-section">
        <h3 className="section-title">Top {topk.length} Predicciones</h3>
        <div className="topk-list">
          {topk.map((pred) => (
            <div 
              key={pred.rank} 
              className={`topk-item ${pred.rank === 1 ? 'topk-top1' : ''}`}
            >
              <span className="topk-rank">#{pred.rank}</span>
              <span className="topk-gloss">
                {pred.gloss}
                {pred.is_other && <span className="other-mini">⚠️</span>}
              </span>
              <div className="topk-confidence-container">
                <div 
                  className="topk-confidence-bar"
                  style={{ width: `${pred.confidence * 100}%` }}
                />
              </div>
              <span className="topk-percent">
                {formatConfidence(pred.confidence)}
              </span>
              <span className={`bucket-mini ${getBucketColor(pred.bucket)}`}>
                {pred.bucket}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Metadata */}
      <div className="meta-section">
        <details className="meta-details">
          <summary>Información del modelo</summary>
          <div className="meta-content">
            <div className="meta-item">
              <span className="meta-label">Modelo:</span>
              <span className="meta-value">{meta.model}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Clases:</span>
              <span className="meta-value">{meta.num_classes}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Dispositivo:</span>
              <span className="meta-value">{meta.device}</span>
            </div>
          </div>
        </details>
      </div>

      <div className="status-badge success">
        ✓ Inferencia completada
      </div>
    </div>
  )
}

export default PredictionResult
