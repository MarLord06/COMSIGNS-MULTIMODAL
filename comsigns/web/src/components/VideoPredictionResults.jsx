import React from 'react';
import './VideoPredictionResults.css';

/**
 * VideoPredictionResults - Display inference results for multiple videos
 * 
 * Shows:
 * - Video name
 * - Recognized word (gloss)
 * - Confidence score with visual bar
 * - Acceptance status
 */
function VideoPredictionResults({ results, errors = [] }) {
  if ((!results || results.length === 0) && (!errors || errors.length === 0)) {
    return null;
  }

  const acceptedCount = results?.filter(r => r.accepted).length || 0;
  const rejectedCount = results?.filter(r => !r.accepted).length || 0;

  // Format score as percentage
  const formatScore = (score) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  // Get score color based on value
  const getScoreColor = (score) => {
    if (score >= 0.6) return 'high';
    if (score >= 0.3) return 'medium';
    return 'low';
  };

  return (
    <div className="video-prediction-results">
      {/* Summary */}
      <div className="results-summary">
        <div className="summary-item total">
          <span className="summary-value">{(results?.length || 0) + (errors?.length || 0)}</span>
          <span className="summary-label">Videos</span>
        </div>
        <div className="summary-item accepted">
          <span className="summary-value">{acceptedCount}</span>
          <span className="summary-label">Aceptados</span>
        </div>
        <div className="summary-item rejected">
          <span className="summary-value">{rejectedCount}</span>
          <span className="summary-label">Rechazados</span>
        </div>
        {errors?.length > 0 && (
          <div className="summary-item errors">
            <span className="summary-value">{errors.length}</span>
            <span className="summary-label">Errores</span>
          </div>
        )}
      </div>

      {/* Results List */}
      <div className="results-list">
        {results?.map((result, index) => (
          <div 
            key={`${result.video}-${index}`} 
            className={`result-card ${result.accepted ? 'accepted' : 'rejected'}`}
          >
            {/* Status Badge */}
            <div className={`status-badge ${result.accepted ? 'accepted' : 'rejected'}`}>
              {result.accepted ? '✓' : '✗'}
            </div>

            {/* Video Info */}
            <div className="result-header">
              <span className="video-icon">📹</span>
              <span className="video-name" title={result.video}>
                {result.video}
              </span>
            </div>

            {/* Main Prediction */}
            <div className="prediction-main">
              <div className="gloss-display">
                <span className="gloss-label">Palabra reconocida:</span>
                <span className="gloss-value">{result.gloss}</span>
              </div>
              
              <div className="class-info">
                <span className="class-name">{result.class_name}</span>
                <span className="class-id">ID: {result.class_id}</span>
              </div>
            </div>

            {/* Confidence Score */}
            <div className="confidence-section">
              <div className="confidence-header">
                <span className="confidence-label">Confianza</span>
                <span className={`confidence-value ${getScoreColor(result.score)}`}>
                  {formatScore(result.score)}
                </span>
              </div>
              <div className="confidence-bar-container">
                <div 
                  className={`confidence-bar ${getScoreColor(result.score)}`}
                  style={{ width: `${result.score * 100}%` }}
                />
              </div>
            </div>

            {/* Status & Reason */}
            <div className="status-section">
              <div className={`status-label ${result.accepted ? 'accepted' : 'rejected'}`}>
                {result.accepted ? '✅ Aceptada' : '❌ Rechazada'}
              </div>
              {result.reason && (
                <p className="status-reason">{result.reason}</p>
              )}
            </div>
          </div>
        ))}

        {/* Error Cards */}
        {errors?.map((error, index) => (
          <div key={`error-${index}`} className="result-card error">
            <div className="status-badge error">!</div>
            
            <div className="result-header">
              <span className="video-icon">📹</span>
              <span className="video-name" title={error.video}>
                {error.video}
              </span>
            </div>

            <div className="error-message">
              <span className="error-icon">⚠️</span>
              <span>{error.error}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Recognized Words Summary */}
      {acceptedCount > 0 && (
        <div className="words-summary">
          <h4>Palabras reconocidas:</h4>
          <div className="words-list">
            {results
              ?.filter(r => r.accepted)
              .map((r, i) => (
                <span key={i} className="word-chip">
                  {r.gloss}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default VideoPredictionResults;
