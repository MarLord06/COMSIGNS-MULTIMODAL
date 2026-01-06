import React, { useState } from 'react'
import VideoUploader from './components/VideoUploader'
import CameraCapture from './components/CameraCapture'
import InferenceResult from './components/InferenceResult'
import RealtimeResult from './components/RealtimeResult'
import './App.css'

function App() {
  const [mode, setMode] = useState('camera') // 'camera' o 'upload'
  const [result, setResult] = useState(null)
  const [realtimePrediction, setRealtimePrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInference = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/infer/video', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Error en la inferencia')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Error desconocido')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRealtimePrediction = (prediction) => {
    setRealtimePrediction(prediction)
  }

  const handleCameraError = (errorMsg) => {
    setError(errorMsg)
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🤟</div>
            <div>
              <h1>COMSIGNS</h1>
              <p>Traducción de Lengua de Señas en Tiempo Real</p>
            </div>
          </div>
          
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
              onClick={() => setMode('camera')}
            >
              <span className="btn-icon">🎥</span>
              <span>Cámara en Vivo</span>
            </button>
            <button
              className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
              onClick={() => setMode('upload')}
            >
              <span className="btn-icon">📤</span>
              <span>Subir Video</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="dashboard">
        {mode === 'camera' ? (
          <div className="dashboard-grid">
            {/* Left Column - Camera */}
            <div className="dashboard-left">
              <div className="section-card">
                <div className="section-header">
                  <h2>📹 Vista de Cámara</h2>
                  <p>Captura en tiempo real a 10 FPS</p>
                </div>
                <CameraCapture
                  onPrediction={handleRealtimePrediction}
                  onError={handleCameraError}
                />
              </div>
            </div>

            {/* Right Column - Results */}
            <div className="dashboard-right">
              <div className="section-card">
                <div className="section-header">
                  <h2>💬 Traducción en Tiempo Real</h2>
                  <p>Resultados instantáneos del modelo</p>
                </div>
                {realtimePrediction ? (
                  <RealtimeResult prediction={realtimePrediction} />
                ) : (
                  <div className="empty-state">
                    <div className="empty-icon">🎯</div>
                    <h3>Esperando señas...</h3>
                    <p>Inicia la cámara y comienza a hacer señas para ver la traducción aquí</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="upload-container">
            <div className="section-card full-width">
              <div className="section-header">
                <h2>📤 Subir Video</h2>
                <p>Procesa un archivo de video completo</p>
              </div>
              <VideoUploader
                onUpload={handleInference}
                loading={loading}
              />
              {result && (
                <div className="results-section">
                  <InferenceResult result={result} />
                </div>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="error-toast">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
            <button onClick={() => setError(null)} className="close-btn">✕</button>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <p>COMSIGNS v0.2.0 - Sistema de Traducción de Lengua de Señas</p>
          <div className="footer-links">
            <span>Powered by MediaPipe + PyTorch</span>
            <span>•</span>
            <span>Real-time WebSocket</span>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

