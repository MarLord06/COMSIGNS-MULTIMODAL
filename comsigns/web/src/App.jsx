import React, { useState } from 'react'
import VideoUploader from './components/VideoUploader'
import CameraCapture from './components/CameraCapture'
import InferenceResult from './components/InferenceResult'
import RealtimeResult from './components/RealtimeResult'
import SampleUploader from './components/SampleUploader'
import PredictionResult from './components/PredictionResult'
import ExpertPanel from './components/ExpertPanel'
import VideoTranslator from './components/VideoTranslator'
import VideoPredictionResults from './components/VideoPredictionResults'
import { inferFromVideos } from './api/videoApi'
import './App.css'

// API URL - change for production
const API_URL = 'http://localhost:8000'

function App() {
  const [mode, setMode] = useState('video') // 'video', 'expert', 'sample', 'camera', or 'upload'
  const [result, setResult] = useState(null)
  const [predictionResult, setPredictionResult] = useState(null)
  const [realtimePrediction, setRealtimePrediction] = useState(null)
  const [videoResults, setVideoResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Handle video upload inference (existing)
  const handleVideoInference = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/infer/video`, {
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

  // Handle .pkl sample inference (new)
  const handleSampleInference = async (file, topk = 5) => {
    setLoading(true)
    setError(null)
    setPredictionResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/infer?topk=${topk}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Error en la inferencia')
      }

      const data = await response.json()
      setPredictionResult(data)
    } catch (err) {
      setError(err.message || 'Error de conexión con el servidor')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  // Handle video translation (multiple videos)
  const handleVideoTranslation = async (files) => {
    setLoading(true)
    setError(null)
    setVideoResults(null)

    try {
      const data = await inferFromVideos(files, { topk: 5 })
      setVideoResults(data)
    } catch (err) {
      setError(err.message || 'Error en la traducción de video')
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
              className={`mode-btn ${mode === 'video' ? 'active' : ''}`}
              onClick={() => setMode('video')}
            >
              <span className="btn-icon">🔮</span>
              <span>Traducir Video</span>
            </button>
            <button
              className={`mode-btn ${mode === 'expert' ? 'active' : ''}`}
              onClick={() => setMode('expert')}
            >
              <span className="btn-icon">🧠</span>
              <span>Sistema Experto</span>
            </button>
            <button
              className={`mode-btn ${mode === 'sample' ? 'active' : ''}`}
              onClick={() => setMode('sample')}
            >
              <span className="btn-icon">📦</span>
              <span>Inferir Sample</span>
            </button>
            <button
              className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
              onClick={() => setMode('camera')}
            >
              <span className="btn-icon">🎥</span>
              <span>Cámara en Vivo</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="dashboard">
        {mode === 'video' ? (
          <div className="upload-container">
            <div className="section-card full-width">
              <div className="section-header">
                <h2>🔮 Traducir Videos</h2>
                <p>Sube uno o varios videos de señas para traducir</p>
              </div>
              <VideoTranslator
                onInfer={handleVideoTranslation}
                loading={loading}
              />
              {videoResults && (
                <div className="results-section">
                  <VideoPredictionResults 
                    results={videoResults.results}
                    errors={videoResults.errors}
                  />
                </div>
              )}
            </div>
          </div>
        ) : mode === 'expert' ? (
          <ExpertPanel />
        ) : mode === 'sample' ? (
          <div className="upload-container">
            <div className="section-card full-width">
              <div className="section-header">
                <h2>📦 Inferir Sample</h2>
                <p>Sube un archivo .pkl con features extraídos</p>
              </div>
              <SampleUploader
                onUpload={handleSampleInference}
                loading={loading}
              />
              {predictionResult && (
                <div className="results-section">
                  <PredictionResult result={predictionResult} />
                </div>
              )}
            </div>
          </div>
        ) : mode === 'camera' ? (
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
        ) : null}

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
          <p>COMSIGNS v0.3.0 - Sistema de Traducción de Lengua de Señas</p>
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

