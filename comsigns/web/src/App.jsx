import React, { useState } from 'react'
import VideoUploader from './components/VideoUploader'
import InferenceResult from './components/InferenceResult'
import './App.css'

function App() {
  const [result, setResult] = useState(null)
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

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>COMSIGNS</h1>
          <p>Sistema de Interpretación de Lengua de Señas</p>
        </header>

        <main className="main">
          <VideoUploader
            onUpload={handleInference}
            loading={loading}
          />

          {error && (
            <div className="error-message">
              <p>❌ Error: {error}</p>
            </div>
          )}

          {result && <InferenceResult result={result} />}
        </main>

        <footer className="footer">
          <p>Versión 0.1.0 - Sistema experimental</p>
        </footer>
      </div>
    </div>
  )
}

export default App

