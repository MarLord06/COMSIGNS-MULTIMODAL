import React, { useRef, useState, useEffect, useCallback } from 'react'
import './CameraCapture.css'

const CAPTURE_FPS = 10 // Capturar 10 frames por segundo
const FRAME_WIDTH = 640
const FRAME_HEIGHT = 480
const WS_URL = 'ws://localhost:8000/ws/infer'

function CameraCapture({ onPrediction, onError }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const captureIntervalRef = useRef(null)
  
  const [isActive, setIsActive] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [error, setError] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [frameCount, setFrameCount] = useState(0)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)

  // Conectar WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    console.log('Conectando WebSocket...')
    setConnectionStatus('connecting')
    
    const ws = new WebSocket(WS_URL)
    
    ws.onopen = () => {
      console.log('WebSocket conectado')
      setConnectionStatus('connected')
      setReconnectAttempts(0)
      setError(null)
    }
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        
        if (message.type === 'status') {
          if (message.status === 'connected') {
            setSessionId(message.session_id)
            console.log('Sesión iniciada:', message.session_id)
          }
        } else if (message.type === 'prediction') {
          setIsProcessing(false)
          if (onPrediction) {
            onPrediction(message.data)
          }
        } else if (message.type === 'error') {
          console.error('Error del servidor:', message.error)
          setError(message.error)
          if (onError) {
            onError(message.error)
          }
        }
      } catch (err) {
        console.error('Error parseando mensaje:', err)
      }
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnectionStatus('error')
      setError('Error de conexión WebSocket')
    }
    
    ws.onclose = () => {
      console.log('WebSocket desconectado')
      setConnectionStatus('disconnected')
      
      // Reconexión automática si la cámara está activa
      if (isActive) {
        const delay = Math.min(3000 * Math.pow(2, reconnectAttempts), 30000)
        console.log(`Reintentando conexión en ${delay}ms...`)
        
        setTimeout(() => {
          setReconnectAttempts(prev => prev + 1)
          connectWebSocket()
        }, delay)
      }
    }
    
    wsRef.current = ws
  }, [isActive, reconnectAttempts, onPrediction, onError])

  // Enviar frame al servidor
  const sendFrame = useCallback((frameBase64) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setIsProcessing(true)
      
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: {
          frame: frameBase64,
          timestamp: Date.now(),
          sequence: frameCount
        }
      }))
      
      setFrameCount(prev => prev + 1)
    }
  }, [frameCount])

  // Capturar frame desde video
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return
    
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    // Dibujar frame actual en canvas
    ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT)
    
    // Convertir a base64
    const frameBase64 = canvas.toDataURL('image/jpeg', 0.8)
    
    // Enviar al servidor
    sendFrame(frameBase64)
  }, [sendFrame])

  // Iniciar cámara
  const startCamera = async () => {
    try {
      setError(null)
      
      // Solicitar acceso a la cámara
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: FRAME_WIDTH },
          height: { ideal: FRAME_HEIGHT },
          facingMode: 'user'
        },
        audio: false
      })
      
      streamRef.current = stream
      
      // Asignar stream al elemento video
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      
      // Conectar WebSocket
      connectWebSocket()
      
      // Iniciar captura de frames
      const interval = setInterval(captureFrame, 1000 / CAPTURE_FPS)
      captureIntervalRef.current = interval
      
      setIsActive(true)
      setFrameCount(0)
      
    } catch (err) {
      console.error('Error accediendo a la cámara:', err)
      setError(err.message || 'No se pudo acceder a la cámara')
      if (onError) {
        onError(err.message)
      }
    }
  }

  // Detener cámara
  const stopCamera = () => {
    // Detener captura de frames
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    
    // Detener stream de video
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    // Cerrar WebSocket
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'control',
        action: 'stop'
      }))
      wsRef.current.close()
      wsRef.current = null
    }
    
    setIsActive(false)
    setConnectionStatus('disconnected')
    setSessionId(null)
    setFrameCount(0)
  }

  // Resetear sesión
  const resetSession = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'control',
        action: 'reset'
      }))
      setFrameCount(0)
    }
  }

  // Cleanup al desmontar
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  // Obtener color del indicador de estado
  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#4ade80'
      case 'connecting': return '#fbbf24'
      case 'error': return '#ef4444'
      default: return '#9ca3af'
    }
  }

  // Obtener texto del estado
  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Conectado'
      case 'connecting': return 'Conectando...'
      case 'error': return 'Error'
      default: return 'Desconectado'
    }
  }

  return (
    <div className="camera-capture">
      <div className="camera-container">
        <video
          ref={videoRef}
          className="camera-preview"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          width={FRAME_WIDTH}
          height={FRAME_HEIGHT}
          style={{ display: 'none' }}
        />
        
        {!isActive && (
          <div className="camera-overlay">
            <div className="camera-icon">📹</div>
            <p>Cámara inactiva</p>
          </div>
        )}
        
        {isProcessing && isActive && (
          <div className="processing-indicator">
            <div className="spinner-small"></div>
          </div>
        )}
      </div>

      <div className="camera-controls">
        <div className="status-bar">
          <div className="status-indicator">
            <div 
              className="status-dot" 
              style={{ backgroundColor: getStatusColor() }}
            />
            <span>{getStatusText()}</span>
          </div>
          
          {sessionId && (
            <div className="session-info">
              <span className="session-label">Sesión:</span>
              <span className="session-id">{sessionId.substring(0, 8)}</span>
            </div>
          )}
          
          {isActive && (
            <div className="frame-counter">
              <span>Frames: {frameCount}</span>
            </div>
          )}
        </div>

        {error && (
          <div className="error-banner">
            ⚠️ {error}
          </div>
        )}

        <div className="button-group">
          {!isActive ? (
            <button
              onClick={startCamera}
              className="btn btn-primary"
            >
              🎥 Iniciar Cámara
            </button>
          ) : (
            <>
              <button
                onClick={stopCamera}
                className="btn btn-danger"
              >
                ⏹️ Detener
              </button>
              <button
                onClick={resetSession}
                className="btn btn-secondary"
                disabled={connectionStatus !== 'connected'}
              >
                🔄 Resetear
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default CameraCapture
