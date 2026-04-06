"use client"

import { useState, useEffect } from "react"
import { X, FolderOpen, Maximize2, Minimize2, Eye, EyeOff, ArrowRight } from "lucide-react"
import Header from "../components/Header"
import InputArea from "../components/InputArea"
import DiagramViewer from "../components/DiagramViewer"

export default function ChatPage() {
  const [currentDiagrams, setCurrentDiagrams] = useState<Array<{
    id: string
    title: string
    image_base64: string
    dsl?: string
    createdAt: Date
    prompt: string
  }>>([])
  const [selectedDiagramIndex, setSelectedDiagramIndex] = useState<number>(0)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState("")
  const [solution, setSolution] = useState<string>("")
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [zoomLevel, setZoomLevel] = useState(100)
  const [inputAreaWidth, setInputAreaWidth] = useState(50)
  const [isResizing, setIsResizing] = useState(false)
  const [windowWidth, setWindowWidth] = useState(1024)
  const [showArchiveModal, setShowArchiveModal] = useState(false)
  const [showGrid, setShowGrid] = useState(false)
  const [modalPosition, setModalPosition] = useState({ x: 0, y: 0 })
  const [modalSize, setModalSize] = useState({ width: 600, height: 600 })
  const [isDraggingModal, setIsDraggingModal] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [isResizingModal, setIsResizingModal] = useState(false)
  const [resizeDirection, setResizeDirection] = useState<'right' | 'bottom' | 'corner' | null>(null)
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, width: 0, height: 0 })
  const [isModalFullscreen, setIsModalFullscreen] = useState(false)
  
  // Welcome modal states
  const [showWelcomeModal, setShowWelcomeModal] = useState(true)
  const [welcomeModalPage, setWelcomeModalPage] = useState(0) // 0: welcome, 1: login
  const [showPassword, setShowPassword] = useState(false)
  const [isRegisterMode, setIsRegisterMode] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  
  // Welcome modal handlers
  const handleWelcomeNext = () => {
    setWelcomeModalPage(1)
  }
  
  const handleWelcomeSkip = () => {
    localStorage.setItem('geomath_visited', 'true')
    setShowWelcomeModal(false)
  }
  
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    if (username && password) {
      localStorage.setItem('geomath_visited', 'true')
      setShowWelcomeModal(false)
    }
  }

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirmPassword) {
      alert('Mật khẩu không khớp!')
      return
    }
    if (username && password && confirmPassword) {
      localStorage.setItem('geomath_visited', 'true')
      setShowWelcomeModal(false)
    }
  }

  const handleSwitchMode = () => {
    setIsRegisterMode(!isRegisterMode)
    setUsername('')
    setPassword('')
    setConfirmPassword('')
    setShowPassword(false)
  }
  
  const handleExport = () => {
    if (currentDiagrams.length > 0 && currentDiagrams[selectedDiagramIndex]) {
      const diagram = currentDiagrams[selectedDiagramIndex]
      const link = document.createElement('a')
      link.href = `data:image/png;base64,${diagram.image_base64}`
      link.download = `${diagram.title.replace(/\s+/g, '-')}.png`
      link.click()
    }
  }

  const handleShare = () => {
    alert("Chức năng chia sẻ - Sắp ra mắt!")
  }

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 10, 200))
  }

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 10, 50))
  }

  const handleZoomReset = () => {
    setZoomLevel(100)
  }

  const handleArchiveItemClick = (index: number) => {
    setShowArchiveModal(false)
    setSelectedDiagramIndex(index)
  }

  const handleModalMouseDown = (e: React.MouseEvent) => {
    setIsDraggingModal(true)
    setDragStart({
      x: e.clientX - modalPosition.x,
      y: e.clientY - modalPosition.y
    })
  }

  const handleResizeMouseDown = (e: React.MouseEvent, direction: 'right' | 'bottom' | 'corner') => {
    e.stopPropagation()
    setIsResizingModal(true)
    setResizeDirection(direction)
    setResizeStart({
      x: e.clientX,
      y: e.clientY,
      width: modalSize.width,
      height: modalSize.height
    })
  }

  const toggleModalFullscreen = () => {
    setIsModalFullscreen(!isModalFullscreen)
  }

  // Reset modal position and size when opened
  useEffect(() => {
    if (showArchiveModal) {
      const modalWidth = 600
      const modalHeight = 600
      // Calculate centered position
      const x = (window.innerWidth - modalWidth) / 2
      const y = (window.innerHeight - modalHeight) / 2
      setModalPosition({ x, y })
      setModalSize({ width: modalWidth, height: modalHeight })
      setIsModalFullscreen(false)
    }
  }, [showArchiveModal])

  useEffect(() => {
    const handleModalMouseMove = (e: MouseEvent) => {
      if (isDraggingModal) {
        setModalPosition({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y
        })
      }
      
      if (isResizingModal && resizeDirection) {
        const deltaX = e.clientX - resizeStart.x
        const deltaY = e.clientY - resizeStart.y
        
        if (resizeDirection === 'right') {
          setModalSize({
            width: Math.max(400, resizeStart.width + deltaX),
            height: modalSize.height
          })
        } else if (resizeDirection === 'bottom') {
          setModalSize({
            width: modalSize.width,
            height: Math.max(400, resizeStart.height + deltaY)
          })
        } else if (resizeDirection === 'corner') {
          setModalSize({
            width: Math.max(400, resizeStart.width + deltaX),
            height: Math.max(400, resizeStart.height + deltaY)
          })
        }
      }
    }

    const handleModalMouseUp = () => {
      setIsDraggingModal(false)
      setIsResizingModal(false)
      setResizeDirection(null)
    }

    if (isDraggingModal || isResizingModal) {
      document.addEventListener('mousemove', handleModalMouseMove)
      document.addEventListener('mouseup', handleModalMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleModalMouseMove)
      document.removeEventListener('mouseup', handleModalMouseUp)
    }
  }, [isDraggingModal, isResizingModal, dragStart, resizeDirection, resizeStart, modalSize])

  const handleMouseDown = () => {
    setIsResizing(true)
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -10 : 10
    setZoomLevel((prev) => Math.max(20, Math.min(300, prev + delta)))
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return
      
      const windowWidth = window.innerWidth
      const availableWidth = windowWidth
      const mouseX = e.clientX
      
      const newWidthPercent = (mouseX / availableWidth) * 100
      
      if (newWidthPercent >= 30 && newWidthPercent <= 70) {
        setInputAreaWidth(newWidthPercent)
      }
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing])

  useEffect(() => {
    const updateWindowWidth = () => {
      setWindowWidth(window.innerWidth)
    }
    
    updateWindowWidth()
    window.addEventListener('resize', updateWindowWidth)
    
    return () => window.removeEventListener('resize', updateWindowWidth)
  }, [])

  const handleSubmit = async (topic: string, uploadedImage?: string) => {
    if (loading) return

    setLoading(true)
    setProgress(0)
    setStatusMessage("Connecting...")

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL

      // If there's an uploaded image, we might want to handle it differently
      // For now, we'll just use the topic for the API call
      const params = new URLSearchParams({
        user_input: topic || "Tạo hình học từ ảnh",
        max_tokens: "1024",
        temperature: "0.7",
        language: "vi"
      })

      const sseUrl = `${backendUrl}/api/v1/diagrams/stream-pipeline?${params.toString()}`
      console.log("🔗 SSE URL:", sseUrl)

      const response = await fetch(sseUrl, {
        headers: {
          'Accept': 'text/event-stream',
          'ngrok-skip-browser-warning': 'true'
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      console.log("✅ SSE connection opened")

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error("No response body")
      }

      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')

buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim()
            if (!jsonStr) continue

            try {
              console.log("📨 SSE message:", jsonStr)
              const data = JSON.parse(jsonStr)

              setProgress(data.progress || 0)
              setStatusMessage(data.status || "")

              if (data.status === "completed") {
                const imageTitle = topic.slice(0, 30) + (topic.length > 30 ? "..." : "")
                
                // Add to current diagrams display
                const newDiagram = {
                  id: Date.now().toString(),
                  title: imageTitle,
                  image_base64: data.image_base64,
                  dsl: data.dsl,
                  createdAt: new Date(),
                  prompt: topic || "Tạo hình học từ ảnh"
                }
                setCurrentDiagrams(prev => {
                  const updated = [...prev, newDiagram]
                  setSelectedDiagramIndex(updated.length - 1) // Select the newly added diagram
                  return updated
                })

                // Set solution if available
                if (data.solution || data.explanation) {
                  setSolution(data.solution || data.explanation)
                } else {
                  setSolution(`Đã tạo hình vẽ thành công cho đề bài: "${topic}"\n\nHình vẽ đã được tạo với các thông số như mô tả.`)
                }

                setLoading(false)
                setProgress(0)
                setStatusMessage("")

                reader.cancel()
                return
              } else if (data.status === "error") {
                throw new Error(data.error || "Unknown error")
              }
            } catch (parseError) {
              console.warn("Failed to parse SSE data:", jsonStr, parseError)
            }
          }
        }
      }

    } catch (error) {
      console.error("Error:", error)
      alert(`Lỗi hệ thống: ${error instanceof Error ? error.message : "Lỗi không xác định"}`)
      setLoading(false)
      setProgress(0)
      setStatusMessage("")
    }
  }

  return (
    <div className="flex h-screen flex-col bg-white">
      {/* Welcome Modal */}
      {showWelcomeModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-[100] p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl overflow-hidden shadow-2xl">
            {welcomeModalPage === 0 ? (
              // Page 1: Welcome/Hero
              <div className="relative p-12 md:p-16 bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white min-h-[500px] flex flex-col items-center justify-center">
                <button
                  onClick={handleWelcomeSkip}
                  className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
                >
                  <X className="h-6 w-6" />
                </button>
                
                <div className="text-center max-w-3xl">
                  <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
                    <span className="inline-block bg-gradient-to-r from-white via-cyan-400 via-blue-400 via-cyan-300 to-white bg-clip-text text-transparent animate-gradient-x" style={{backgroundSize: '200% auto'}}>
                      Hệ thống vẽ hình học 2D
                    </span>
                    <br />
                    <span className="inline-block bg-gradient-to-r from-white via-cyan-400 via-blue-400 via-cyan-300 to-white bg-clip-text text-transparent animate-gradient-x" style={{backgroundSize: '200% auto'}}>
                      chính xác & tự động
                    </span>
                  </h1>
                  
                  <p className="text-xl text-gray-300 mb-12 leading-relaxed">
                    Chỉ cần nhập bài toán, hệ thống AI sẽ tự động vẽ hình 2D chính xác. Tiết kiệm thời gian, nâng cao độ chính xác cho học sinh và giáo viên THCS.
                  </p>
                  
                  <button
                    onClick={handleWelcomeNext}
                    className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-lg font-semibold text-lg hover:shadow-xl hover:shadow-blue-500/50 hover:scale-105 transition-all duration-300"
                  >
                    Đăng nhập
                    <ArrowRight className="h-5 w-5" />
                  </button>
                </div>
              </div>
            ) : (
              // Page 2: Login/Register  
              <div className="relative p-12 md:p-16 bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white min-h-[500px] flex items-center justify-center">
                <button
                  onClick={() => setWelcomeModalPage(0)}
                  className="absolute top-6 left-6 text-gray-400 hover:text-white transition-colors text-sm flex items-center gap-1"
                >
                  ← Quay lại
                </button>
                
                <div className="w-full max-w-4xl grid md:grid-cols-2 gap-12 items-center">
                  {/* Left Column - Login Form */}
                  <div className="space-y-6">
                    <h2 className="text-3xl font-bold text-white mb-6 text-center">
                      {isRegisterMode ? 'Đăng ký' : 'Đăng nhập'}
                    </h2>

                    <form onSubmit={isRegisterMode ? handleRegister : handleLogin} className="space-y-4">
                      <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="Nhập tên đăng nhập"
                        required
                      />

                      <div className="relative">
                        <input
                          type={showPassword ? "text" : "password"}
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent pr-12"
                          placeholder="Nhập mật khẩu"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                        >
                          {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                        </button>
                      </div>

                      {isRegisterMode && (
                        <input
                          type={showPassword ? "text" : "password"}
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          placeholder="Nhập lại mật khẩu"
                          required
                        />
                      )}

                      <button
                        type="submit"
                        className="w-full bg-gradient-to-r from-orange-500 to-orange-600 text-white py-3 rounded-lg font-semibold hover:from-orange-600 hover:to-orange-700 transition-all duration-300 shadow-lg hover:shadow-xl"
                      >
                        {isRegisterMode ? 'Đăng ký' : 'Đăng nhập'}
                      </button>
                    </form>

                    <p className="text-center text-gray-400">
                      {isRegisterMode ? 'Đã có tài khoản?' : 'Chưa có tài khoản?'}{' '}
                      <button 
                        type="button"
                        onClick={handleSwitchMode}
                        className="text-blue-400 font-semibold hover:text-blue-300 transition-colors"
                      >
                        {isRegisterMode ? 'Đăng nhập ngay' : 'Đăng ký ngay'}
                      </button>
                    </p>
                  </div>

                  {/* Right Column - Alternative Options */}
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-gray-400 text-center">Hoặc</h3>
                    
                    <div className="space-y-4">
                      <button className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white/10 border border-gray-600 rounded-lg text-white font-medium hover:bg-white/20 transition-colors">
                        <svg className="h-5 w-5" viewBox="0 0 24 24">
                          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                          <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                        Tiếp tục với Google
                      </button>
                      
                      <button
                        onClick={handleWelcomeSkip}
                        className="w-full px-4 py-3 text-gray-300 bg-white/5 border border-gray-600 rounded-lg font-medium hover:bg-white/10 transition-colors"
                      >
                        Bỏ qua, dùng thử ngay
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Archive Modal */}
      {showArchiveModal && (
        <div className="fixed inset-0 z-50 pointer-events-none">
          <div 
            className="bg-white rounded-lg flex flex-col shadow-2xl pointer-events-auto absolute" 
            style={isModalFullscreen ? {
              left: 0,
              top: 0,
              width: '100vw',
              height: '100vh'
            } : { 
              left: `${modalPosition.x}px`,
              top: `${modalPosition.y}px`,
              width: `${modalSize.width}px`,
              height: `${modalSize.height}px`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div 
              className="flex items-center justify-between p-6 border-b border-gray-200 cursor-move select-none"
              onMouseDown={isModalFullscreen ? undefined : handleModalMouseDown}
            >
              <h2 className="text-xl font-bold text-gray-900">Thư mục chứa ảnh</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={toggleModalFullscreen}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  title={isModalFullscreen ? "Thu nhỏ" : "Phóng to"}
                >
                  {isModalFullscreen ? (
                    <Minimize2 className="h-5 w-5" />
                  ) : (
                    <Maximize2 className="h-5 w-5" />
                  )}
                </button>
                <button
                  onClick={() => setShowArchiveModal(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  title="Đóng"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              {currentDiagrams.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-gray-400">
                  <FolderOpen className="h-16 w-16 mb-4" />
                  <p className="text-lg">Chưa có hình vẽ nào được tạo</p>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-4">
                  {currentDiagrams.map((diagram, index) => (
                    <div
                      key={diagram.id}
                      className="cursor-pointer group relative border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-all hover:border-blue-400"
                      onClick={() => handleArchiveItemClick(index)}
                    >
                      <div className="aspect-square bg-gray-50 flex items-center justify-center p-2">
                        <img
                          src={`data:image/png;base64,${diagram.image_base64}`}
                          alt={diagram.title}
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                      <div className="p-3 bg-white group-hover:bg-blue-50 transition-colors">
                        <p className="text-sm text-gray-700 truncate font-medium">{diagram.title}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Resize Handles - Hidden in fullscreen mode */}
            {!isModalFullscreen && (
              <>
                <div 
                  className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-blue-400 hover:bg-opacity-50 transition-colors"
                  onMouseDown={(e) => handleResizeMouseDown(e, 'right')}
                />
                <div 
                  className="absolute left-0 right-0 bottom-0 h-2 cursor-ns-resize hover:bg-blue-400 hover:bg-opacity-50 transition-colors"
                  onMouseDown={(e) => handleResizeMouseDown(e, 'bottom')}
                />
                <div 
                  className="absolute right-0 bottom-0 w-4 h-4 cursor-nwse-resize hover:bg-blue-500 hover:bg-opacity-70 transition-colors rounded-bl-lg"
                  onMouseDown={(e) => handleResizeMouseDown(e, 'corner')}
                />
              </>
            )}
          </div>
        </div>
      )}

      <Header 
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        onExport={handleExport}
        onShare={handleShare}
        hasCurrentDiagram={currentDiagrams.length > 0}
        onOpenArchive={() => setShowArchiveModal(true)}
        diagramCount={currentDiagrams.length}
      />

      <div className="relative flex flex-1 overflow-hidden bg-white">
        {/* Input Area */}
        <div 
          className="flex w-full flex-col bg-white"
          style={{ width: windowWidth >= 1024 ? `${inputAreaWidth}%` : '100%' }} 
        >
          <div className="flex flex-1 flex-col overflow-hidden border-3 border-[#DAE3E3]">
            <InputArea 
              onSubmit={handleSubmit}
              loading={loading}
              progress={progress}
              statusMessage={statusMessage}
              solution={solution}
              diagrams={currentDiagrams}
              selectedDiagramIndex={selectedDiagramIndex}
              onSelectDiagram={setSelectedDiagramIndex}
            />
          </div>
        </div>

        {/* Resize Handle */}
        <div
          className="hidden lg:flex cursor-col-resize transition-colors hover:bg-[#b8c5c5]"
          onMouseDown={handleMouseDown}
          style={{ 
            width: '2.3px',
            backgroundColor: '#DAE3E3',
            cursor: 'col-resize'
          }}
        />

        {/* Right Panel - Diagram Display */}
        <div
          className="hidden flex-col border-t-3 border-[#DAE3E3] bg-white lg:flex"
          style={{ width: windowWidth >= 1024 ? `${100 - inputAreaWidth}%` : '0%', backgroundColor: '#ffffff' }}
        >
          <DiagramViewer 
            currentDiagrams={currentDiagrams}
            selectedDiagramIndex={selectedDiagramIndex}
            setSelectedDiagramIndex={setSelectedDiagramIndex}
            zoomLevel={zoomLevel}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onZoomReset={handleZoomReset}
            onWheel={handleWheel}
            onExport={handleExport}
            hasCurrentDiagram={currentDiagrams.length > 0}
            showGrid={showGrid}
            onToggleGrid={() => setShowGrid((prev) => !prev)}
          />
        </div>
      </div>
    </div>
  )
}
