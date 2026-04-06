import { ZoomIn, ZoomOut, RotateCcw, Image, Download, Grid3X3, Grid2X2X } from "lucide-react"

interface Diagram {
  id: string
  title: string
  image_base64: string
  dsl?: string
}

interface DiagramViewerProps {
  currentDiagrams: Diagram[]
  selectedDiagramIndex: number
  setSelectedDiagramIndex: (index: number) => void
  zoomLevel: number
  onZoomIn: () => void
  onZoomOut: () => void
  onZoomReset: () => void
  onWheel: (e: React.WheelEvent) => void
  onExport: () => void
  hasCurrentDiagram: boolean
  showGrid: boolean
  onToggleGrid: () => void
}

export default function DiagramViewer({
  currentDiagrams,
  selectedDiagramIndex,
  setSelectedDiagramIndex,
  zoomLevel,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onWheel,
  onExport,
  hasCurrentDiagram,
  showGrid,
  onToggleGrid,
}: DiagramViewerProps) {
  return (
    <div className="flex flex-1 flex-col bg-white" style={{ backgroundColor: "#ffffff" }}>
      {/* Diagram Display */}
      <div 
        className="relative flex flex-1 items-center justify-center bg-white p-8"
        style={{ backgroundColor: "#ffffff" }}
        onWheel={onWheel}
      >
        {/* Zoom Instructions */}
        <div className="absolute top-4 right-4 z-20 flex flex-col gap-2 text-xs text-gray-400">
          {/* Zoom percentage, reset button, and download button */}
          <div className="flex items-center justify-end gap-2">
            <button
              onClick={onToggleGrid}
              className={`rounded p-1 transition-colors ${showGrid ? "bg-blue-100 hover:bg-blue-200" : "hover:bg-gray-100"}`}
              title={showGrid ? "Tắt lưới" : "Bật lưới"}
              aria-label={showGrid ? "Tắt lưới" : "Bật lưới"}
            >
              {showGrid ? (
                <Grid2X2X className="h-3.5 w-3.5 text-blue-600" />
              ) : (
                <Grid3X3 className="h-3.5 w-3.5 text-gray-500" />
              )}
            </button>
            <span className="font-semibold text-gray-600">{zoomLevel}%</span>
            <button
              onClick={onZoomReset}
              className="p-1 rounded hover:bg-gray-100 transition-colors"
              title="Reset zoom"
            >
              <RotateCcw className="h-3.5 w-3.5 text-gray-500" />
            </button>
            <button
              onClick={onExport}
              disabled={!hasCurrentDiagram}
              className="p-1 rounded hover:bg-gray-100 transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              title="Tải xuống"
            >
              <Download className="h-3.5 w-3.5 text-gray-500" />
            </button>
          </div>
          
          {/* Instructions */}
          <div className="flex items-center gap-1.5">
            <ZoomIn className="h-3.5 w-3.5" />
            <span>scroll lên để phóng to</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ZoomOut className="h-3.5 w-3.5" />
            <span>scroll xuống để thu nhỏ</span>
          </div>
        </div>

        {/* White background layer for the drawing area */}
        <div className="absolute inset-0 pointer-events-none bg-white" style={{ backgroundColor: "#ffffff" }} />

        {showGrid && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(to right, rgba(229, 231, 235, 0.7) 1px, transparent 1px), linear-gradient(to bottom, rgba(229, 231, 235, 0.7) 1px, transparent 1px)",
              backgroundSize: "40px 40px",
            }}
          >
            <div className="absolute h-px w-full bg-gray-300" style={{ top: "50%" }}></div>
            <div className="absolute h-full w-px bg-gray-300" style={{ left: "50%" }}></div>
          </div>
        )}
        
        {currentDiagrams.length > 0 && currentDiagrams[selectedDiagramIndex] ? (
          <div className="relative z-10 flex h-full w-full flex-col">
            <div className="mb-4 text-center">
              <h3 className="text-lg font-semibold text-gray-800">
                {currentDiagrams[selectedDiagramIndex].title}
              </h3>
            </div>
            <div className="flex flex-1 items-center justify-center overflow-auto">
              <img
                src={`data:image/png;base64,${currentDiagrams[selectedDiagramIndex].image_base64}`}
                alt="Geometry Diagram"
                className="max-h-full max-w-full object-contain transition-transform duration-200"
                style={{ transform: `scale(${zoomLevel / 100})` }}
              />
            </div>
          </div>
        ) : (
          <div className="relative z-10 text-center text-gray-400">
            <div className="mb-4 flex justify-center">
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
                <Image className="h-16 w-16 text-blue-600" strokeWidth={1.5} />
              </div>
            </div>
            <p className="text-sm">Hình vẽ của bạn sẽ xuất hiện ở đây</p>
          </div>
        )}
      </div>

      {/* Diagram Gallery/Tabs */}
      {currentDiagrams.length > 0 && (
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-medium text-gray-600">
              HÌNH VẼ ({currentDiagrams.length})
            </p>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            {currentDiagrams.map((diagram, idx) => (
              <button
                key={diagram.id}
                onClick={() => setSelectedDiagramIndex(idx)}
                className={`group relative flex-shrink-0 overflow-hidden border-2 transition-all ${
                  selectedDiagramIndex === idx
                    ? "border-blue-500 ring-2 ring-blue-200"
                    : "border-gray-200 hover:border-blue-300"
                }`}
              >
                <img
                  src={`data:image/png;base64,${diagram.image_base64}`}
                  alt={diagram.title}
                  className="h-20 w-28 object-cover"
                />
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-0 transition-all group-hover:bg-opacity-30">
                  <span className="text-xs font-medium text-white opacity-0 group-hover:opacity-100">
                    Xem
                  </span>
                </div>
                {selectedDiagramIndex === idx && (
                  <div className="absolute right-1 top-1 rounded-full bg-blue-600 p-1">
                    <svg className="h-3 w-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
