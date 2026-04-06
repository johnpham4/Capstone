import { Share2, HelpCircle, FolderOpen, LogOut, Menu } from "lucide-react"
import { useRouter } from "next/navigation"
import { useState, useRef, useEffect } from "react"

interface HeaderProps {
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  onExport: () => void
  onShare: () => void
  hasCurrentDiagram: boolean
  onOpenArchive: () => void
  diagramCount: number
}

export default function Header({ 
  sidebarOpen, 
  setSidebarOpen, 
  onExport, 
  onShare, 
  hasCurrentDiagram,
  onOpenArchive,
  diagramCount
}: HeaderProps) {
  const router = useRouter()
  const [showDropdown, setShowDropdown] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const handleLogout = () => {
    localStorage.removeItem('geomath_visited')
    router.push('/')
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false)
      }
    }

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showDropdown])

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="bg-gradient-to-br from-blue-600 to-cyan-500 rounded-xl p-2">
          <span className="text-lg font-bold text-white">G</span>
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">GeoMath</h1>
          <p className="text-sm text-gray-500">Hệ thống vẽ hình học 2D tự động</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="relative" ref={dropdownRef}>
          <button 
            onClick={() => setShowDropdown(!showDropdown)}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>
          
          {showDropdown && (
            <div className="absolute right-0 top-full mt-2 w-56 bg-white rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 z-50">
              <div className="py-1">
                <button 
                  onClick={() => {
                    onOpenArchive()
                    setShowDropdown(false)
                  }}
                  className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-amber-700 hover:bg-amber-50 w-full text-left transition-colors relative"
                >
                  <FolderOpen className="h-4 w-4" />
                  <span>Lịch sử ảnh</span>
                  {diagramCount > 0 && (
                    <span className="ml-auto bg-amber-600 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                      {diagramCount}
                    </span>
                  )}
                </button>
                <button 
                  onClick={() => {
                    onShare()
                    setShowDropdown(false)
                  }}
                  className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-100 w-full text-left transition-colors"
                >
                  <Share2 className="h-4 w-4" />
                  <span>Chia sẻ</span>
                </button>
                <button 
                  onClick={() => setShowDropdown(false)}
                  className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-100 w-full text-left transition-colors"
                >
                  <HelpCircle className="h-4 w-4" />
                  <span>Trợ giúp</span>
                </button>
                <div className="border-t border-gray-100 my-1"></div>
                <button 
                  onClick={() => {
                    handleLogout()
                    setShowDropdown(false)
                  }}
                  className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-red-600 hover:bg-red-50 w-full text-left transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Đăng xuất</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
