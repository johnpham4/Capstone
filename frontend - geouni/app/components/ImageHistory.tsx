import { Search, Image as ImageIcon, Filter, Calendar } from "lucide-react"
import { GeneratedImage, ImageCategory } from "../types"

interface ImageHistoryProps {
  images: GeneratedImage[]
  categories: ImageCategory[]
  searchQuery: string
  setSearchQuery: (query: string) => void
  selectedImage: string | null
  onSelectImage: (id: string) => void
  onDeleteImage: (id: string) => void
  selectedCategory: string | null
  setSelectedCategory: (id: string | null) => void
}

export default function ImageHistory(props: ImageHistoryProps) {
  const {
    images,
    categories,
    searchQuery,
    setSearchQuery,
    selectedImage,
    onSelectImage,
    onDeleteImage,
    selectedCategory,
    setSelectedCategory,
  } = props

  // Filter images based on search and category
  const filteredImages = images.filter(img => {
    const matchesSearch = img.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          img.topic?.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = !selectedCategory || img.topic === selectedCategory
    return matchesSearch && matchesCategory
  })

  // Group images by date
  const groupedImages = filteredImages.reduce((acc, img) => {
    const date = new Date(img.createdAt).toLocaleDateString('vi-VN')
    if (!acc[date]) {
      acc[date] = []
    }
    acc[date].push(img)
    return acc
  }, {} as Record<string, GeneratedImage[]>)

  return (
    <div className="flex h-full flex-col overflow-hidden bg-white border-r border-gray-200">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <h2 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
          <ImageIcon className="h-5 w-5 text-blue-600" />
          Lịch sử tạo ảnh
        </h2>
        
        {/* Search Box */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Tìm kiếm ảnh..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full border border-gray-300 rounded-lg py-2 pl-10 pr-4 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Category Filter */}
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg whitespace-nowrap transition-colors ${
              !selectedCategory 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Tất cả
          </button>
          {categories.map(cat => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.name)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg whitespace-nowrap transition-colors ${
                selectedCategory === cat.name
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {cat.icon} {cat.name}
            </button>
          ))}
        </div>
      </div>

      {/* Images List */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(groupedImages).length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center">
            <ImageIcon className="h-16 w-16 text-gray-300 mb-3" />
            <p className="text-gray-500 text-sm">Chưa có ảnh nào được tạo</p>
          </div>
        ) : (
          <div className="p-3 space-y-4">
            {Object.entries(groupedImages).map(([date, imgs]) => (
              <div key={date}>
                <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-gray-500">
                  <Calendar className="h-3 w-3" />
                  {date}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {imgs.map(img => (
                    <div
                      key={img.id}
                      onClick={() => onSelectImage(img.id)}
                      className={`relative group cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
                        selectedImage === img.id 
                          ? 'border-blue-600 shadow-lg' 
                          : 'border-gray-200 hover:border-blue-400'
                      }`}
                    >
                      <div className="aspect-square bg-gray-100">
                        <img
                          src={`data:image/png;base64,${img.image_base64}`}
                          alt={img.title}
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="absolute bottom-0 left-0 right-0 p-2">
                          <p className="text-white text-xs font-medium truncate">{img.title}</p>
                          {img.topic && (
                            <p className="text-white/80 text-xs truncate">{img.topic}</p>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onDeleteImage(img.id)
                        }}
                        className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="border-t border-gray-200 p-3 bg-gray-50">
        <p className="text-xs text-gray-600 text-center">
          Tổng: <span className="font-semibold text-gray-800">{images.length}</span> ảnh đã tạo
        </p>
      </div>
    </div>
  )
}
