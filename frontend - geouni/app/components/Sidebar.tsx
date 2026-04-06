import { Plus, MessageSquare, Search, ChevronRight, ChevronDown, MoreVertical, Edit3, Trash2, Folder, Pin } from "lucide-react"
import { Conversation, FolderType } from "../types"

interface SidebarProps {
  conversations: Conversation[]
  activeConversation: string | null
  folders: FolderType[]
  searchQuery: string
  setSearchQuery: (query: string) => void
  onNewConversation: () => void
  onSwitchConversation: (id: string) => void
  onDeleteConversation: (id: string) => void
  onRenameConversation: (id: string) => void
  onPinConversation: (id: string) => void
  onCreateFolder: () => void
  onDeleteFolder: (id: string) => void
  onStartRenamingFolder: (id: string) => void
  editingConvId: string | null
  editingConvValue: string
  setEditingConvValue: (value: string) => void
  onSaveConversationEdit: () => void
  onCancelConversationEdit: () => void
  editingFolderId: string | null
  editingFolderValue: string
  setEditingFolderValue: (value: string) => void
  onSaveFolderEdit: () => void
  onCancelFolderEdit: () => void
  openMenuId: string | null
  setOpenMenuId: (id: string | null) => void
  openFolderMenuId: string | null
  setOpenFolderMenuId: (id: string | null) => void
  menuPosition: { top: number; left: number } | null
  setMenuPosition: (pos: { top: number; left: number } | null) => void
  showFolderSubmenu: boolean
  setShowFolderSubmenu: (show: boolean) => void
  folderSubmenuPosition: { top: number; left: number } | null
  setFolderSubmenuPosition: (pos: { top: number; left: number } | null) => void
  selectedConvForMove: string | null
  setSelectedConvForMove: (id: string | null) => void
  expandedFolders: string[]
  setExpandedFolders: (folders: string[]) => void
  onAssignToFolder: (folderId: string) => void
}

export default function Sidebar(props: SidebarProps) {
  const {
    conversations,
    activeConversation,
    folders,
    searchQuery,
    setSearchQuery,
    onNewConversation,
    onSwitchConversation,
    onDeleteConversation,
    onRenameConversation,
    onPinConversation,
    onCreateFolder,
    onDeleteFolder,
    onStartRenamingFolder,
    editingConvId,
    editingConvValue,
    setEditingConvValue,
    onSaveConversationEdit,
    onCancelConversationEdit,
    editingFolderId,
    editingFolderValue,
    setEditingFolderValue,
    onSaveFolderEdit,
    onCancelFolderEdit,
    openMenuId,
    setOpenMenuId,
    openFolderMenuId,
    setOpenFolderMenuId,
    menuPosition,
    setMenuPosition,
    showFolderSubmenu,
    setShowFolderSubmenu,
    folderSubmenuPosition,
    setFolderSubmenuPosition,
    selectedConvForMove,
    setSelectedConvForMove,
    expandedFolders,
    setExpandedFolders,
    onAssignToFolder,
  } = props

  return (
    <div className="flex h-full flex-col overflow-visible">
      {/* New Conversation Button */}
      <div className="border-b border-gray-200 bg-white p-3">
        <button
          onClick={onNewConversation}
          className="flex w-full items-center justify-center gap-2 px-4 py-3 text-sm font-semibold text-white shadow-md transition-all hover:shadow-lg"
          style={{ backgroundColor: '#2563EB' }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1d4ed8'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#2563EB'}
        >
          <Plus className="h-5 w-5" />
          Tạo hội thoại mới
        </button>
      </div>

      {/* Search Box */}
      <div className="border-b border-gray-200 bg-white px-3 py-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm kiếm nội dung..."
            className="w-full border border-gray-300 bg-white pl-10 pr-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200 transition-colors"
          />
        </div>
      </div>

      {/* Scrollable Content: Folders + Conversations */}
      <div className="flex-1 overflow-y-auto overflow-x-visible">
        {/* Folders Section */}
        <div className="border-b border-gray-200 bg-white px-3 py-2">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Thư mục của bạn</h3>
            <button
              onClick={onCreateFolder}
              className="rounded p-1 hover:bg-gray-100 transition-colors"
              title="Tạo thư mục mới"
            >
              <Plus className="h-3.5 w-3.5 text-gray-500" />
            </button>
          </div>
          <div className="space-y-1">
            {folders.map((folder) => {
              const isExpanded = expandedFolders.includes(folder.id)
              const folderConversations = conversations
                .filter(conv => conv.folderId === folder.id)
                .sort((a, b) => {
                  if (a.pinned && !b.pinned) return -1
                  if (!a.pinned && b.pinned) return 1
                  return 0
                })
              
              return (
                <div key={folder.id} className="relative">
                  <button 
                    onClick={() => {
                      if (editingFolderId !== folder.id) {
                        setExpandedFolders(
                          expandedFolders.includes(folder.id) 
                            ? expandedFolders.filter(id => id !== folder.id)
                            : [...expandedFolders, folder.id]
                        )
                      }
                    }}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-gray-50 transition-colors"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    )}
                    <Folder className="h-4 w-4 flex-shrink-0" style={{ color: folder.color }} />
                    <div className="flex-1 overflow-hidden">
                      {editingFolderId === folder.id ? (
                        <input
                          type="text"
                          value={editingFolderValue}
                          onChange={(e) => setEditingFolderValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault()
                              onSaveFolderEdit()
                            }
                            if (e.key === 'Escape') {
                              e.preventDefault()
                              onCancelFolderEdit()
                            }
                          }}
                          onBlur={onSaveFolderEdit}
                          onFocus={(e) => e.target.select()}
                          onClick={(e) => e.stopPropagation()}
                          className="w-full rounded bg-white border border-blue-600 px-2 py-1 text-sm font-medium text-gray-900 focus:outline-none selection:bg-blue-500 selection:text-white"
                          autoFocus
                        />
                      ) : (
                        <span className="truncate text-gray-700">{folder.name}</span>
                      )}
                    </div>
                    {editingFolderId !== folder.id && (
                      <span className="ml-1 text-xs text-gray-400">({folderConversations.length})</span>
                    )}
                    {editingFolderId !== folder.id && (
                      <div className="w-8"></div>
                    )}
                  </button>
                  
                  {/* Conversations in Folder */}
                  {isExpanded && folderConversations.length > 0 && (
                    <div className="ml-6 mt-1 space-y-1">
                      {folderConversations.map((conv) => (
                        <div key={conv.id} className="relative">
                          <button
                            onClick={() => {
                              if (editingConvId !== conv.id) {
                                onSwitchConversation(conv.id)
                              }
                            }}
                            className={`flex w-full items-start gap-2 px-3 py-2 text-left transition-all ${
                              activeConversation === conv.id
                                ? "shadow-sm ring-1 ring-blue-100"
                                : "hover:bg-gray-50"
                            }`}
                            style={activeConversation === conv.id ? { backgroundColor: '#3C5CB8' } : {}}
                          >
                            <MessageSquare className={`mt-0.5 h-3.5 w-3.5 flex-shrink-0 ${
                              activeConversation === conv.id ? "text-white" : "text-gray-400"
                            }`} />
                            <div className="flex-1 overflow-hidden">
                              {editingConvId === conv.id ? (
                                <input
                                  type="text"
                                  value={editingConvValue}
                                  onChange={(e) => setEditingConvValue(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') {
                                      e.preventDefault()
                                      onSaveConversationEdit()
                                    }
                                    if (e.key === 'Escape') {
                                      e.preventDefault()
                                      onCancelConversationEdit()
                                    }
                                  }}
                                  onBlur={onSaveConversationEdit}
                                  onFocus={(e) => e.target.select()}
                                  onClick={(e) => e.stopPropagation()}
                                  className="w-full rounded bg-white border border-blue-600 px-2 py-0.5 text-xs font-medium text-gray-900 focus:outline-none selection:bg-blue-500 selection:text-white"
                                  autoFocus
                                />
                              ) : (
                                <div className="flex items-center gap-1">
                                  <p className={`truncate text-xs ${
                                    activeConversation === conv.id ? "font-medium text-white" : "font-normal text-gray-700"
                                  }`}>{conv.title}</p>
                                  {conv.pinned && (
                                    <Pin className="h-3 w-3 flex-shrink-0 text-red-500 fill-red-500" />
                                  )}
                                </div>
                              )}
                            </div>
                            {editingConvId !== conv.id && (
                              <div className="w-6"></div>
                            )}
                          </button>
                          
                          {/* Three-dot Menu Button for Folder Conversations */}
                          {editingConvId !== conv.id && (
                            <button
                              data-menu-button
                              onClick={(e) => {
                                e.stopPropagation()
                                if (openMenuId === conv.id) {
                                  setOpenMenuId(null)
                                  setMenuPosition(null)
                                } else {
                                  const rect = e.currentTarget.getBoundingClientRect()
                                  const menuHeight = 250
                                  const spaceBelow = window.innerHeight - rect.bottom
                                  const spaceAbove = rect.top
                                  
                                  let top = rect.top
                                  if (spaceBelow < menuHeight && spaceAbove > spaceBelow) {
                                    top = rect.bottom - menuHeight
                                  }
                                  
                                  setMenuPosition({ top, left: rect.right + 8 })
                                  setOpenMenuId(conv.id)
                                }
                              }}
                              className="absolute right-2 top-2 p-1 hover:bg-gray-200 transition-colors z-10"
                            >
                              <MoreVertical className="h-3.5 w-3.5 text-gray-500" />
                            </button>
                          )}
                          
                          {/* Dropdown Menu */}
                          {openMenuId === conv.id && menuPosition && (
                            <div
                              data-menu 
                              className="fixed z-[60] w-64 max-h-[80vh] overflow-y-auto bg-white shadow-lg ring-1 ring-black ring-opacity-5"
                              style={{ top: `${menuPosition.top}px`, left: `${menuPosition.left}px` }}
                            >
                              <div className="py-1">
                                <button
                                  onClick={() => onRenameConversation(conv.id)}
                                  className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                                >
                                  <Edit3 className="h-4 w-4" />
                                  Sửa tên hội thoại
                                </button>
                                <button
                                  onClick={() => onPinConversation(conv.id)}
                                  className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                                >
                                  <Pin className="h-4 w-4" />
                                  {conv.pinned ? 'Bỏ ghim hội thoại' : 'Pin hội thoại'}
                                </button>
                                <div className="my-1 border-t border-gray-100"></div>
                                <button
                                  onClick={() => onDeleteConversation(conv.id)}
                                  className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                                >
                                  <Trash2 className="h-4 w-4" />
                                  Xóa hội thoại
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Three-dot Menu Button for Folders */}
                  {editingFolderId !== folder.id && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setOpenFolderMenuId(openFolderMenuId === folder.id ? null : folder.id)
                      }}
                      className="absolute right-2 top-2 rounded p-1 hover:bg-gray-200 transition-colors z-10"
                    >
                      <MoreVertical className="h-3.5 w-3.5 text-gray-500" />
                    </button>
                  )}
                  
                  {/* Folder Dropdown Menu */}
                  {openFolderMenuId === folder.id && (
                    <div className="absolute right-2 top-10 z-50 w-48 bg-white shadow-lg ring-1 ring-black ring-opacity-5 overflow-hidden">
                      <div className="py-1">
                        <button
                          onClick={() => onStartRenamingFolder(folder.id)}
                          className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        >
                          <Edit3 className="h-4 w-4" />
                          Đổi tên thư mục
                        </button>
                        <button
                          onClick={() => onDeleteFolder(folder.id)}
                          className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                          Xóa thư mục
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Conversation List */}
        <div className="overflow-x-visible p-3">
          <h3 className="mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">Hội thoại của bạn</h3>
          {(() => {
            const filteredConversations = conversations.filter((conv) => {
              if (conv.folderId) return false
              if (!searchQuery.trim()) return true
              const query = searchQuery.toLowerCase()
              if (conv.title.toLowerCase().includes(query)) return true
              return conv.messages.some((msg) => 
                msg.content.toLowerCase().includes(query)
              )
            }).sort((a, b) => {
              if (a.pinned && !b.pinned) return -1
              if (!a.pinned && b.pinned) return 1
              return 0
            })
            
            if (conversations.length === 0) {
              return (
                <div className="px-3 py-8 text-center text-sm text-gray-400">
                  Chưa có hội thoại nào
                </div>
              )
            }
            
            if (filteredConversations.length === 0) {
              return (
                <div className="px-3 py-8 text-center text-sm text-gray-400">
                  Không tìm thấy kết quả
                </div>
              )
            }
            
            return filteredConversations.map((conv) => (
              <div key={conv.id} className="relative mb-2">
                <button
                  onClick={() => {
                    if (editingConvId !== conv.id) {
                      onSwitchConversation(conv.id)
                    }
                  }}
                  className={`flex w-full items-start gap-3 px-4 py-3 text-left transition-all ${
                    activeConversation === conv.id
                      ? "bg-blue-50 shadow-sm ring-1 ring-blue-100"
                      : "hover:bg-[#EDF2F2]"
                  }`}
                >
                  <MessageSquare className={`mt-0.5 h-4 w-4 flex-shrink-0 ${
                    activeConversation === conv.id ? "text-blue-600" : "text-gray-400"
                  }`} />
                  <div className="flex-1 overflow-hidden">
                    {editingConvId === conv.id ? (
                      <input
                        type="text"
                        value={editingConvValue}
                        onChange={(e) => setEditingConvValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault()
                            onSaveConversationEdit()
                          }
                          if (e.key === 'Escape') {
                            e.preventDefault()
                            onCancelConversationEdit()
                          }
                        }}
                        onBlur={onSaveConversationEdit}
                        onFocus={(e) => e.target.select()}
                        onClick={(e) => e.stopPropagation()}
                        className="w-full rounded bg-white border border-blue-600 px-2 py-1 text-[15px] font-medium text-gray-900 focus:outline-none selection:bg-blue-500 selection:text-white"
                        autoFocus
                      />
                    ) : (
                      <div className="flex items-center gap-1.5">
                        <p className={`truncate text-sm ${
                          activeConversation === conv.id ? "font-medium text-blue-700" : "font-normal text-gray-700"
                        }`}>{conv.title}</p>
                        {conv.pinned && (
                          <Pin className="h-3.5 w-3.5 flex-shrink-0 text-red-500 fill-red-500" />
                        )}
                      </div>
                    )}
                    <p className="mt-0.5 text-xs text-gray-500">
                      {conv.diagrams.length} hình vẽ
                    </p>
                  </div>
                  {editingConvId !== conv.id && (
                    <div className="ml-2 w-8"></div>
                  )}
                </button>
                
                {/* Three-dot Menu Button */}
                {editingConvId !== conv.id && (
                  <button
                    data-menu-button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (openMenuId === conv.id) {
                        setOpenMenuId(null)
                        setMenuPosition(null)
                      } else {
                        const rect = e.currentTarget.getBoundingClientRect()
                        const menuHeight = 300
                        const spaceBelow = window.innerHeight - rect.bottom
                        const spaceAbove = rect.top
                        
                        let top = rect.top
                        if (spaceBelow < menuHeight && spaceAbove > spaceBelow) {
                          top = rect.bottom - menuHeight
                        }
                        
                        setMenuPosition({ top, left: rect.right + 8 })
                        setOpenMenuId(conv.id)
                      }
                    }}
                    className="absolute right-3 top-3 p-1.5 hover:bg-gray-200 transition-colors z-10"
                  >
                    <MoreVertical className="h-4 w-4 text-gray-500" />
                  </button>
                )}
                
                {/* Dropdown Menu */}
                {openMenuId === conv.id && menuPosition && (
                  <div
                    data-menu 
                    className="fixed z-[60] w-64 max-h-[80vh] overflow-y-auto bg-white shadow-lg ring-1 ring-black ring-opacity-5"
                    style={{ top: `${menuPosition.top}px`, left: `${menuPosition.left}px` }}
                  >
                    <div className="py-1">
                      <button
                        onClick={() => onRenameConversation(conv.id)}
                        className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <Edit3 className="h-4 w-4" />
                        Sửa tên hội thoại
                      </button>
                      
                      {/* Move to Folder with Submenu */}
                      <button
                        data-menu-button
                        onClick={(e) => {
                          e.stopPropagation()
                          if (showFolderSubmenu && selectedConvForMove === conv.id) {
                            setShowFolderSubmenu(false)
                            setFolderSubmenuPosition(null)
                          } else {
                            const rect = e.currentTarget.getBoundingClientRect()
                            setFolderSubmenuPosition({
                              top: rect.top,
                              left: rect.right + 8
                            })
                            setShowFolderSubmenu(true)
                            setSelectedConvForMove(conv.id)
                          }
                        }}
                        className="flex w-full items-center justify-between px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <Folder className="h-4 w-4 flex-shrink-0" />
                          <span className="whitespace-nowrap">Di chuyển vào thư mục</span>
                        </div>
                        <ChevronRight className="h-4 w-4 flex-shrink-0" />
                      </button>
                      
                      <button
                        onClick={() => onPinConversation(conv.id)}
                        className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <Pin className="h-4 w-4" />
                        {conv.pinned ? 'Bỏ ghim hội thoại' : 'Pin hội thoại'}
                      </button>
                      <div className="my-1 border-t border-gray-100"></div>
                      <button
                        onClick={() => onDeleteConversation(conv.id)}
                        className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <Trash2 className="h-4 w-4" />
                        Xóa hội thoại
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))
          })()}
        </div>
      </div>

      {/* Statistics Section */}
      <div className="border-t border-gray-200 bg-white px-4" style={{ height: '82px' }}>
        <div className="flex h-full items-center">
          <div className="flex w-full flex-col gap-1.5 text-center">
            <div className="flex items-center justify-center gap-2 text-sm">
              <span className="text-gray-500">Tổng hội thoại</span>
              <span className="font-semibold text-gray-700">{conversations.length}</span>
            </div>
            <div className="flex items-center justify-center gap-2 text-sm">
              <span className="text-gray-500">Tổng hình vẽ</span>
              <span className="font-semibold text-gray-700">
                {conversations.reduce((sum, conv) => sum + conv.diagrams.length, 0)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Folder Submenu */}
      {showFolderSubmenu && folderSubmenuPosition && selectedConvForMove && (
        <div
          data-menu
          className="fixed z-[70] w-56 bg-white shadow-lg ring-1 ring-black ring-opacity-5"
          style={{ top: `${folderSubmenuPosition.top}px`, left: `${folderSubmenuPosition.left}px` }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="py-1">
            {folders.map((folder) => (
              <button
                key={folder.id}
                onClick={() => {
                  onAssignToFolder(folder.id)
                  setShowFolderSubmenu(false)
                  setFolderSubmenuPosition(null)
                }}
                className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
              >
                <Folder className="h-4 w-4 flex-shrink-0" style={{ color: folder.color }} />
                <span>{folder.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
