import { Send, Loader2, Upload, Image as ImageIcon, X, History, Shapes, BookText, Undo2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"

interface DiagramHistoryItem {
  id: string
  title: string
  image_base64: string
  createdAt: Date
  prompt: string
}

interface InputAreaProps {
  onSubmit: (topic: string, uploadedImage?: string) => void
  loading: boolean
  progress: number
  statusMessage: string
  solution?: string
  diagrams: DiagramHistoryItem[]
  selectedDiagramIndex: number
  onSelectDiagram: (index: number) => void
}

export default function InputArea({ 
  onSubmit, 
  loading, 
  progress, 
  statusMessage,
  solution,
  diagrams,
  selectedDiagramIndex,
  onSelectDiagram
}: InputAreaProps) {
  const [activeTab, setActiveTab] = useState<"input-history" | "solution">("input-history")
  const [topic, setTopic] = useState("")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [displayedPrompt, setDisplayedPrompt] = useState("")
  const [isComposingPrompt, setIsComposingPrompt] = useState(true)
  const [pendingHistory, setPendingHistory] = useState<Array<{ id: string; prompt: string; createdAt: Date }>>([])
  const composerRef = useRef<HTMLTextAreaElement | null>(null)

  const sortedHistory = diagrams
    .map((diagram, index) => ({ diagram, index }))
    .sort((a, b) => new Date(b.diagram.createdAt).getTime() - new Date(a.diagram.createdAt).getTime())

  const diagramPromptSet = new Set(
    sortedHistory
      .map(({ diagram }) => diagram.prompt?.trim().toLowerCase())
      .filter((value): value is string => Boolean(value))
  )

  const visiblePendingHistory = pendingHistory.filter(
    (item) => !diagramPromptSet.has(item.prompt.trim().toLowerCase())
  )

  useEffect(() => {
    if (!isComposingPrompt || !composerRef.current) return

    composerRef.current.style.height = "auto"
    composerRef.current.style.height = `${Math.min(composerRef.current.scrollHeight, 180)}px`
  }, [topic, isComposingPrompt])

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64 = reader.result as string
        setUploadedImage(base64.split(',')[1]) // Remove data:image/...;base64, prefix
        setImagePreview(base64)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = () => {
    setUploadedImage(null)
    setImagePreview(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if ((topic.trim() || uploadedImage) && !loading) {
      const normalizedPrompt = topic.trim() || "Tạo hình học từ ảnh"

      setDisplayedPrompt(normalizedPrompt)
      setIsComposingPrompt(false)
      setPendingHistory((prev) => [
        {
          id: `${Date.now()}`,
          prompt: normalizedPrompt,
          createdAt: new Date(),
        },
        ...prev,
      ])

      onSubmit(topic.trim(), uploadedImage || undefined)
      setTopic("")
      setUploadedImage(null)
      setImagePreview(null)
    }
  }

  const handleStartNewPrompt = () => {
    setIsComposingPrompt(true)
    setTopic("")
    setUploadedImage(null)
    setImagePreview(null)
    setTimeout(() => composerRef.current?.focus(), 0)
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-white">
      <div className="grid grid-cols-2 border-b border-[#1f1f1f]">
        <button
          onClick={() => setActiveTab("input-history")}
          className={`border-r border-[#1f1f1f] px-4 py-3 text-sm font-semibold transition-colors ${
            activeTab === "input-history" ? "bg-[#1447E6] text-white" : "bg-white text-[#1f1f1f]"
          }`}
        >
          Nhập mô tả & Lịch sử
        </button>
        <button
          onClick={() => setActiveTab("solution")}
          className={`px-4 py-3 text-sm font-semibold transition-colors ${
            activeTab === "solution" ? "bg-[#1447E6] text-white" : "bg-white text-[#1f1f1f]"
          }`}
        >
          Lời giải chi tiết
        </button>
      </div>

      {activeTab === "input-history" ? (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="border-b border-[#1f1f1f] px-5 py-7">
            <div className="mb-5 grid grid-cols-[32px_1fr_32px] items-center text-[#1447E6]">
              <div></div>
              <div className="flex items-center justify-center gap-2">
                <h2 className="text-[25px] font-bold leading-none">Mô tả yêu cầu</h2>
                <Shapes className="h-6 w-6" />
              </div>
              <button
                type="button"
                onClick={handleStartNewPrompt}
                className="flex h-8 w-8 items-center justify-center rounded-full text-black transition-colors hover:bg-black/10"
                title="Nhập đề mới"
                aria-label="Nhập đề mới"
              >
                <Undo2 className="h-6 w-6" />
              </button>
            </div>

            {isComposingPrompt ? (
              <form onSubmit={handleSubmit} className="space-y-3">
                <div className="rounded-[40px] border border-[#b5b5b5] bg-white px-4 py-3 shadow-sm">
                  <div className="flex items-end gap-2">
                    <textarea
                      ref={composerRef}
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                      placeholder="Nhập mô tả bài toán hình học bạn muốn vẽ..."
                      className="min-h-[30px] max-h-[180px] w-full resize-none overflow-y-auto bg-transparent text-lg text-[#1f1f1f] placeholder:text-[#8a8a8a] focus:outline-none"
                      disabled={loading}
                      rows={1}
                    />
                    <button
                      type="submit"
                      disabled={loading || (!topic.trim() && !uploadedImage)}
                      className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[#1447E6] text-white transition-colors hover:bg-[#0f3bc4] disabled:cursor-not-allowed disabled:bg-[#8ea9ef]"
                      aria-label="Gửi yêu cầu"
                    >
                      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-center">
                  <label className="cursor-pointer">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      disabled={loading}
                    />
                    <div className="inline-flex items-center gap-2 bg-[#1447E6] px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-[#0f3bc4]">
                      <Upload className="h-4 w-4" />
                      Tải ảnh lên
                    </div>
                  </label>
                </div>
              </form>
            ) : (
              <div className="rounded border border-transparent px-1">
                <p className="whitespace-pre-wrap text-[38px] leading-snug text-[#1f1f1f]">
                  {displayedPrompt || "Chưa có đề bài gần nhất."}
                </p>
              </div>
            )}

            {imagePreview && (
              <div className="mt-4 inline-block rounded-md border border-[#c8c8c8] bg-white p-2">
                <div className="relative">
                  <img src={imagePreview} alt="Preview" className="max-h-40 w-auto object-contain" />
                  <button
                    onClick={removeImage}
                    className="absolute -right-2 -top-2 rounded-full bg-red-500 p-1 text-white transition-colors hover:bg-red-600"
                    aria-label="Xóa ảnh"
                    type="button"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <p className="mt-1 flex items-center gap-1 text-xs text-green-700">
                  <ImageIcon className="h-3.5 w-3.5" />
                  Đã tải ảnh lên.
                </p>
              </div>
            )}

            {loading && (
              <div className="mt-4 space-y-2">
                <div className="flex items-center gap-2 text-sm text-[#374151]">
                  <Loader2 className="h-4 w-4 animate-spin text-[#1447E6]" />
                  <span>{statusMessage || "Đang xử lý yêu cầu..."}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-[#d5d5d5]">
                  <div className="h-full bg-[#1447E6] transition-all duration-300" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="flex items-center justify-center gap-2 bg-[#1447E6] px-4 py-2 text-[21px] font-bold leading-none text-white">
              <History className="h-4 w-4" />
              Lịch sử
            </div>

            <div className="flex-1 overflow-y-auto p-3">
              {sortedHistory.length === 0 && visiblePendingHistory.length === 0 ? (
                <div className="p-4 text-sm text-[#606060]">Chưa có lịch sử vẽ.</div>
              ) : (
                <div className="space-y-2">
                  {visiblePendingHistory.map((item) => (
                    <div
                      key={item.id}
                      className="grid w-full grid-cols-[56px_1fr] items-center gap-2 bg-[rgba(233,233,233,0.40)] px-3 py-2 text-left"
                      title={item.prompt}
                    >
                      <span className="text-xl text-[#1f1f1f]">
                        {item.createdAt.toLocaleTimeString("vi-VN", {
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: false,
                        })}
                      </span>
                      <span className="truncate text-2xl text-[#1f1f1f]">{item.prompt}</span>
                    </div>
                  ))}

                  {sortedHistory.map(({ diagram, index }) => (
                    <button
                      key={diagram.id}
                      onClick={() => onSelectDiagram(index)}
                      className={`grid w-full grid-cols-[56px_1fr] items-center gap-2 px-3 py-2 text-left transition-colors ${
                        selectedDiagramIndex === index ? "bg-[#dfe7ff]" : "bg-[rgba(233,233,233,0.40)] hover:bg-[#e4e4e4]"
                      }`}
                      title={diagram.prompt}
                    >
                      <span className="text-xl text-[#1f1f1f]">
                        {new Date(diagram.createdAt).toLocaleTimeString("vi-VN", {
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: false,
                        })}
                      </span>
                      <span className="truncate text-2xl text-[#1f1f1f]">{diagram.prompt || diagram.title}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="h-full overflow-y-auto px-5 py-6 text-[#1f1f1f]">
          {solution ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-[#1447E6]">
                <BookText className="h-6 w-6" />
                <h3 className="text-3xl font-semibold">Lời giải chi tiết</h3>
              </div>
              <p className="whitespace-pre-wrap text-2xl leading-relaxed">{solution}</p>
            </div>
          ) : (
            <div className="mt-10 rounded border border-dashed border-[#b8b8b8] bg-white/60 p-6 text-center text-[#6b7280]">
              Chưa có lời giải. Hãy nhập đề bài và gửi yêu cầu để hiển thị lời giải chi tiết.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
