import { Send, Loader2, Ruler } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { Message } from "../types"

interface ChatAreaProps {
  messages: Message[]
  input: string
  setInput: (value: string) => void
  loading: boolean
  progress: number
  statusMessage: string
  onSubmit: (e: React.FormEvent) => void
}

export default function ChatArea({ 
  messages, 
  input, 
  setInput, 
  loading, 
  progress, 
  statusMessage, 
  onSubmit 
}: ChatAreaProps) {
  return (
    <div className="flex h-full flex-col bg-white">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-2xl text-center">
              <div className="mb-6 flex justify-center">
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6">
                  <Ruler className="h-16 w-16 text-blue-600" strokeWidth={1.5} />
                </div>
              </div>
              <h2 className="mb-3 text-xl font-semibold text-gray-800 whitespace-nowrap">
                Chào mừng bạn đến với hệ thống GeoMath
              </h2>
              <p className="mb-6 text-gray-600">
                Đưa ra các mô tả hình học 2D của bạn, tôi sẽ vẽ theo yêu cầu lập tức.
              </p>
              <div className="space-y-2 text-left">
                <button
                  onClick={() => setInput("Vẽ tam giác cân với chiều cao 8 đơn vị và đáy 6 đơn vị")}
                  className="w-full border border-gray-200 bg-gray-50 p-4 text-left text-sm text-gray-700 hover:border-blue-300 hover:bg-blue-50"
                >
                  Vẽ tam giác cân với chiều cao 8 đơn vị và đáy 6 đơn vị
                </button>
                <button
                  onClick={() => setInput("Vẽ hình tròn bán kính 5 cm")}
                  className="w-full border border-gray-200 bg-gray-50 p-4 text-left text-sm text-gray-700 hover:border-blue-300 hover:bg-blue-50"
                >
                  Vẽ hình tròn bán kính 5 cm
                </button>
                <button
                  onClick={() => setInput("Vẽ hình vuông cạnh 4 đơn vị")}
                  className="w-full border border-gray-200 bg-gray-50 p-4 text-left text-sm text-gray-700 hover:border-blue-300 hover:bg-blue-50"
                >
                  Vẽ hình vuông cạnh 4 đơn vị
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className="space-y-2">
                {msg.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="max-w-[80%] px-5 py-3 text-white shadow-sm" style={{ backgroundColor: '#3C5CB8' }}>
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <div className="flex-shrink-0">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-200 text-xs font-semibold text-gray-700">
                        GM
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="px-5 py-3" style={{ backgroundColor: '#E8F5E9' }}>
                        <div className="prose prose-sm max-w-none text-gray-800">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-3">
                <div className="flex-shrink-0">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-200 text-xs font-semibold text-gray-700">
                    GM
                  </div>
                </div>
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>{statusMessage || "Đang tạo hình..."}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">{progress}% hoàn thành</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={onSubmit}>
          <div className="relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Hãy nhập mô tả hình học bạn muốn vẽ..."
              className="w-full rounded-full border border-gray-300 bg-gray-50 px-5 py-3 pr-12 text-sm text-gray-900 placeholder-gray-500 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="absolute right-2 flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-md transition-all hover:from-blue-600 hover:to-blue-700 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
