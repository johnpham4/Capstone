export type GeneratedImage = {
  id: string
  title: string
  image_base64: string
  dsl?: string
  topic?: string
  createdAt: Date
  problem?: string
}

export type ImageCategory = {
  id: string
  name: string
  icon: string
  color: string
}

export type Message = {
  role: 'user' | 'assistant'
  content: string
  timestamp?: Date
}

export type Conversation = {
  id: string
  title: string
  messages: Message[]
  diagrams: GeneratedImage[]
  pinned: boolean
  folderId?: string
  createdAt: Date
}

export type FolderType = {
  id: string
  name: string
  color: string
  createdAt?: Date
}
