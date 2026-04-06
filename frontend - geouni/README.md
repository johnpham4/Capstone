This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## 📁 Project Structure

```
app/
├── page.tsx                    # 🏠 Landing Page (Home page)
├── chat/
│   └── page.tsx               # 💬 Main Chat page (drawing application)
├── components/
│   ├── Header.tsx             # Navigation bar & action buttons
│   ├── Sidebar.tsx            # Conversation list & folders
│   ├── ChatArea.tsx           # Chat area & input form
│   └── DiagramViewer.tsx      # Diagram display & zoom controls
├── types/
│   └── index.ts               # TypeScript type definitions
├── api/
│   └── chat/
│       └── route.ts           # API routes
├── layout.tsx                 # Root layout
└── globals.css                # Global styles
```

### 🎯 Purpose of Each File

**Main Pages:**
- `page.tsx` - Landing page with hero section, features, and call-to-action
- `chat/page.tsx` - Main chat application page with all drawing logic

**Components:**
- `Header.tsx` - Navigation bar with logo, sidebar toggle, download & share
- `Sidebar.tsx` - Manages conversations, folders, search, and statistics
- `ChatArea.tsx` - Chat interface with message display and input form
- `DiagramViewer.tsx` - Diagram display with zoom and gallery thumbnails

**Types:**
- `types/index.ts` - Shared TypeScript interfaces (Message, Conversation, FolderType)

### 🔗 Routes

- `/` - Landing page (introduction home page)
- `/chat` - Main chat page (drawing application)

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the pages:
- **Landing page**: Modify `app/page.tsx` 
- **Chat application**: Modify `app/chat/page.tsx`
- **Components**: Edit files in `app/components/`

The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## ✨ Features

- 🎨 **AI-Powered Geometry Drawing** - Automatically draw 2D geometry from natural language descriptions
- 💬 **Chat Interface** - Friendly and easy-to-use chat interface
- 📁 **Conversation Management** - Manage conversations with folders and pinning
- 🔍 **Search & Filter** - Search content within conversations
- 🖼️ **Diagram Gallery** - View and manage multiple diagrams
- 🔎 **Zoom Controls** - Zoom in/out diagrams with mouse wheel
- 💾 **Export** - Download diagrams as PNG files
- 📱 **Responsive Design** - Optimized for all screen sizes

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Markdown**: React Markdown

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
