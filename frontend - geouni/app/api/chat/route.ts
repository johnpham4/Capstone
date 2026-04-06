import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { message } = await request.json()

    if (!message) {
      return NextResponse.json({ error: "Tin nhắn trống" }, { status: 400 })
    }

    const backendUrl =
      process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

    // Gọi backend FastAPI
    const response = await fetch(
      `${backendUrl}/api/v1/diagrams/full-pipeline`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_input: message,
          max_tokens: 1024,
          temperature: 0.7,
        }),
      }
    )

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        {
          error: errorData.detail || `Backend error: ${response.statusText}`,
        },
        { status: response.status }
      )
    }

    const data = await response.json()

    // Trả về kết quả
    return NextResponse.json({
      response: `Đã vẽ hình: "${message}"`,
      diagram: {
        image_base64: data.image_base64,
        svg_content: data.svg_content,
        dsl: data.dsl,
        request_id: data.request_id,
      },
    })
  } catch (error) {
    console.error("API Error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Lỗi không xác định",
      },
      { status: 500 }
    )
  }
}
