import { useState, useEffect, useRef } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"
import TextareaAutosize from "react-textarea-autosize"
import { Send, Plus, MessageSquare, Trash2, UploadIcon, ChevronDown, ChevronUp } from "lucide-react"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
} from "@/components/ui/input-group"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { askQuestion, getChats, getChat, deleteChat, searchSources, type Chat, type ChatMessage } from "@/lib/api"
import { ModeToggle } from "@/components/theme/mode-toggle"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import { Upload } from "./upload"

type PageType = "chat" | "upload"

function MessageInput({ onSend, disabled }: { onSend: (text: string) => void; disabled: boolean }) {
  const [text, setText] = useState("")

  function send() {
    const value = text.trim()
    if (!value || disabled) return
    onSend(value)
    setText("")
  }

  return (
    <div className="border-t p-4">
      <InputGroup>
        <TextareaAutosize
          data-slot="input-group-control"
          className="max-h-[200px] min-h-16 w-full min-w-0 resize-none whitespace-pre-wrap wrap-break-word rounded-md bg-transparent px-3 py-2.5 text-base outline-none placeholder:text-muted-foreground md:text-sm"
          placeholder="Ask a question about your documents..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          disabled={disabled}
        />
        <InputGroupAddon align="block-end">
          <InputGroupButton
            className="ml-auto"
            size="sm"
            variant="default"
            onClick={send}
            disabled={!text.trim() || disabled}
          >
            <Send />
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
    </div>
  )
}

function AppSidebar({
  chats,
  currentChatId,
  currentPage,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onSelectPage,
  isLoading,
}: {
  chats: Chat[]
  currentChatId: string | null
  currentPage: PageType
  onSelectChat: (chatId: string | null) => void
  onNewChat: () => void
  onDeleteChat: (chatId: string) => void
  onSelectPage: (page: PageType) => void
  isLoading: boolean
}) {
  return (
    <Sidebar>
      <SidebarHeader className="flex flex-row items-center justify-between p-4">
        <h2 className="text-lg font-semibold">RAG API</h2>
        <ModeToggle />
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                isActive={currentPage === "chat"}
                onClick={() => onSelectPage("chat")}
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                Chat
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton
                isActive={currentPage === "upload"}
                onClick={() => onSelectPage("upload")}
              >
                <UploadIcon className="mr-2 h-4 w-4" />
                Upload
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>
        <Separator />
        {currentPage === "chat" && (
          <>
            <div className="px-2 py-2">
              <Button onClick={onNewChat} className="w-full justify-start gap-2" variant="outline">
                <Plus className="h-4 w-4" />
                New Chat
              </Button>
            </div>
            <Separator />
            <SidebarGroup>
              <SidebarGroupLabel>Chat History</SidebarGroupLabel>
              <ScrollArea className="h-[calc(100vh-280px)]">
                <SidebarMenu>
                  {isLoading ? (
                    <div className="p-4 text-center text-sm text-muted-foreground">Loading...</div>
                  ) : chats.length === 0 ? (
                    <div className="p-4 text-center text-sm text-muted-foreground">
                      No chats yet
                    </div>
                  ) : (
                    chats.map((chat) => (
                      <SidebarMenuItem key={chat.chat_id} className="group">
                        <div className="flex w-full items-center">
                          <SidebarMenuButton
                            isActive={currentChatId === chat.chat_id}
                            onClick={() => onSelectChat(chat.chat_id)}
                            className="flex-1 truncate pr-2"
                          >
                            <MessageSquare className="mr-2 h-4 w-4 shrink-0" />
                            <span className="truncate">{chat.title || "New Chat"}</span>
                          </SidebarMenuButton>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 shrink-0 opacity-0 group-hover:opacity-100"
                            onClick={(e) => {
                              e.stopPropagation()
                              toast("Delete this chat?", {
                                description: "This action cannot be undone.",
                                action: {
                                  label: "Delete",
                                  onClick: () => onDeleteChat(chat.chat_id),
                                },
                                cancel: {
                                  label: "Cancel",
                                  onClick: () => {},
                                },
                              })
                            }}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </SidebarMenuItem>
                    ))
                  )}
                </SidebarMenu>
              </ScrollArea>
            </SidebarGroup>
          </>
        )}
      </SidebarContent>
    </Sidebar>
  )
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    if (typeof window === "undefined") return
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)")
    setReduced(mq.matches)
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches)
    mq.addEventListener?.("change", handler)
    return () => mq.removeEventListener?.("change", handler)
  }, [])
  return reduced
}

function SourcesBlock({
  sources,
  status,
  isExpanded,
  onToggle,
}: {
  sources: Array<{ document_id: string; file_path?: string; count?: number; original_name?: string }>
  status: "reading" | "read" | "hidden"
  isExpanded: boolean
  onToggle: () => void
}) {
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  if (status === "hidden") return null
  const files = (sources || []).filter(Boolean)
  if (files.length === 0 && status === "read") return null
  if (files.length === 0 && status === "reading") {
    return (
      <div className="flex w-full justify-start mb-2">
        <div className="max-w-[80%] md:max-w-[70%] w-full">
          <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <span className="animate-pulse">Reading files...</span>
            </div>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="flex w-full justify-start mb-2">
      <div className="max-w-[80%] md:max-w-[70%] w-full">
        <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
          {status === "reading" ? (
            <div className="flex items-center gap-2">
              <span className="animate-pulse">Reading files...</span>
            </div>
          ) : (
            <div>
              <button onClick={onToggle} className="flex w-full items-center justify-between gap-2 hover:text-foreground">
                <span>Read {files.length} file{files.length === 1 ? "" : "s"}</span>
                {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </button>
              {isExpanded && (
                <div className="mt-2 space-y-1">
                      {files.map((f, idx) => {
                    const docId = f.document_id || (f.file_path ? f.file_path.split('/').pop() : undefined)
                    const name = f.original_name || docId || (f.file_path ? f.file_path.split('/').pop() : "")
                    return (
                      <div key={idx} className="flex items-center">
                        {docId ? (
                          <a
                            href={`${API_BASE_URL}/documents/${encodeURIComponent(docId)}/raw`}
                            target="_blank"
                            rel="noreferrer"
                            className="truncate hover:underline text-foreground/80"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {name}
                          </a>
                        ) : (
                          <span className="truncate">{name}</span>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function RevealingMessage({ fullContent, messageId }: { fullContent: string; messageId?: string }) {
  const reduced = usePrefersReducedMotion()
  const key = messageId || fullContent
  const [hasRevealed, setHasRevealed] = useState(() => {
    try {
      if (typeof window === "undefined") return false
      return localStorage.getItem(`revealed:${key}`) === "1"
    } catch {
      return false
    }
  })
  const [revealed, setRevealed] = useState(reduced || hasRevealed ? fullContent : "")
  const bufferRef = useRef(fullContent)
  const revealedRef = useRef(reduced || hasRevealed ? fullContent : "")
  const queueRef = useRef<string[]>([])
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    bufferRef.current = fullContent
  }, [fullContent])

  useEffect(() => {
    if (reduced || hasRevealed) {
      setRevealed(fullContent)
      revealedRef.current = fullContent
      queueRef.current = []
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
      try {
        if (typeof window !== "undefined" && messageId) {
          localStorage.setItem(`revealed:${messageId}`, "1")
        }
      } catch {}
      return
    }
    if (fullContent && revealedRef.current.length === 0 && fullContent.length > 0) {
      const tokens = fullContent.match(/\S+|\s+/g)
      if (tokens) {
        queueRef.current = [...tokens]
      } else {
        queueRef.current = [fullContent]
      }
    } else if (fullContent.length > revealedRef.current.length) {
      const newText = fullContent.slice(revealedRef.current.length)
      const tokens = newText.match(/\S+|\s+/g)
      if (tokens) {
        queueRef.current.push(...tokens)
      } else if (newText) {
        queueRef.current.push(newText)
      }
    } else if (fullContent.length === 0) {
      queueRef.current = []
      revealedRef.current = ""
      setRevealed("")
    }
  }, [fullContent, reduced, hasRevealed, messageId])

  useEffect(() => {
    if (reduced || hasRevealed) return
    if (timerRef.current) return
    timerRef.current = window.setInterval(() => {
      if (queueRef.current.length === 0) {
        if (revealedRef.current.length < fullContent.length) {
          revealedRef.current = fullContent
          setRevealed(fullContent)
          queueRef.current = []
          if (timerRef.current) {
            window.clearInterval(timerRef.current)
            timerRef.current = null
          }
          try {
            if (typeof window !== "undefined" && messageId) {
              localStorage.setItem(`revealed:${messageId}`, "1")
            }
          } catch {}
          return
        }
        if (timerRef.current) {
          window.clearInterval(timerRef.current)
          timerRef.current = null
        }
        try {
          if (typeof window !== "undefined" && messageId && revealedRef.current.length > 0) {
            localStorage.setItem(`revealed:${messageId}`, "1")
          }
        } catch {}
        return
      }
      const batch = queueRef.current.length > 80 ? 4 : queueRef.current.length > 40 ? 3 : queueRef.current.length > 20 ? 2 : 1
      for (let i = 0; i < batch && queueRef.current.length > 0; i++) {
        const token = queueRef.current.shift()!
        revealedRef.current += token
      }
      setRevealed(revealedRef.current)
    }, 25)
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [fullContent, reduced, hasRevealed, messageId])

  useEffect(() => {
    if (!reduced && !hasRevealed && revealedRef.current.length === fullContent.length && fullContent.length > 0) {
      setHasRevealed(true)
      try {
        if (typeof window !== "undefined" && messageId) {
          localStorage.setItem(`revealed:${messageId}`, "1")
        }
      } catch {}
    }
  }, [revealed, fullContent, reduced, hasRevealed, messageId])

  if (reduced || hasRevealed) {
    return <Markdown remarkPlugins={[remarkGfm]}>{fullContent}</Markdown>
  }
  return <Markdown remarkPlugins={[remarkGfm]}>{revealed || ""}</Markdown>
}

function MessageBubble({ message, messageKey }: { message: ChatMessage & { sources?: any[], sourcesStatus?: string, sourcesExpanded?: boolean, onToggleSources?: () => void }, messageKey?: string }) {
  const isUser = message.role === "user"
  if (isUser) {
    return (
      <div className={cn("flex w-full", "justify-end")}>
        <div
          className={cn(
            "max-w-[80%] rounded-lg px-4 py-2 text-sm md:max-w-[70%] break-words",
            "bg-primary text-primary-foreground whitespace-pre-wrap"
          )}
        >
          {message.content}
        </div>
      </div>
    )
  }
  const showSources = message.sourcesStatus && message.sourcesStatus !== "hidden" && Array.isArray(message.sources) && message.sources.length > 0
  return (
    <div className="flex w-full justify-start flex-col items-start">
      {showSources && (
        <SourcesBlock
          sources={message.sources || []}
          status={message.sourcesStatus as any}
          isExpanded={!!message.sourcesExpanded}
          onToggle={() => message.onToggleSources?.()}
        />
      )}
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-2 text-sm md:max-w-[70%] break-words",
          "bg-muted text-muted-foreground markdown"
        )}
      >
        <RevealingMessage fullContent={message.content} messageId={messageKey} />
      </div>
    </div>
  )
}

export function ChatApp() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentPage, setCurrentPage] = useState<PageType>("chat")
  const [isLoadingChats, setIsLoadingChats] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [isLoadingMessages, setIsLoadingMessages] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    const scrollArea = document.querySelector('[data-slot="scroll-area-viewport"]')
    if (scrollArea) {
      const { scrollTop, scrollHeight, clientHeight } = scrollArea as HTMLElement
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      if (isNearBottom) {
        scrollToBottom()
      }
    } else {
      scrollToBottom()
    }
  }, [messages])

  const loadChats = async () => {
    setIsLoadingChats(true)
    try {
      const res = await getChats()
      setChats(res.chats)
    } catch (error) {
      console.error("Failed to load chats:", error)
    } finally {
      setIsLoadingChats(false)
    }
  }

  useEffect(() => {
    loadChats()
  }, [])

  const loadChat = async (chatId: string) => {
    setIsLoadingMessages(true)
    try {
      const res = await getChat(chatId)
      const loaded = (res.messages || []).map((m: any, idx: number) => {
        if (m.role === "assistant") {
          // Try to restore sources if stored
          return {
            ...m,
            sourcesStatus: m.sources && Array.isArray(m.sources) && m.sources.length > 0 ? "read" : "hidden",
            sourcesExpanded: false,
            sources: m.sources || [],
          }
        }
        return {
          ...m,
          sourcesStatus: "hidden",
          sourcesExpanded: false,
        }
      })
      setMessages(loaded)
      setCurrentChatId(chatId)
    } catch (error) {
      console.error("Failed to load chat:", error)
    } finally {
      setIsLoadingMessages(false)
    }
  }

  const handleNewChat = () => {
    setCurrentChatId(null)
    setMessages([])
  }

  const handleSelectChat = (chatId: string | null) => {
    if (chatId === null) {
      handleNewChat()
    } else {
      loadChat(chatId)
    }
  }

  const handleDeleteChat = async (chatId: string) => {
    try {
      await deleteChat(chatId)
      if (currentChatId === chatId) {
        setCurrentChatId(null)
        setMessages([])
      }
      await loadChats()
    } catch (error) {
      console.error("Failed to delete chat:", error)
    }
  }

  const handleSelectPage = (page: PageType) => {
    setCurrentPage(page)
  }

  const handleSend = async (text: string) => {
    const userMessage: ChatMessage = { role: "user", content: text }
    setMessages((prev) => [...prev, userMessage])
    setIsSending(true)

      // Create placeholder assistant message with sources state
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "",
          sources: [],
          sourcesStatus: "reading",
          sourcesExpanded: false,
        } as any,
      ])

    try {
      // Step 1: search to get sources first
      let sourcesData: any = { sources: [], chunks: [] }
      try {
        sourcesData = await searchSources(text, 5)
      } catch (e) {
        // Fallback to getting from answer if search fails
        sourcesData = { sources: [], chunks: [] }
      }
      const sources = sourcesData.sources || []

      if (sources.length === 0) {
        // No relevant files - don't show sources block
        setMessages((prev) => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
            updated[lastIdx] = {
              ...updated[lastIdx],
              sources: [],
              sourcesStatus: "hidden",
              sourcesExpanded: false,
            } as any
          }
          return updated
        })
        // Small pause or skip - but requirement says show sources only if exist; if none, don't show
        await new Promise((r) => setTimeout(r, 400))
      } else {
        // Show reading state, then reveal files one by one
        setMessages((prev) => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
            updated[lastIdx] = {
              ...updated[lastIdx],
              sources: [],
              sourcesStatus: "reading",
            } as any
          }
          return updated
        })

        // Reveal files one by one with small delays
        for (let i = 0; i < sources.length; i++) {
          await new Promise((r) => setTimeout(r, 150))
          setMessages((prev) => {
            const updated = [...prev]
            const lastIdx = updated.length - 1
            if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
              updated[lastIdx] = {
                ...updated[lastIdx],
                sources: sources.slice(0, i + 1),
                sourcesStatus: "reading",
              } as any
            }
            return updated
          })
        }

        // Mark as read
        setMessages((prev) => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
            updated[lastIdx] = {
              ...updated[lastIdx],
              sources: sources,
              sourcesStatus: "read",
              sourcesExpanded: false,
            } as any
          }
          return updated
        })

        // Pause 1-2 seconds so user can register
        const pause = Math.max(1000, Math.min(2000, 1200 + (sources.length * 50)))
        await new Promise((r) => setTimeout(r, pause))
      }

      // Step 2: get answer
      const res = await askQuestion(text, currentChatId)
      if (!currentChatId || currentChatId !== res.chat_id) {
        setCurrentChatId(res.chat_id)
      }
      await loadChats()

      // Collapse sources into compact summary (already in read state, user can expand)
      setMessages((prev) => {
        const updated = [...prev]
        const lastIdx = updated.length - 1
        if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
          const finalSources = sources.length > 0 ? sources : (res.sources || [])
          updated[lastIdx] = {
            ...updated[lastIdx],
            content: res.answer,
            sources: finalSources,
            sourcesStatus: finalSources.length > 0 ? "read" : "hidden",
            sourcesExpanded: false,
          } as any
        }
        return updated
      })
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
      } as any
      setMessages((prev) => {
        // Replace last assistant placeholder if exists
        const updated = [...prev]
        const lastIdx = updated.length - 1
        if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
          updated[lastIdx] = errorMessage
          return updated
        }
        return [...updated, errorMessage]
      })
    } finally {
      setIsSending(false)
    }
  }

  return (
    <SidebarProvider className="h-full">
      <AppSidebar
        chats={chats}
        currentChatId={currentChatId}
        currentPage={currentPage}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onSelectPage={handleSelectPage}
        isLoading={isLoadingChats}
      />
      <main className="flex flex-1 flex-col h-full min-h-0">
        <div className="flex h-14 items-center justify-between border-b px-4">
          <div className="flex items-center gap-2">
            <SidebarTrigger />
            <h1 className="text-lg font-semibold">
              {currentPage === "chat"
                ? currentChatId
                  ? chats.find((c) => c.chat_id === currentChatId)?.title || "Chat"
                  : "New Chat"
                : "Upload Documents"}
            </h1>
          </div>
        </div>
        {currentPage === "chat" ? (
          <div className="flex flex-1 flex-col min-h-0 overflow-hidden">
            <ScrollArea className="flex-1 min-h-0 overflow-y-auto">
              <div className="mx-auto max-w-4xl space-y-4 p-4">
                {messages.length === 0 && !isLoadingMessages ? (
                  <div className="flex h-full min-h-0 flex-col items-center justify-center text-center py-8">
                    <MessageSquare className="text-muted-foreground mb-4 h-12 w-12" />
                    <h2 className="text-xl font-semibold">Start a conversation</h2>
                    <p className="text-muted-foreground mt-2 max-w-md text-sm">
                      Ask questions about your uploaded documents. Upload files first to get started!
                    </p>
                  </div>
                ) : (
                  messages.map((message, idx) => (
                    <MessageBubble
                      key={`${currentChatId || "new"}-${idx}`}
                      messageKey={`${currentChatId || "new"}-${idx}`}
                      message={{
                        ...message,
                        onToggleSources: () => {
                          setMessages((prev) => {
                            const updated = [...prev]
                            if (updated[idx] && updated[idx].role === "assistant") {
                              updated[idx] = {
                                ...updated[idx],
                                sourcesExpanded: !updated[idx].sourcesExpanded,
                              } as any
                            }
                            return updated
                          })
                        },
                      }}
                    />
                  ))
                )}
                {isSending && messages.length > 0 && messages[messages.length - 1]?.role === "assistant" && messages[messages.length - 1]?.content ? null : isSending ? (
                  <div className="flex justify-start">
                    <div className="bg-muted max-w-[80%] rounded-lg px-4 py-2 text-sm">
                      Thinking...
                    </div>
                  </div>
                ) : null}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
            <div className="flex-shrink-0">
              <MessageInput onSend={handleSend} disabled={isSending} />
            </div>
          </div>
        ) : (
          <Upload />
        )}
      </main>
    </SidebarProvider>
  )
}

export default ChatApp
