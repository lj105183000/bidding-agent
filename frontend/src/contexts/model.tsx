/* eslint-disable react-refresh/only-export-components -- 按设计规格: Provider/常量/hook 同文件 */
import { createContext, useContext, useState, type ReactNode } from 'react'

/** 真实 provider 列表（接口联调 spec §3：替换 mock 假名单，localStorage 持久化） */
export const MODEL_OPTIONS = ['deepseek', 'zhipu', 'vllm', 'ollama'] as const

export type ModelName = (typeof MODEL_OPTIONS)[number]

const STORAGE_KEY = 'chat_llm_provider'

function initModel(): ModelName {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v && (MODEL_OPTIONS as readonly string[]).includes(v)) {
      return v as ModelName
    }
  } catch {
    /* localStorage 不可用时回退默认 */
  }
  return MODEL_OPTIONS[0]
}

interface ModelContextValue {
  model: ModelName
  setModel: (m: ModelName) => void
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
  const [model, setModelState] = useState<ModelName>(initModel)

  const setModel = (m: ModelName) => {
    setModelState(m)
    try {
      localStorage.setItem(STORAGE_KEY, m)
    } catch {
      /* 忽略持久化失败 */
    }
  }

  return <ModelContext.Provider value={{ model, setModel }}>{children}</ModelContext.Provider>
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext)
  if (!ctx) throw new Error('useModel 必须在 <ModelProvider> 内使用')
  return ctx
}
