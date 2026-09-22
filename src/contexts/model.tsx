/* eslint-disable react-refresh/only-export-components -- 按设计规格: Provider/常量/hook 同文件 */
import { createContext, useContext, useState, type ReactNode } from 'react'

export const MODEL_OPTIONS = ['GPT-4o', 'Claude-3.5', '通义千问']

interface ModelContextValue {
  model: string
  setModel: (m: string) => void
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState(MODEL_OPTIONS[0])
  return <ModelContext.Provider value={{ model, setModel }}>{children}</ModelContext.Provider>
}

export function useModel(): ModelContextValue {
  const ctx = useContext(ModelContext)
  if (!ctx) throw new Error('useModel 必须在 <ModelProvider> 内使用')
  return ctx
}
