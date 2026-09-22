import { useEffect, useState } from 'react'

/**
 * 逐字打字效果: 每 tick 输出 1 字, charsPerSec 字/秒
 * text 变化时从头开始
 */
export function useTypewriter(text: string, charsPerSec = 30) {
  const [count, setCount] = useState(0)
  const done = count >= text.length

  useEffect(() => {
    setCount(0)
    if (!text) return
    const timer = setInterval(() => {
      setCount((c) => {
        if (c >= text.length) {
          clearInterval(timer)
          return c
        }
        return c + 1
      })
    }, 1000 / charsPerSec)
    return () => clearInterval(timer)
  }, [text, charsPerSec])

  return { typed: text.slice(0, count), done }
}
