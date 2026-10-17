import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    // host: true 让 dev server 同时绑定 IPv4 (0.0.0.0) 与 IPv6 (::)。
    //
    // ★ 为什么必须显式设置（实测踩坑）
    // ---------------------------------
    // 不设此项时 Vite 只绑 `localhost`，在 Windows 上解析为 IPv6，
    // 实际监听的是 `[::1]:5173` —— 于是：
    //
    //     http://localhost:5173/    → 200  ✅
    //     http://[::1]:5173/        → 200  ✅
    //     http://127.0.0.1:5173/    → 连不上 ❌
    //
    // 用户习惯输入 127.0.0.1，会直接看到"无法访问此网站"；
    // 而所有用 `localhost` 做的健康检查都显示正常 —— 排查时极易误判为
    // "服务没起来"，实际服务跑得好好的。
    // 绑到全部接口后三种写法都能打开。
    host: true,
    port: 5173,
    // 端口被占用时直接失败，而不是自动换到 5174：
    // 本项目文档、启动脚本、后端 CORS 白名单都写死了 5173，
    // 悄悄换端口会让"前端打开了但接口全挂"变成更难查的问题。
    strictPort: true,
  },

  build: {
    rollupOptions: {
      output: {
        // vendor 拆包: 控制单 chunk 体积
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
})
