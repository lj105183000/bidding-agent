import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Layout from './components/Layout'
import { ToastProvider } from './components/Toast'
import { ModelProvider } from './contexts/model'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import Extract from './pages/Extract'
import PriceCompare from './pages/PriceCompare'
import Material from './pages/Material'
import Analysis from './pages/Analysis'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'chat', element: <Chat /> },
      { path: 'extract', element: <Extract /> },
      { path: 'price', element: <PriceCompare /> },
      { path: 'material', element: <Material /> },
      { path: 'analysis', element: <Analysis /> },
    ],
  },
])

export default function App() {
  return (
    <ModelProvider>
      <ToastProvider>
        <RouterProvider router={router} />
      </ToastProvider>
    </ModelProvider>
  )
}
