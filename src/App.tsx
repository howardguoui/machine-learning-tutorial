import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LangProvider } from './context/LangContext'
import { ThemeProvider } from './context/ThemeContext'
import { AppLayout } from './components/Layout/AppLayout'
import { HomePage } from './pages/HomePage'
import { TopicPage } from './pages/TopicPage'
import { PythonPlaygroundPage } from './pages/PythonPlaygroundPage'
import { LiveDemoPage } from './pages/LiveDemoPage'
import { ResourcesPage } from './pages/ResourcesPage'

export default function App() {
  return (
    <ThemeProvider>
      <LangProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/learn/:topicId" element={<TopicPage />} />
              <Route path="/playground" element={<PythonPlaygroundPage />} />
              <Route path="/demo" element={<LiveDemoPage />} />
              <Route path="/resources" element={<ResourcesPage />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </LangProvider>
    </ThemeProvider>
  )
}
