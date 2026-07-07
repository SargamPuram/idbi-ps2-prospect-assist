import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Leads from './pages/Leads';
import LeadDetails from './pages/LeadDetails';
import Analytics from './pages/Analytics';

function App() {
  // Use VITE_BASE_PATH if available (for production deployment), else default to /
  const basename = import.meta.env.VITE_BASE_PATH || '/';
  
  return (
    <Router basename={basename}>
      <div className="min-h-screen bg-background text-gray-100 font-sans flex flex-col">
        {/* Navigation Bar */}
        <nav className="bg-idbi text-white shadow-md z-10 sticky top-0">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <span className="font-bold text-xl tracking-tight">IDBI Prospect Assist AI</span>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link to="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-white text-sm font-medium">
                    Command Center
                  </Link>
                  <Link to="/leads" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-white text-sm font-medium">
                    Smart Leads
                  </Link>
                  <Link to="/analytics" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-white text-sm font-medium">
                    Analytics
                  </Link>
                </div>
              </div>
              <div className="flex items-center">
                <span className="text-sm">RM Portal | Logged in as SargamPuram</span>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 w-full max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/leads" element={<Leads />} />
            <Route path="/lead/:id" element={<LeadDetails />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
