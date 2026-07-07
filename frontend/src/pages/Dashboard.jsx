import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import apiClient from '../api/client';
import { Activity, Users, DollarSign, Target, ArrowRight } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [hotLeads, setHotLeads] = useState([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const [dashRes, leadsRes] = await Promise.all([
          apiClient.get('/dashboard'),
          apiClient.get('/leads?status=hot&limit=5')
        ]);
        setData(dashRes.data);
        setHotLeads(leadsRes.data.data);
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <div className="flex justify-center items-center h-64 text-gray-400">Loading AI insights...</div>;
  if (!data) return <div className="text-red-400">Failed to load data. Ensure backend is running.</div>;

  // Mock data for charts
  const trendData = Array.from({ length: 30 }, (_, i) => ({
    name: `Day ${i+1}`,
    leads: Math.floor(Math.random() * 50) + 10,
  }));

  const pieData = [
    { name: 'Personal', value: 50 },
    { name: 'Home', value: 25 },
    { name: 'Auto', value: 15 },
    { name: 'Mortgage', value: 10 },
  ];
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];

  return (
    <div className="space-y-6">
      <div className="bg-idbi/20 border border-idbi/40 p-4 rounded-lg flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">IDBI Prospect Intelligence</h2>
          <p className="text-idbi-200 text-sm mt-1">AI-Powered Lead Engine</p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-400 uppercase tracking-wide">Predicted Conversion Rate</p>
          <p className="text-3xl font-bold text-emerald-400">{data.predicted_conversion_rate}% <span className="text-sm text-emerald-500">↑ from 1%</span></p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-card p-5 rounded-lg border border-gray-800 flex flex-col">
          <div className="flex items-center text-gray-400 mb-2">
            <Users className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Total Prospects</span>
          </div>
          <span className="text-3xl font-bold text-white">{data.total_prospects.toLocaleString()}</span>
        </div>

        <div className="bg-card p-5 rounded-lg border border-orange-500/30 flex flex-col relative overflow-hidden">
          <div className="absolute top-0 right-0 -mt-2 -mr-2 w-16 h-16 bg-orange-500 rounded-full blur-2xl opacity-20"></div>
          <div className="flex items-center text-orange-400 mb-2">
            <Activity className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Hot Leads</span>
          </div>
          <span className="text-3xl font-bold text-orange-500">{data.hot_leads.toLocaleString()}</span>
        </div>

        <div className="bg-card p-5 rounded-lg border border-gray-800 flex flex-col">
          <div className="flex items-center text-emerald-400 mb-2">
            <DollarSign className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Pipeline Value</span>
          </div>
          <span className="text-3xl font-bold text-emerald-400">₹{data.pipeline_value_cr} Cr</span>
        </div>

        <div className="bg-card p-5 rounded-lg border border-gray-800 flex flex-col">
          <div className="flex items-center text-blue-400 mb-2">
            <Target className="h-5 w-5 mr-2" />
            <span className="text-sm font-medium">Avg Lead Score</span>
          </div>
          <span className="text-3xl font-bold text-blue-400">{data.avg_lead_score}/100</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-card p-5 rounded-lg border border-gray-800">
          <h3 className="text-lg font-medium text-white mb-4">Lead Generation Trend (30 Days)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="colorLeads" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="name" stroke="#6b7280" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                <YAxis stroke="#6b7280" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6'}} />
                <Area type="monotone" dataKey="leads" stroke="#3b82f6" fillOpacity={1} fill="url(#colorLeads)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card p-5 rounded-lg border border-gray-800">
          <h3 className="text-lg font-medium text-white mb-4">Product Demand Split</h3>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{backgroundColor: '#1f2937', borderColor: '#374151'}} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-4 text-xs text-gray-400">
            {pieData.map((entry, index) => (
              <div key={entry.name} className="flex items-center">
                <div className="w-3 h-3 rounded-full mr-1" style={{backgroundColor: COLORS[index]}}></div>
                {entry.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-card rounded-lg border border-gray-800 overflow-hidden">
        <div className="p-5 border-b border-gray-800 flex justify-between items-center">
          <h3 className="text-lg font-medium text-white">Top 5 Actionable Hot Leads</h3>
          <Link to="/leads" className="text-sm text-idbi-300 hover:text-white flex items-center">
            View all leads <ArrowRight className="h-4 w-4 ml-1" />
          </Link>
        </div>
        <div className="divide-y divide-gray-800">
          {hotLeads.map((lead) => (
            <Link key={lead.customer_id} to={`/lead/${lead.customer_id}`} className="p-4 hover:bg-gray-800/50 flex items-center justify-between transition-colors">
              <div className="flex items-center">
                <div className="h-10 w-10 rounded-full bg-orange-500/20 text-orange-500 flex items-center justify-center font-bold mr-4">
                  {lead.composite_score}
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{lead.name}</p>
                  <p className="text-xs text-gray-400">{lead.customer_id} • {lead.city}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-300">{lead.recommended_product}</p>
                <p className="text-xs text-emerald-400">₹{(lead.estimated_amount).toLocaleString()}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
