import { useState, useEffect } from 'react';
import apiClient from '../api/client';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, FunnelChart, Funnel, LabelList,
} from 'recharts';

const RAG_COLORS = { Hot: '#f97316', Warm: '#eab308', Cold: '#64748b' };
const FUNNEL_COLORS = ['#3b82f6', '#8b5cf6', '#eab308', '#f97316'];

export default function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [conversion, setConversion] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [aRes, cRes] = await Promise.all([
          apiClient.get('/analytics'),
          apiClient.get('/analytics/conversion'),
        ]);
        setAnalytics(aRes.data);
        setConversion(cRes.data);
      } catch (err) {
        console.error('Failed to fetch analytics:', err);
        setError('Failed to load analytics. Ensure backend is running.');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <div className="text-gray-400 text-center py-16">Loading analytics...</div>;
  if (error || !analytics || !conversion) return <div className="text-red-400 text-center py-16">{error}</div>;

  const distData = [
    { name: 'Hot', value: analytics.score_distribution.hot },
    { name: 'Warm', value: analytics.score_distribution.warm },
    { name: 'Cold', value: analytics.score_distribution.cold },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Analytics</h2>
        <p className="text-gray-400 text-sm mt-1">Conversion funnel, lead quality, and prediction tracking</p>
      </div>

      {/* Conversion callout */}
      <div className="bg-idbi/20 border border-idbi/40 p-5 rounded-lg grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div>
          <p className="text-xs text-gray-400 uppercase">Baseline Conversion</p>
          <p className="text-2xl font-bold text-gray-300">{conversion.baseline_conversion_rate}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 uppercase">Actual Conversion (historical)</p>
          <p className="text-2xl font-bold text-white">{conversion.actual_conversion_rate}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 uppercase">Predicted Conversion</p>
          <p className="text-2xl font-bold text-emerald-400">{conversion.predicted_conversion_rate}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 uppercase">Improvement Multiplier</p>
          <p className="text-2xl font-bold text-orange-400">{conversion.improvement_multiplier}x</p>
        </div>
      </div>

      {/* Funnel + distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card border border-gray-800 rounded-lg p-5">
          <h3 className="text-lg font-medium text-white mb-4">Conversion Funnel</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <FunnelChart>
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
                <Funnel dataKey="count" data={analytics.funnel} isAnimationActive>
                  <LabelList position="right" dataKey="stage" fill="#e5e7eb" stroke="none" />
                  <LabelList position="left" dataKey="count" fill="#e5e7eb" stroke="none" />
                  {analytics.funnel.map((_, i) => (
                    <Cell key={i} fill={FUNNEL_COLORS[i % FUNNEL_COLORS.length]} />
                  ))}
                </Funnel>
              </FunnelChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-gray-800 rounded-lg p-5">
          <h3 className="text-lg font-medium text-white mb-4">Lead Score Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 12 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {distData.map((entry) => (
                    <Cell key={entry.name} fill={RAG_COLORS[entry.name]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Conversion by loan type */}
      <div className="bg-card border border-gray-800 rounded-lg p-5">
        <h3 className="text-lg font-medium text-white mb-4">Conversion by Loan Type</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={conversion.by_loan_type}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis dataKey="loan_type" stroke="#6b7280" tick={{ fontSize: 12 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
              <Bar dataKey="total" fill="#334155" radius={[4, 4, 0, 0]} name="Viewed" />
              <Bar dataKey="converted" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Converted" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
