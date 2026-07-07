import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import apiClient from '../api/client';

const STATUS_COLORS = {
  Hot: 'bg-hot/20 text-hot border-hot/40',
  Warm: 'bg-warm/20 text-warm border-warm/40',
  Cold: 'bg-cold/20 text-cold border-cold/40',
};

function ScoreGauge({ value }) {
  const color = value >= 75 ? 'text-hot' : value >= 50 ? 'text-warm' : 'text-cold';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-gray-800 overflow-hidden">
        <div
          className={`h-full rounded-full ${value >= 75 ? 'bg-hot' : value >= 50 ? 'bg-warm' : 'bg-cold'}`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
      <span className={`text-xs font-medium ${color}`}>{Math.round(value)}</span>
    </div>
  );
}

export default function Leads() {
  const [leads, setLeads] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState('all');
  const [loanType, setLoanType] = useState('all');
  const [sort, setSort] = useState('composite_score');
  const limit = 25;

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    try {
      const params = { page, limit, sort };
      if (status !== 'all') params.status = status;
      if (loanType !== 'all') params.loan_type = loanType;
      const res = await apiClient.get('/leads', { params });
      setLeads(res.data.data);
      setTotal(res.data.total);
    } catch (err) {
      console.error('Failed to fetch leads:', err);
    } finally {
      setLoading(false);
    }
  }, [page, status, loanType, sort]);

  useEffect(() => {
    fetchLeads();
  }, [fetchLeads]);

  useEffect(() => {
    setPage(1);
  }, [status, loanType, sort]);

  const totalPages = Math.max(1, Math.ceil(total / limit));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Smart Lead Table</h2>
        <p className="text-gray-400 text-sm mt-1">{total.toLocaleString()} prospects analyzed</p>
      </div>

      {/* Filter bar */}
      <div className="bg-card p-4 rounded-lg border border-gray-800 flex flex-wrap items-center gap-3">
        <div className="flex gap-2">
          {['all', 'hot', 'warm', 'cold'].map((s) => (
            <button
              key={s}
              onClick={() => setStatus(s)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium capitalize border transition-colors ${
                status === s
                  ? 'bg-idbi border-idbi text-white'
                  : 'border-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <select
          value={loanType}
          onChange={(e) => setLoanType(e.target.value)}
          className="bg-background border border-gray-700 rounded-md px-3 py-1.5 text-sm text-gray-200"
        >
          <option value="all">All Loan Types</option>
          <option value="Personal Loan">Personal Loan</option>
          <option value="Home Loan">Home Loan</option>
          <option value="Auto Loan">Auto Loan</option>
          <option value="Mortgage">Mortgage</option>
        </select>

        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="bg-background border border-gray-700 rounded-md px-3 py-1.5 text-sm text-gray-200"
        >
          <option value="composite_score">Sort: Lead Score</option>
          <option value="intent_score">Sort: Intent</option>
          <option value="capacity_score">Sort: Capacity</option>
          <option value="propensity_score">Sort: Propensity</option>
          <option value="estimated_amount">Sort: Amount</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-card rounded-lg border border-gray-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-900/60 text-gray-400 text-xs uppercase tracking-wide">
              <tr>
                <th className="text-left px-4 py-3">Customer</th>
                <th className="text-left px-4 py-3">Lead Score</th>
                <th className="text-left px-4 py-3">Intent</th>
                <th className="text-left px-4 py-3">Capacity</th>
                <th className="text-left px-4 py-3">Propensity</th>
                <th className="text-left px-4 py-3">Product</th>
                <th className="text-right px-4 py-3">Est. Amount</th>
                <th className="text-left px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {loading && (
                <tr>
                  <td colSpan={8} className="text-center py-8 text-gray-500">
                    Loading leads...
                  </td>
                </tr>
              )}
              {!loading && leads.length === 0 && (
                <tr>
                  <td colSpan={8} className="text-center py-8 text-gray-500">
                    No leads match these filters.
                  </td>
                </tr>
              )}
              {!loading &&
                leads.map((lead) => (
                  <tr key={lead.customer_id} className="hover:bg-gray-800/40 transition-colors">
                    <td className="px-4 py-3">
                      <Link to={`/lead/${lead.customer_id}`} className="text-white font-medium hover:text-idbi-300">
                        {lead.name}
                      </Link>
                      <p className="text-xs text-gray-500">
                        {lead.customer_id} • {lead.city}
                      </p>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center justify-center h-8 w-8 rounded-full border text-xs font-bold ${STATUS_COLORS[lead.rag_status] || STATUS_COLORS.Cold}`}
                      >
                        {Math.round(lead.composite_score)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <ScoreGauge value={lead.intent_score} />
                    </td>
                    <td className="px-4 py-3">
                      <ScoreGauge value={lead.capacity_score} />
                    </td>
                    <td className="px-4 py-3">
                      <ScoreGauge value={lead.propensity_score} />
                    </td>
                    <td className="px-4 py-3 text-gray-300">{lead.recommended_product}</td>
                    <td className="px-4 py-3 text-right text-emerald-400">
                      ₹{lead.estimated_amount.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-gray-400">{lead.status}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800 text-sm text-gray-400">
          <span>
            Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="px-3 py-1 rounded-md border border-gray-700 disabled:opacity-40 hover:text-white"
            >
              Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="px-3 py-1 rounded-md border border-gray-700 disabled:opacity-40 hover:text-white"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
