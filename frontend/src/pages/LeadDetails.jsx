import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import apiClient from '../api/client';
import { ArrowLeft, TrendingUp, Home, Book, Activity, Users } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar,
} from 'recharts';

const EVENT_ICONS = {
  'trending-up': TrendingUp,
  home: Home,
  book: Book,
  activity: Activity,
  users: Users,
};

const SCORE_COLORS = { Hot: '#f97316', Warm: '#eab308', Cold: '#64748b' };
const EXPENSE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f97316'];

// Gemini-generated RM script/reasons may contain markdown (e.g. **bold**).
// Render inline so it drops into existing <p>/<li> wrappers without adding
// its own block-level margins, keeping the surrounding Tailwind typography.
function InlineMarkdown({ text }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => <>{children}</>,
        strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

function ScoreDial({ label, value, accent }) {
  return (
    <div className="bg-card border border-gray-800 rounded-lg p-5 flex flex-col items-center">
      <div className="relative h-28 w-28 flex items-center justify-center">
        <svg className="absolute inset-0 -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" fill="none" stroke="#1f2937" strokeWidth="10" />
          <circle
            cx="50" cy="50" r="42" fill="none" stroke={accent} strokeWidth="10"
            strokeDasharray={`${(value / 100) * 264} 264`} strokeLinecap="round"
          />
        </svg>
        <span className="text-2xl font-bold text-white">{Math.round(value)}</span>
      </div>
      <p className="text-sm text-gray-400 mt-2">{label}</p>
    </div>
  );
}

export default function LeadDetails() {
  const { id } = useParams();
  const [profile, setProfile] = useState(null);
  const [income, setIncome] = useState(null);
  const [spending, setSpending] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [recLoading, setRecLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchAll() {
      setLoading(true);
      setError(null);
      try {
        const [detailRes, incomeRes, spendingRes] = await Promise.all([
          apiClient.get(`/lead/${id}`),
          apiClient.get(`/lead/${id}/income`),
          apiClient.get(`/lead/${id}/spending`),
        ]);
        if (cancelled) return;
        setProfile(detailRes.data);
        setIncome(incomeRes.data);
        setSpending(spendingRes.data);
      } catch (err) {
        if (!cancelled) setError('Customer not found or backend unavailable.');
        console.error(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function generateScript() {
    setRecLoading(true);
    try {
      const res = await apiClient.post(`/recommend/${id}`);
      setRecommendation(res.data);
    } catch (err) {
      console.error('Failed to generate recommendation:', err);
    } finally {
      setRecLoading(false);
    }
  }

  if (loading) return <div className="text-gray-400 text-center py-16">Loading customer profile...</div>;
  if (error || !profile) return <div className="text-red-400 text-center py-16">{error || 'Not found'}</div>;

  const { profile: p, scores, behavior, life_events, recommendation: baseRec } = profile;

  const incomeTrend = income.trend.map((v, i) => ({ month: `M${i + 1}`, income: v }));
  const expenseData = [
    { name: 'Needs', value: income.breakdown.needs },
    { name: 'Wants', value: income.breakdown.wants },
    { name: 'Investments', value: income.breakdown.investments },
    { name: 'EMIs', value: income.breakdown.emis },
  ];

  return (
    <div className="space-y-6">
      <Link to="/leads" className="inline-flex items-center text-sm text-gray-400 hover:text-white">
        <ArrowLeft className="h-4 w-4 mr-1" /> Back to leads
      </Link>

      {/* Profile card */}
      <div className="bg-card border border-gray-800 rounded-lg p-5 flex flex-wrap justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">{p.name}</h2>
          <p className="text-sm text-gray-400 mt-1">
            {p.age} yrs • {p.gender} • {p.city} • {p.occupation} @ {p.employer}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {p.customer_id} • Account tenure: {p.account_tenure} yrs
          </p>
        </div>
        <span
          className="self-start px-3 py-1 rounded-full text-sm font-semibold border"
          style={{
            color: SCORE_COLORS[scores.rag_status] || '#64748b',
            borderColor: SCORE_COLORS[scores.rag_status] || '#64748b',
            backgroundColor: `${SCORE_COLORS[scores.rag_status] || '#64748b'}20`,
          }}
        >
          {scores.rag_status} Lead • {scores.composite}/100
        </span>
      </div>

      {/* Score gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <ScoreDial label="Intent Score" value={scores.intent} accent="#3b82f6" />
        <ScoreDial label="Capacity Score" value={scores.capacity} accent="#10b981" />
        <ScoreDial label="Propensity Score" value={scores.propensity} accent="#f97316" />
      </div>

      {/* Income analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-card border border-gray-800 rounded-lg p-5">
          <h3 className="text-lg font-medium text-white mb-4">Monthly Income Trend</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={incomeTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="month" stroke="#6b7280" tick={{ fontSize: 12 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
                <Line type="monotone" dataKey="income" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-6 mt-4 text-sm">
            <div>
              <p className="text-gray-500">Est. Monthly Income</p>
              <p className="text-white font-semibold">₹{income.estimated_monthly_income.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-gray-500">Income Type</p>
              <p className="text-white font-semibold">{income.income_type}</p>
            </div>
            <div>
              <p className="text-gray-500">FOIR</p>
              <p className="text-white font-semibold">{income.foir}%</p>
            </div>
          </div>
        </div>

        <div className="bg-card border border-gray-800 rounded-lg p-5 flex flex-col">
          <h3 className="text-lg font-medium text-white mb-2">Expense Breakdown</h3>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={expenseData} cx="50%" cy="50%" innerRadius={40} outerRadius={65} dataKey="value">
                  {expenseData.map((_, i) => (
                    <Cell key={i} fill={EXPENSE_COLORS[i % EXPENSE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-center">
            <p className="text-xs text-gray-500 uppercase tracking-wide">Disposable Income</p>
            <p className="text-2xl font-bold text-emerald-400">₹{income.disposable_income.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Available for new EMI</p>
          </div>
        </div>
      </div>

      {/* Behavioral + spending */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card border border-gray-800 rounded-lg p-5">
          <h3 className="text-lg font-medium text-white mb-4">Behavioral Insights</h3>
          <ul className="space-y-2 text-sm text-gray-300">
            <li>App login frequency: <span className="text-white">{behavior.app_login_frequency}</span></li>
            <li>Loan page visits (30d): <span className="text-white">{behavior.loan_page_visits}</span></li>
            <li>EMI calculator uses: <span className="text-white">{behavior.calculator_usage}</span></li>
            <li>
              Application started but not completed:{' '}
              <span className={behavior.application_started ? 'text-hot font-semibold' : 'text-white'}>
                {behavior.application_started ? 'Yes' : 'No'}
              </span>
            </li>
          </ul>
        </div>

        <div className="bg-card border border-gray-800 rounded-lg p-5">
          <h3 className="text-lg font-medium text-white mb-4">Spending Categories</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={spending} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" stroke="#6b7280" tick={{ fontSize: 11 }} width={90} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Life events */}
      <div className="bg-card border border-gray-800 rounded-lg p-5">
        <h3 className="text-lg font-medium text-white mb-4">Life Event Detection</h3>
        {life_events.length === 0 ? (
          <p className="text-sm text-gray-500">No significant life events detected for this customer.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {life_events.map((ev, i) => {
              const Icon = EVENT_ICONS[ev.icon] || Activity;
              return (
                <div key={i} className="flex items-start gap-3 p-3 rounded-md bg-gray-900/40 border border-gray-800">
                  <Icon className="h-5 w-5 text-idbi-300 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-white">{ev.event}</p>
                    <p className="text-xs text-gray-400">{ev.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Recommendation */}
      <div className="bg-card border border-gray-800 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">Product Recommendation</h3>
          <button
            onClick={generateScript}
            disabled={recLoading}
            className="text-sm px-3 py-1.5 rounded-md bg-idbi hover:bg-idbi/80 text-white disabled:opacity-50"
          >
            {recLoading ? 'Generating...' : 'Generate RM Talking Points'}
          </button>
        </div>
        <div className="flex flex-wrap gap-6 mb-4">
          <div>
            <p className="text-xs text-gray-500 uppercase">Recommended Product</p>
            <p className="text-xl font-bold text-white">{baseRec.product}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase">Estimated Amount</p>
            <p className="text-xl font-bold text-emerald-400">₹{baseRec.estimated_amount.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase">Confidence</p>
            <p className="text-xl font-bold text-blue-400">{baseRec.confidence}</p>
          </div>
        </div>
        {recommendation && (
          <div className="border-t border-gray-800 pt-4">
            <p className="text-sm text-gray-300 italic mb-3">
              &ldquo;<InlineMarkdown text={recommendation.script} />&rdquo;
            </p>
            <p className="text-xs text-gray-500 uppercase mb-2">Why this product fits</p>
            <ul className="list-disc list-inside text-sm text-gray-300 space-y-1">
              {recommendation.reasons.map((r, i) => (
                <li key={i}>
                  <InlineMarkdown text={r} />
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
