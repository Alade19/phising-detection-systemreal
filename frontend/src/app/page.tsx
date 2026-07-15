"use client";

import { useEffect, useState } from "react";
import {
  Search,
  Mail,
  ShieldCheck,
  ShieldAlert,
  Trash2,
  Loader2,
} from "lucide-react";
import {
  checkUrl,
  clearHistory,
  fetchHistory,
  checkEmail,
  clearEmailHistory,
  fetchEmailHistory,
  type PredictionEntry,
  type EmailPredictionEntry,
} from "@/lib/api";

type Tab = "url" | "email";

export default function Home() {
  const [tab, setTab] = useState<Tab>("url");

  return (
    <main className="flex-1 flex flex-col items-center px-4 py-16">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <ShieldCheck className="w-9 h-9 text-blue-600" strokeWidth={2.5} />
          <h1 className="text-3xl sm:text-4xl font-bold text-blue-600">
            AI Phishing Detector
          </h1>
        </div>

        {/* Tab switcher */}
        <div className="flex justify-center gap-2 mb-6">
          <button
            onClick={() => setTab("url")}
            className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors ${
              tab === "url"
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            <Search className="w-4 h-4" />
            Check URL
          </button>
          <button
            onClick={() => setTab("email")}
            className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors ${
              tab === "email"
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            <Mail className="w-4 h-4" />
            Check Email
          </button>
        </div>

        {tab === "url" ? <UrlDetector /> : <EmailDetector />}

        <p className="text-center text-xs text-gray-400 mt-8">
          Built for educational purposes. Always verify suspicious links and
          messages through an official source.
        </p>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// URL Detector
// ---------------------------------------------------------------------------
function UrlDetector() {
  const [url, setUrl] = useState("");
  const [history, setHistory] = useState<PredictionEntry[]>([]);
  const [latest, setLatest] = useState<PredictionEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory()
      .then((data) => setHistory(data.history))
      .catch(() => {
        // Backend may not be reachable yet; fail silently on first load.
      });
  }, []);

  async function handleCheck(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim() || loading) return;

    setLoading(true);
    setError(null);
    try {
      const data = await checkUrl(url.trim());
      setLatest(data.prediction);
      setHistory(data.history);
      setUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to check URL.");
    } finally {
      setLoading(false);
    }
  }

  async function handleClearHistory() {
    if (clearing) return;
    setClearing(true);
    try {
      const data = await clearHistory();
      setHistory(data.history);
      setLatest(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to clear history."
      );
    } finally {
      setClearing(false);
    }
  }

  return (
    <>
      {/* Check URL card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 sm:p-8 mb-6">
        <form onSubmit={handleCheck} className="flex flex-col gap-4">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter URL to check"
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={loading || !url.trim()}
            className="self-start inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {loading ? "Checking..." : "Check URL"}
          </button>
        </form>

        {error && (
          <p className="mt-4 text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-4 py-2">
            {error}
          </p>
        )}

        {latest && !error && (
          <ResultBanner result={latest.result} confidence={latest.confidence}>
            <p className="text-sm text-gray-600 break-all mt-0.5">
              {latest.url}
            </p>
          </ResultBanner>
        )}
      </div>

      {/* History card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 sm:p-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="flex items-center gap-2 text-lg font-semibold text-gray-800">
            <span aria-hidden>🕒</span> Last 10 Predictions
          </h2>
          <button
            onClick={handleClearHistory}
            disabled={clearing || history.length === 0}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-red-600 disabled:opacity-40 disabled:hover:text-gray-500 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear History
          </button>
        </div>

        <div className="overflow-x-auto -mx-2">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-100 text-left text-gray-600">
                <th className="px-3 py-2 rounded-l-lg font-semibold w-10">#</th>
                <th className="px-3 py-2 font-semibold">URL</th>
                <th className="px-3 py-2 rounded-r-lg font-semibold w-32">
                  Result
                </th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-3 py-8 text-center text-gray-400">
                    No predictions yet. Check a URL above to get started.
                  </td>
                </tr>
              ) : (
                history.map((entry, i) => (
                  <tr
                    key={`${entry.timestamp}-${i}`}
                    className="border-b border-gray-50 last:border-0"
                  >
                    <td className="px-3 py-3 text-gray-400">{i + 1}</td>
                    <td className="px-3 py-3 text-gray-700 break-all max-w-xs">
                      {entry.url}
                    </td>
                    <td className="px-3 py-3">
                      <ResultBadge result={entry.result} />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Email Detector
// ---------------------------------------------------------------------------
function EmailDetector() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [history, setHistory] = useState<EmailPredictionEntry[]>([]);
  const [latest, setLatest] = useState<EmailPredictionEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEmailHistory()
      .then((data) => setHistory(data.history))
      .catch(() => {
        // Backend may not be reachable yet; fail silently on first load.
      });
  }, []);

  async function handleCheck(e: React.FormEvent) {
    e.preventDefault();
    if ((!subject.trim() && !body.trim()) || loading) return;

    setLoading(true);
    setError(null);
    try {
      const data = await checkEmail(subject.trim(), body.trim());
      setLatest(data.prediction);
      setHistory(data.history);
      setSubject("");
      setBody("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to check email.");
    } finally {
      setLoading(false);
    }
  }

  async function handleClearHistory() {
    if (clearing) return;
    setClearing(true);
    try {
      const data = await clearEmailHistory();
      setHistory(data.history);
      setLatest(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to clear history."
      );
    } finally {
      setClearing(false);
    }
  }

  return (
    <>
      {/* Check Email card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 sm:p-8 mb-6">
        <form onSubmit={handleCheck} className="flex flex-col gap-4">
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="Email subject (optional)"
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Paste the email body here"
            rows={6}
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-y"
          />
          <button
            type="submit"
            disabled={loading || (!subject.trim() && !body.trim())}
            className="self-start inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {loading ? "Checking..." : "Check Email"}
          </button>
        </form>

        {error && (
          <p className="mt-4 text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-4 py-2">
            {error}
          </p>
        )}

        {latest && !error && (
          <ResultBanner result={latest.result} confidence={latest.confidence}>
            {latest.subject && (
              <p className="text-sm font-medium text-gray-700 mt-0.5">
                {latest.subject}
              </p>
            )}
            <p className="text-sm text-gray-600 mt-0.5">{latest.preview}</p>
          </ResultBanner>
        )}
      </div>

      {/* History card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 sm:p-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="flex items-center gap-2 text-lg font-semibold text-gray-800">
            <span aria-hidden>🕒</span> Last 10 Email Checks
          </h2>
          <button
            onClick={handleClearHistory}
            disabled={clearing || history.length === 0}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-gray-500 hover:text-red-600 disabled:opacity-40 disabled:hover:text-gray-500 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear History
          </button>
        </div>

        <div className="overflow-x-auto -mx-2">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-100 text-left text-gray-600">
                <th className="px-3 py-2 rounded-l-lg font-semibold w-10">#</th>
                <th className="px-3 py-2 font-semibold">Subject / Preview</th>
                <th className="px-3 py-2 rounded-r-lg font-semibold w-32">
                  Result
                </th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-3 py-8 text-center text-gray-400">
                    No emails checked yet. Paste an email above to get started.
                  </td>
                </tr>
              ) : (
                history.map((entry, i) => (
                  <tr
                    key={`${entry.timestamp}-${i}`}
                    className="border-b border-gray-50 last:border-0"
                  >
                    <td className="px-3 py-3 text-gray-400">{i + 1}</td>
                    <td className="px-3 py-3 text-gray-700 max-w-xs">
                      {entry.subject && (
                        <span className="font-medium block truncate">
                          {entry.subject}
                        </span>
                      )}
                      <span className="text-gray-500 block truncate">
                        {entry.preview}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <ResultBadge result={entry.result} />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Shared small components
// ---------------------------------------------------------------------------
function ResultBanner({
  result,
  confidence,
  children,
}: {
  result: "phishing" | "legitimate";
  confidence: number;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`mt-6 rounded-xl border px-5 py-4 flex items-start gap-3 ${
        result === "phishing"
          ? "bg-red-50 border-red-200"
          : "bg-green-50 border-green-200"
      }`}
    >
      {result === "phishing" ? (
        <ShieldAlert className="w-6 h-6 text-red-600 shrink-0 mt-0.5" />
      ) : (
        <ShieldCheck className="w-6 h-6 text-green-600 shrink-0 mt-0.5" />
      )}
      <div className="min-w-0 flex-1">
        <p
          className={`font-semibold ${
            result === "phishing" ? "text-red-700" : "text-green-700"
          }`}
        >
          {result === "phishing" ? "Likely Phishing" : "Looks Legitimate"}
          <span className="ml-2 font-normal text-sm text-gray-500">
            ({confidence}% confidence)
          </span>
        </p>
        {children}
      </div>
    </div>
  );
}

function ResultBadge({ result }: { result: "phishing" | "legitimate" }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${
        result === "phishing"
          ? "bg-red-100 text-red-700"
          : "bg-green-100 text-green-700"
      }`}
    >
      {result === "phishing" ? "Phishing" : "Legitimate"}
    </span>
  );
}
