"use client";

import React, { useState, useCallback, useMemo } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ScatterChart, Scatter, ZAxis, ComposedChart, ReferenceLine,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Play, BarChart3, TrendingUp, Activity, Zap, Settings, Timer,
  Target, Award, GitCompare, Layers, Sun, Thermometer, Wind,
  ChevronRight, CheckCircle2, XCircle, AlertCircle, Loader2,
  Moon, Lightbulb, Database, Shield, Gauge, Sparkles,
} from "lucide-react";
import { useTheme } from "next-themes";

// ─── Types ─────────────────────────────────────────────────────
interface PredictionPoint {
  timestamp: number;
  actual: number;
  predicted: number;
  persistence: number;
}

interface MetricsResult {
  mae: number;
  rmse: number;
  r2: number;
  mape: number;
  skillScore: number;
}

interface FoldMetrics extends MetricsResult {
  fold: number;
  trainStart: number;
  trainEnd: number;
  testStart: number;
  testEnd: number;
}

interface BacktestResult {
  modelName: string;
  target: string;
  folds: FoldMetrics[];
  meanMetrics: MetricsResult;
}

interface TargetResult {
  target: string;
  selectedModel: string;
  finalMetrics: MetricsResult;
  persistenceMetrics: MetricsResult;
  skillScore: number;
  predictions: PredictionPoint[];
  cvResults: BacktestResult[];
}

interface ExperimentResult {
  config: {
    experimentName: string;
    forecastHorizon: number;
    models: string[];
    nSplits: number;
    testSize: number;
    gap: number;
    maxTrainSize: number | null;
    finalTestSize: number;
    syntheticYears: number;
    seed: number;
  };
  targets: TargetResult[];
  rawData: { timestamp: number; load_kw: number; solar_kw: number; temperature_c: number }[];
  featureImportance: { feature: string; loadWeight: number; solarWeight: number }[];
  summary: {
    experimentName: string;
    forecastHorizon: number;
    totalSamples: number;
    trainingSamples: number;
    testSamples: number;
    selectedModels: Record<string, string>;
    totalRuntimeMs: number;
  };
}

// ─── Constants ─────────────────────────────────────────────────
const COLORS = ["#10b981", "#f59e0b", "#6366f1", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"];
const CHART_COLORS = {
  grid: "var(--chart-grid, #e2e8f0)",
  text: "var(--chart-text, #64748b)",
  bg: "var(--chart-bg, transparent)",
};

const MODEL_LABELS: Record<string, string> = {
  persistence: "Persistence",
  ridge: "Ridge (λ=1)",
  ridge_strong: "Ridge (λ=10)",
  ridge_weak: "Ridge (λ=0.1)",
  knn: "KNN (k=7)",
};

// ─── Helpers ───────────────────────────────────────────────────
function fmt(n: number, d = 2) { return n.toFixed(d); }
function fmtTime(ms: number) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
function formatTs(ts: number) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:00`;
}
function formatTsShort(ts: number) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}
function formatHour(ts: number) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:00`;
}

function getHistogramData(values: number[], bins = 25) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins || 1;
  const histogram: { bin: string; count: number }[] = [];
  for (let i = 0; i < bins; i++) {
    const lo = min + i * binWidth;
    const hi = lo + binWidth;
    const count = values.filter(v => v >= lo && (i === bins - 1 ? v <= hi : v < hi)).length;
    histogram.push({ bin: `${lo.toFixed(1)}`, count });
  }
  return histogram;
}

function getFoldComparisonData(cvResults: BacktestResult[]) {
  const maxFolds = Math.max(...cvResults.map(cv => cv.folds.length));
  const data: Record<string, number | string>[] = [];
  for (let f = 0; f < maxFolds; f++) {
    const row: Record<string, number | string> = { fold: `Fold ${f + 1}` };
    for (const cv of cvResults) {
      if (cv.folds[f]) row[cv.modelName] = cv.folds[f].rmse;
    }
    data.push(row);
  }
  return data;
}

function getRadarData(cvResults: BacktestResult[]) {
  const allMAE = cvResults.map(cv => cv.meanMetrics.mae);
  const allRMSE = cvResults.map(cv => cv.meanMetrics.rmse);
  const allR2 = cvResults.map(cv => cv.meanMetrics.r2);
  const allMAPE = cvResults.map(cv => cv.meanMetrics.mape);
  const maxMAE = Math.max(...allMAE, 0.001);
  const maxRMSE = Math.max(...allRMSE, 0.001);
  const maxR2 = Math.max(...allR2, 0.001);
  const maxMAPE = Math.max(...allMAPE, 0.001);
  const metrics = ["MAE", "RMSE", "R²", "MAPE"];

  return metrics.map(metric => {
    const row: Record<string, string | number> = { metric };
    for (const cv of cvResults) {
      let score = 0;
      switch (metric) {
        case "MAE": score = 1 - cv.meanMetrics.mae / maxMAE; break;
        case "RMSE": score = 1 - cv.meanMetrics.rmse / maxRMSE; break;
        case "R²": score = cv.meanMetrics.r2 / maxR2; break;
        case "MAPE": score = 1 - cv.meanMetrics.mape / maxMAPE; break;
      }
      row[cv.modelName] = Math.round(Math.max(0, score) * 100);
    }
    return row;
  });
}

// ─── Tooltip Style ─────────────────────────────────────────────
const tooltipStyle = { borderRadius: 10, fontSize: 12, boxShadow: "0 4px 12px rgba(0,0,0,0.1)" };

// ─── Component ─────────────────────────────────────────────────
export default function ForecastDashboard() {
  const { theme, setTheme } = useTheme();
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTarget, setActiveTarget] = useState("load_kw");

  const [config, setConfig] = useState({
    syntheticYears: 2,
    forecastHorizon: 24,
    nSplits: 4,
    testSize: 168,
    gap: 24,
    maxTrainSize: 8760,
    finalTestSize: 720,
    seed: 42,
    models: ["ridge", "ridge_strong", "knn"] as string[],
  });

  const availableModels = [
    { id: "ridge", label: "Ridge (λ=1)", desc: "L2-regularized linear regression" },
    { id: "ridge_strong", label: "Ridge (λ=10)", desc: "Stronger regularization" },
    { id: "ridge_weak", label: "Ridge (λ=0.1)", desc: "Weaker regularization" },
    { id: "knn", label: "KNN (k=7)", desc: "K-Nearest Neighbors" },
  ];

  const toggleModel = (id: string) => {
    setConfig(prev => ({
      ...prev,
      models: prev.models.includes(id)
        ? prev.models.filter(m => m !== id)
        : [...prev.models, id],
    }));
  };

  const runForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed");
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [config]);

  const currentTarget = result?.targets.find(t => t.target === activeTarget) ?? result?.targets[0];

  // Compute prediction intervals from residual std
  const predictionIntervalData = useMemo(() => {
    if (!currentTarget) return null;
    const residuals = currentTarget.predictions.map(p => p.predicted - p.actual);
    const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
    const std = Math.sqrt(residuals.reduce((s, r) => s + (r - mean) ** 2, 0) / residuals.length);
    return { mean, std, upper: 1.96 * std, lower: -1.96 * std };
  }, [currentTarget]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50/20 dark:from-slate-950 dark:via-slate-900 dark:to-emerald-950/20">
      {/* ─── Header ─── */}
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1640px] mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Zap className="w-4.5 h-4.5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 dark:text-white tracking-tight">Energy Forecast Lab</h1>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 -mt-0.5">Enhanced Day-Ahead Load & Solar Forecasting</p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            {result && (
              <Badge variant="outline" className="hidden sm:flex items-center gap-1 text-[11px]">
                <Timer className="w-3 h-3" />
                {fmtTime(result.summary.totalRuntimeMs)}
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Lightbulb className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </Button>
            <Button
              onClick={runForecast}
              disabled={loading || config.models.length === 0}
              className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-lg shadow-emerald-500/25 text-white h-8 px-4 text-sm"
            >
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" /> : <Play className="w-3.5 h-3.5 mr-1.5" />}
              {loading ? "Running..." : "Run Experiment"}
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-[1640px] mx-auto px-4 sm:px-6 lg:px-8 py-5">
        {/* ─── Config Panel ─── */}
        <Card className="mb-5 border-slate-200/60 dark:border-slate-800/60 shadow-sm bg-white dark:bg-slate-900/50">
          <CardHeader className="pb-3 pt-4 px-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-3.5 h-3.5 text-slate-400" />
                <CardTitle className="text-sm font-semibold">Experiment Configuration</CardTitle>
              </div>
              <CardDescription className="text-[11px] text-slate-400">
                {config.syntheticYears}yr data · {config.forecastHorizon}h horizon · {config.nSplits} folds · gap={config.gap}h
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-x-4 gap-y-3">
              <div className="sm:col-span-2 lg:col-span-3">
                <Label className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1.5 block uppercase tracking-wider">Models</Label>
                <div className="flex flex-wrap gap-1.5">
                  {availableModels.map(m => (
                    <Badge
                      key={m.id}
                      variant={config.models.includes(m.id) ? "default" : "outline"}
                      className={`cursor-pointer transition-all text-[11px] px-2.5 py-0.5 ${
                        config.models.includes(m.id)
                          ? "bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm"
                          : "hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400"
                      }`}
                      onClick={() => toggleModel(m.id)}
                    >
                      {config.models.includes(m.id) && <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" />}
                      {m.label}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1 block uppercase tracking-wider">Horizon</Label>
                <Select value={String(config.forecastHorizon)} onValueChange={v => setConfig(p => ({ ...p, forecastHorizon: Number(v), gap: Number(v) }))}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="12">12 hours</SelectItem>
                    <SelectItem value="24">24 hours</SelectItem>
                    <SelectItem value="48">48 hours</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1 block uppercase tracking-wider">CV Folds</Label>
                <Select value={String(config.nSplits)} onValueChange={v => setConfig(p => ({ ...p, nSplits: Number(v) }))}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {[2, 3, 4, 5, 6].map(n => (
                      <SelectItem key={n} value={String(n)}>{n} folds</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1 block uppercase tracking-wider">Test Window</Label>
                <Select value={String(config.testSize)} onValueChange={v => setConfig(p => ({ ...p, testSize: Number(v) }))}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="72">3 days</SelectItem>
                    <SelectItem value="168">7 days</SelectItem>
                    <SelectItem value="336">14 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1 block uppercase tracking-wider">Holdout</Label>
                <Select value={String(config.finalTestSize)} onValueChange={v => setConfig(p => ({ ...p, finalTestSize: Number(v) }))}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="360">15 days</SelectItem>
                    <SelectItem value="720">30 days</SelectItem>
                    <SelectItem value="1440">60 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="mb-5 border-red-200 dark:border-red-900 bg-red-50/50 dark:bg-red-950/20">
            <CardContent className="flex items-center gap-2 py-3 px-4">
              <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
              <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* ─── Loading ─── */}
        {loading && (
          <div className="space-y-4">
            <Card className="border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50">
              <CardContent className="py-12 flex flex-col items-center gap-3">
                <div className="relative">
                  <Loader2 className="w-10 h-10 animate-spin text-emerald-600" />
                  <Zap className="w-4 h-4 text-emerald-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <div className="text-center">
                  <p className="font-semibold text-slate-700 dark:text-slate-200">Training Forecasting Models</p>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                    {config.models.length} models × {config.nSplits} CV folds × 2 targets
                  </p>
                </div>
                <div className="flex gap-2 mt-1">
                  {config.models.map(m => (
                    <Badge key={m} variant="secondary" className="text-[10px]">{MODEL_LABELS[m] || m}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
            {[1, 2].map(i => <Skeleton key={i} className="h-72 w-full rounded-xl bg-slate-100 dark:bg-slate-800" />)}
          </div>
        )}

        {/* ─── Results ─── */}
        {result && !loading && (
          <div className="space-y-5">
            {/* ─── Summary Cards ─── */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <KpiCard
                icon={<Database className="w-4 h-4" />}
                title="Total Samples"
                value={result.summary.totalSamples.toLocaleString()}
                subtitle={`${result.summary.trainingSamples.toLocaleString()} train · ${result.summary.testSamples.toLocaleString()} test`}
                gradient="from-emerald-500 to-teal-600"
                iconBg="bg-emerald-50 dark:bg-emerald-950/50"
              />
              <KpiCard
                icon={<Target className="w-4 h-4" />}
                title="Forecast Horizon"
                value={`${result.config.forecastHorizon}h`}
                subtitle={`${result.config.nSplits}-fold purged CV`}
                gradient="from-blue-500 to-indigo-600"
                iconBg="bg-blue-50 dark:bg-blue-950/50"
              />
              <KpiCard
                icon={<Activity className="w-4 h-4" />}
                title="Load Skill Score"
                value={`${fmt((result.targets.find(t => t.target === "load_kw")?.skillScore ?? 0) * 100, 1)}%`}
                subtitle={result.targets.find(t => t.target === "load_kw")?.selectedModel || ""}
                gradient="from-violet-500 to-purple-600"
                iconBg="bg-violet-50 dark:bg-violet-950/50"
              />
              <KpiCard
                icon={<Sun className="w-4 h-4" />}
                title="Solar Skill Score"
                value={`${fmt((result.targets.find(t => t.target === "solar_kw")?.skillScore ?? 0) * 100, 1)}%`}
                subtitle={result.targets.find(t => t.target === "solar_kw")?.selectedModel || ""}
                gradient="from-amber-500 to-orange-600"
                iconBg="bg-amber-50 dark:bg-amber-950/50"
              />
              <KpiCard
                icon={<Timer className="w-4 h-4" />}
                title="Runtime"
                value={fmtTime(result.summary.totalRuntimeMs)}
                subtitle="Full pipeline end-to-end"
                gradient="from-slate-600 to-slate-800"
                iconBg="bg-slate-50 dark:bg-slate-800/50"
                className="col-span-2 lg:col-span-1"
              />
            </div>

            {/* ─── Pipeline Diagram ─── */}
            <Card className="border-slate-200/60 dark:border-slate-800/60 bg-white dark:bg-slate-900/50 overflow-hidden">
              <CardContent className="py-3 px-4">
                <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 overflow-x-auto gap-1">
                  {[
                    { icon: <Database className="w-3 h-3" />, label: "Synthetic Data" },
                    { icon: <Layers className="w-3 h-3" />, label: "Feature Engineering" },
                    { icon: <Shield className="w-3 h-3" />, label: "Purged CV Split" },
                    { icon: <Gauge className="w-3 h-3" />, label: "Model Training" },
                    { icon: <Award className="w-3 h-3" />, label: "Model Selection" },
                    { icon: <Target className="w-3 h-3" />, label: "Holdout Eval" },
                  ].map((step, i, arr) => (
                    <React.Fragment key={i}>
                      <div className="flex items-center gap-1.5 whitespace-nowrap">
                        <div className="w-6 h-6 rounded-md bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 flex items-center justify-center">
                          {step.icon}
                        </div>
                        <span className="font-medium">{step.label}</span>
                      </div>
                      {i < arr.length - 1 && (
                        <ChevronRight className="w-3.5 h-3.5 text-slate-300 dark:text-slate-600 shrink-0" />
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* ─── Main Tabs ─── */}
            <Tabs defaultValue="data" className="w-full">
              <TabsList className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm p-1 rounded-lg h-auto flex-wrap gap-1">
                <TabBtn icon={<Layers className="w-3.5 h-3.5" />} value="data">Raw Data</TabBtn>
                <TabBtn icon={<TrendingUp className="w-3.5 h-3.5" />} value="forecasts">Forecasts</TabBtn>
                <TabBtn icon={<GitCompare className="w-3.5 h-3.5" />} value="comparison">Models</TabBtn>
                <TabBtn icon={<BarChart3 className="w-3.5 h-3.5" />} value="metrics">Metrics</TabBtn>
                <TabBtn icon={<Sparkles className="w-3.5 h-3.5" />} value="features">Features</TabBtn>
              </TabsList>

              {/* ═══ RAW DATA ═══ */}
              <TabsContent value="data" className="mt-4">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <ChartCard title="Load (kW) — Full Synthetic Dataset" icon={<Activity className="w-4 h-4 text-emerald-500" />}>
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={result.rawData}>
                        <defs>
                          <linearGradient id="gLoad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis dataKey="timestamp" tickFormatter={formatTsShort} tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <Tooltip labelFormatter={formatTs} contentStyle={tooltipStyle} />
                        <Area type="monotone" dataKey="load_kw" stroke="#10b981" fill="url(#gLoad)" strokeWidth={1.5} name="Load (kW)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </ChartCard>

                  <ChartCard title="Solar PV Generation (kW)" icon={<Sun className="w-4 h-4 text-amber-500" />}>
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={result.rawData}>
                        <defs>
                          <linearGradient id="gSolar" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis dataKey="timestamp" tickFormatter={formatTsShort} tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <Tooltip labelFormatter={formatTs} contentStyle={tooltipStyle} />
                        <Area type="monotone" dataKey="solar_kw" stroke="#f59e0b" fill="url(#gSolar)" strokeWidth={1.5} name="Solar (kW)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </ChartCard>

                  <ChartCard title="Temperature (°C) — Annual + Daily Cycle" icon={<Thermometer className="w-4 h-4 text-red-400" />} className="lg:col-span-2">
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={result.rawData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis dataKey="timestamp" tickFormatter={formatTsShort} tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <Tooltip labelFormatter={formatTs} contentStyle={tooltipStyle} />
                        <ReferenceLine y={0} stroke={CHART_COLORS.grid} />
                        <Line type="monotone" dataKey="temperature_c" stroke="#ef4444" strokeWidth={1.5} dot={false} name="Temperature (°C)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>

                  {/* Duck curve visualization */}
                  <ChartCard title="Net Load Profile (Load − Solar)" icon={<Zap className="w-4 h-4 text-indigo-500" />} className="lg:col-span-2">
                    <ResponsiveContainer width="100%" height={220}>
                      <ComposedChart data={result.rawData.map(d => ({ ...d, net_load: d.load_kw - d.solar_kw }))}>
                        <defs>
                          <linearGradient id="gNet" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis dataKey="timestamp" tickFormatter={formatTsShort} tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                        <Tooltip labelFormatter={formatTs} contentStyle={tooltipStyle} />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="6 3" />
                        <Area type="monotone" dataKey="net_load" stroke="#6366f1" fill="url(#gNet)" strokeWidth={1.5} name="Net Load (kW)" />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </ChartCard>
                </div>
              </TabsContent>

              {/* ═══ FORECASTS ═══ */}
              <TabsContent value="forecasts" className="mt-4">
                <div className="flex items-center gap-2 mb-4">
                  <Label className="text-sm font-medium">Target:</Label>
                  <div className="flex gap-1.5">
                    {result.targets.map(t => (
                      <Badge
                        key={t.target}
                        variant={activeTarget === t.target ? "default" : "outline"}
                        className={`cursor-pointer transition-all ${
                          activeTarget === t.target
                            ? (t.target === "load_kw" ? "bg-emerald-600 text-white" : "bg-amber-500 text-white")
                            : "hover:bg-slate-100 dark:hover:bg-slate-800"
                        }`}
                        onClick={() => setActiveTarget(t.target)}
                      >
                        {t.target === "load_kw" ? <Activity className="w-3 h-3 mr-1" /> : <Sun className="w-3 h-3 mr-1" />}
                        {t.target === "load_kw" ? "Load" : "Solar"}
                      </Badge>
                    ))}
                  </div>
                </div>

                {currentTarget && (
                  <div className="space-y-4">
                    {/* Quick metrics */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                      <MiniMetric label="Selected" value={MODEL_LABELS[currentTarget.selectedModel] || currentTarget.selectedModel} />
                      <MiniMetric label="RMSE" value={fmt(currentTarget.finalMetrics.rmse)} unit=" kW" />
                      <MiniMetric label="MAE" value={fmt(currentTarget.finalMetrics.mae)} unit=" kW" />
                      <MiniMetric label="MAPE" value={fmt(currentTarget.finalMetrics.mape)} unit="%" />
                      <MiniMetric label="R²" value={fmt(currentTarget.finalMetrics.r2, 4)} />
                      <MiniMetric
                        label="Skill Score"
                        value={`${currentTarget.skillScore > 0 ? "+" : ""}${fmt(currentTarget.skillScore * 100, 1)}%`}
                        positive={currentTarget.skillScore > 0}
                      />
                    </div>

                    {/* 95% CI info */}
                    {predictionIntervalData && (
                      <Card className="border-blue-200 dark:border-blue-900 bg-blue-50/40 dark:bg-blue-950/20">
                        <CardContent className="flex items-center gap-3 py-2.5 px-4 text-xs text-blue-700 dark:text-blue-300">
                          <Shield className="w-4 h-4 shrink-0" />
                          <span>95% Prediction Interval: ±{fmt(predictionIntervalData.upper, 2)} kW (residual std: {fmt(predictionIntervalData.std, 2)})</span>
                        </CardContent>
                      </Card>
                    )}

                    {/* Main forecast chart */}
                    <ChartCard
                      title={`Final Holdout — ${activeTarget === "load_kw" ? "Load (kW)" : "Solar (kW)"}`}
                      subtitle={`${currentTarget.selectedModel} vs Actual vs Persistence · ${currentTarget.predictions.length} hours`}
                    >
                      <ResponsiveContainer width="100%" height={380}>
                        <AreaChart data={currentTarget.predictions}>
                          <defs>
                            <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.12} />
                              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#10b981" stopOpacity={0.12} />
                              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                          <XAxis
                            dataKey="timestamp"
                            tickFormatter={formatTsShort}
                            tick={{ fontSize: 10 }}
                            stroke={CHART_COLORS.text}
                            interval={Math.floor(currentTarget.predictions.length / 10)}
                          />
                          <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                          <Tooltip
                            labelFormatter={formatTs}
                            contentStyle={tooltipStyle}
                            formatter={(v: number) => fmt(v, 1)}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Area type="monotone" dataKey="actual" stroke="#6366f1" fill="url(#gActual)" strokeWidth={2} name="Actual" />
                          <Area type="monotone" dataKey="predicted" stroke="#10b981" fill="url(#gPred)" strokeWidth={2} name={MODEL_LABELS[currentTarget.selectedModel] || currentTarget.selectedModel} />
                          <Line type="monotone" dataKey="persistence" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="6 3" name="Persistence" dot={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </ChartCard>

                    {/* Last 7 days zoom */}
                    <ChartCard title="Last 7 Days — Zoomed View" subtitle="Detailed hourly predictions">
                      <ResponsiveContainer width="100%" height={280}>
                        <ComposedChart data={currentTarget.predictions.slice(-168)}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                          <XAxis dataKey="timestamp" tickFormatter={formatHour} tick={{ fontSize: 9 }} stroke={CHART_COLORS.text} interval={5} />
                          <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                          <Tooltip labelFormatter={formatTs} contentStyle={tooltipStyle} formatter={(v: number) => fmt(v, 1)} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Area type="monotone" dataKey="actual" stroke="#6366f1" fill="#6366f1" fillOpacity={0.08} strokeWidth={2} name="Actual" />
                          <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} dot={false} name="Predicted" />
                          <Line type="monotone" dataKey="persistence" stroke="#94a3b8" strokeWidth={1} strokeDasharray="4 3" dot={false} name="Persistence" />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </ChartCard>

                    {/* Residual + Scatter row */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <ChartCard title="Prediction Residuals" subtitle="Predicted − Actual (positive = overestimate)">
                        <ResponsiveContainer width="100%" height={230}>
                          <BarChart data={currentTarget.predictions.map((p, i) => ({ ...p, idx: i, residual: p.predicted - p.actual }))}>
                            <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                            <XAxis dataKey="idx" tick={false} />
                            <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                            <Tooltip
                              labelFormatter={(_, payload) => payload?.[0]?.payload ? formatTs(payload[0].payload.timestamp) : ""}
                              contentStyle={tooltipStyle}
                            />
                            <ReferenceLine y={0} stroke="#64748b" strokeWidth={1} />
                            <Bar dataKey="residual" name="Residual">
                              {currentTarget.predictions.map((p, i) => (
                                <Cell key={i} fill={p.predicted - p.actual >= 0 ? "#f59e0b" : "#6366f1"} fillOpacity={0.7} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </ChartCard>

                      <ChartCard title="Actual vs Predicted" subtitle="Points on the diagonal = perfect forecast">
                        <ResponsiveContainer width="100%" height={230}>
                          <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                            <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 10 }} type="number" stroke={CHART_COLORS.text} />
                            <YAxis dataKey="predicted" name="Predicted" tick={{ fontSize: 10 }} type="number" stroke={CHART_COLORS.text} />
                            <ZAxis range={[20, 20]} />
                            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={tooltipStyle} />
                            <Scatter data={currentTarget.predictions} fill="#10b981" fillOpacity={0.5} />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </ChartCard>
                    </div>

                    {/* Error distribution */}
                    <ChartCard title="Error Distribution" subtitle="Histogram of prediction residuals">
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={getHistogramData(currentTarget.predictions.map(p => p.predicted - p.actual))}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                          <XAxis dataKey="bin" tick={{ fontSize: 9 }} stroke={CHART_COLORS.text} />
                          <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Bar dataKey="count" fill="#8b5cf6" radius={[2, 2, 0, 0]} fillOpacity={0.8} name="Frequency" />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  </div>
                )}
              </TabsContent>

              {/* ═══ MODEL COMPARISON ═══ */}
              <TabsContent value="comparison" className="mt-4">
                <div className="flex items-center gap-2 mb-4">
                  <Label className="text-sm font-medium">Target:</Label>
                  <div className="flex gap-1.5">
                    {result.targets.map(t => (
                      <Badge
                        key={t.target}
                        variant={activeTarget === t.target ? "default" : "outline"}
                        className={`cursor-pointer ${activeTarget === t.target ? (t.target === "load_kw" ? "bg-emerald-600 text-white" : "bg-amber-500 text-white") : ""}`}
                        onClick={() => setActiveTarget(t.target)}
                      >
                        {t.target === "load_kw" ? "Load" : "Solar"}
                      </Badge>
                    ))}
                  </div>
                </div>

                {currentTarget && (
                  <div className="space-y-4">
                    {/* CV RMSE bar chart */}
                    <ChartCard title="Cross-Validation RMSE by Model" subtitle="Selected model highlighted in green">
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={currentTarget.cvResults.map(cv => ({
                            model: MODEL_LABELS[cv.modelName] || cv.modelName,
                            meanRMSE: cv.meanMetrics.rmse,
                            isBest: cv.modelName === currentTarget.selectedModel,
                          }))}
                          layout="vertical"
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                          <XAxis type="number" tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                          <YAxis dataKey="model" type="category" tick={{ fontSize: 11 }} width={110} stroke={CHART_COLORS.text} />
                          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmt(v, 3)} />
                          <Bar dataKey="meanRMSE" name="Mean CV RMSE" radius={[0, 4, 4, 0]} barSize={24}>
                            {currentTarget.cvResults.map((cv, i) => (
                              <Cell key={i} fill={cv.modelName === currentTarget.selectedModel ? "#10b981" : "#cbd5e1"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartCard>

                    {/* Fold-by-fold lines */}
                    <ChartCard title="Fold-by-Fold RMSE Stability" subtitle="Consistency across rolling-origin splits">
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={getFoldComparisonData(currentTarget.cvResults)}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                          <XAxis dataKey="fold" tick={{ fontSize: 11 }} stroke={CHART_COLORS.text} />
                          <YAxis tick={{ fontSize: 10 }} stroke={CHART_COLORS.text} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          {currentTarget.cvResults.map((cv, i) => (
                            <Line
                              key={cv.modelName}
                              type="monotone"
                              dataKey={cv.modelName}
                              stroke={COLORS[i % COLORS.length]}
                              strokeWidth={cv.modelName === currentTarget.selectedModel ? 3 : 1.5}
                              dot={{ r: 4 }}
                              strokeDasharray={cv.modelName === "persistence" ? "6 3" : undefined}
                              name={MODEL_LABELS[cv.modelName] || cv.modelName}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartCard>

                    {/* CV Table */}
                    <Card className="border-slate-200/60 dark:border-slate-800/60 bg-white dark:bg-slate-900/50">
                      <CardHeader className="pb-2 px-4 pt-4">
                        <CardTitle className="text-sm font-semibold">Detailed Cross-Validation Results</CardTitle>
                        <CardDescription className="text-[11px]">Per-fold metrics for all candidate models</CardDescription>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <ScrollArea className="max-h-72">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="text-[11px]">Model</TableHead>
                                <TableHead className="text-[11px] text-right">Fold</TableHead>
                                <TableHead className="text-[11px] text-right">MAE</TableHead>
                                <TableHead className="text-[11px] text-right">RMSE</TableHead>
                                <TableHead className="text-[11px] text-right">R²</TableHead>
                                <TableHead className="text-[11px] text-right">MAPE%</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {currentTarget.cvResults.flatMap(cv =>
                                cv.folds.map(fold => (
                                  <TableRow key={`${cv.modelName}-f${fold.fold}`}>
                                    <TableCell className="text-xs font-medium">
                                      <div className="flex items-center gap-1">
                                        {cv.modelName === currentTarget.selectedModel && <Award className="w-3 h-3 text-amber-500" />}
                                        {MODEL_LABELS[cv.modelName] || cv.modelName}
                                      </div>
                                    </TableCell>
                                    <TableCell className="text-xs text-right">{fold.fold}</TableCell>
                                    <TableCell className="text-xs text-right font-mono">{fmt(fold.mae, 3)}</TableCell>
                                    <TableCell className="text-xs text-right font-mono">{fmt(fold.rmse, 3)}</TableCell>
                                    <TableCell className="text-xs text-right font-mono">{fmt(fold.r2, 4)}</TableCell>
                                    <TableCell className="text-xs text-right font-mono">{fmt(fold.mape, 1)}</TableCell>
                                  </TableRow>
                                ))
                              )}
                            </TableBody>
                          </Table>
                        </ScrollArea>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </TabsContent>

              {/* ═══ METRICS ═══ */}
              <TabsContent value="metrics" className="mt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.targets.map(target => (
                    <Card
                      key={target.target}
                      className={`border-2 transition-all bg-white dark:bg-slate-900/50 ${
                        activeTarget === target.target
                          ? "border-emerald-500 dark:border-emerald-600 shadow-md shadow-emerald-500/10"
                          : "border-slate-200/60 dark:border-slate-800/60"
                      } cursor-pointer`}
                      onClick={() => setActiveTarget(target.target)}
                    >
                      <CardHeader className="pb-2 px-4 pt-4">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm font-semibold flex items-center gap-2">
                            {target.target === "load_kw"
                              ? <><Activity className="w-4 h-4 text-emerald-500" /> Load (kW)</>
                              : <><Sun className="w-4 h-4 text-amber-500" /> Solar (kW)</>}
                          </CardTitle>
                          <Badge variant="outline" className="text-[10px]">{MODEL_LABELS[target.selectedModel] || target.selectedModel}</Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="px-4 pb-4">
                        <div className="space-y-2.5">
                          <MetricCompare label="RMSE" model={target.finalMetrics.rmse} persistence={target.persistenceMetrics.rmse} unit=" kW" />
                          <MetricCompare label="MAE" model={target.finalMetrics.mae} persistence={target.persistenceMetrics.mae} unit=" kW" />
                          <MetricCompare label="MAPE" model={target.finalMetrics.mape} persistence={target.persistenceMetrics.mape} unit="%" />
                          <MetricCompare label="R²" model={target.finalMetrics.r2} persistence={target.persistenceMetrics.r2} lowerBetter={false} />
                          <Separator />
                          <div className="flex justify-between items-center">
                            <span className="text-xs text-slate-500">Skill Score vs Persistence</span>
                            <div className="flex items-center gap-2">
                              <Progress value={Math.max(0, target.skillScore * 100)} className="w-20 h-1.5" />
                              <span className={`text-xs font-bold ${target.skillScore > 0 ? "text-emerald-600" : "text-red-500"}`}>
                                {target.skillScore > 0 ? "+" : ""}{fmt(target.skillScore * 100, 1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {/* Radar */}
                <ChartCard title="Normalized Performance Radar" subtitle={`All models — ${currentTarget?.target}`}>
                  <ResponsiveContainer width="100%" height={320}>
                    <RadarChart data={getRadarData(currentTarget?.cvResults ?? [])}>
                      <PolarGrid stroke={CHART_COLORS.grid} />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} stroke={CHART_COLORS.text} />
                      <PolarRadiusAxis tick={{ fontSize: 8 }} stroke={CHART_COLORS.text} />
                      {currentTarget?.cvResults.map((cv, i) => (
                        <Radar
                          key={cv.modelName}
                          name={MODEL_LABELS[cv.modelName] || cv.modelName}
                          dataKey={cv.modelName}
                          stroke={COLORS[i % COLORS.length]}
                          fill={COLORS[i % COLORS.length]}
                          fillOpacity={0.08}
                          strokeWidth={cv.modelName === currentTarget.selectedModel ? 2.5 : 1}
                        />
                      ))}
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                    </RadarChart>
                  </ResponsiveContainer>
                </ChartCard>
              </TabsContent>

              {/* ═══ FEATURES ═══ */}
              <TabsContent value="features" className="mt-4">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <ChartCard
                    title="Load Model — Feature Weights"
                    subtitle="Ridge regression coefficients (standardized features)"
                    icon={<Activity className="w-4 h-4 text-emerald-500" />}
                  >
                    <ResponsiveContainer width="100%" height={460}>
                      <BarChart
                        data={[...result.featureImportance].sort((a, b) => Math.abs(b.loadWeight) - Math.abs(a.loadWeight))}
                        layout="vertical"
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis type="number" tick={{ fontSize: 9 }} stroke={CHART_COLORS.text} />
                        <YAxis dataKey="feature" type="category" tick={{ fontSize: 9 }} width={110} stroke={CHART_COLORS.text} />
                        <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmt(v, 3)} />
                        <Bar dataKey="loadWeight" name="Weight" radius={[0, 3, 3, 0]} barSize={14}>
                          {[...result.featureImportance].sort((a, b) => Math.abs(b.loadWeight) - Math.abs(a.loadWeight)).map((f, i) => (
                            <Cell key={i} fill={f.loadWeight >= 0 ? "#10b981" : "#ef4444"} fillOpacity={0.8} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartCard>

                  <ChartCard
                    title="Solar Model — Feature Weights"
                    subtitle="Ridge regression coefficients (standardized features)"
                    icon={<Sun className="w-4 h-4 text-amber-500" />}
                  >
                    <ResponsiveContainer width="100%" height={460}>
                      <BarChart
                        data={[...result.featureImportance].sort((a, b) => Math.abs(b.solarWeight) - Math.abs(a.solarWeight))}
                        layout="vertical"
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                        <XAxis type="number" tick={{ fontSize: 9 }} stroke={CHART_COLORS.text} />
                        <YAxis dataKey="feature" type="category" tick={{ fontSize: 9 }} width={110} stroke={CHART_COLORS.text} />
                        <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmt(v, 3)} />
                        <Bar dataKey="solarWeight" name="Weight" radius={[0, 3, 3, 0]} barSize={14}>
                          {[...result.featureImportance].sort((a, b) => Math.abs(b.solarWeight) - Math.abs(a.solarWeight)).map((f, i) => (
                            <Cell key={i} fill={f.solarWeight >= 0 ? "#f59e0b" : "#6366f1"} fillOpacity={0.8} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartCard>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        )}

        {/* ─── Empty State ─── */}
        {!result && !loading && !error && (
          <Card className="border-dashed border-2 border-slate-200 dark:border-slate-700 bg-white/50 dark:bg-slate-900/30">
            <CardContent className="py-20 flex flex-col items-center gap-5">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-100 to-teal-100 dark:from-emerald-950 dark:to-teal-950 flex items-center justify-center">
                <Zap className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div className="text-center max-w-sm">
                <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">Ready to Forecast</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
                  Configure your experiment above and click <strong className="text-slate-700 dark:text-slate-200">Run Experiment</strong> to
                  execute the enhanced day-ahead forecasting pipeline with purged rolling-origin cross-validation.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-1.5 mt-1">
                {[
                  "Leakage-safe features",
                  "Purged CV (gap=24h)",
                  "Ridge + KNN models",
                  "Skill scores",
                  "Feature analysis",
                  "Dark mode",
                ].map(tag => (
                  <Badge key={tag} variant="secondary" className="text-[10px] font-medium">{tag}</Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </main>

      {/* ─── Footer ─── */}
      {result && (
        <footer className="border-t border-slate-200 dark:border-slate-800 mt-8">
          <div className="max-w-[1640px] mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-400">
            <span>Enhanced from <span className="font-medium">ieeexplorer/Time-Series-Forecasting-for-Load-Solar-Generation-Python-ML</span></span>
            <span>27 features · {result.config.models.length} candidate models · {result.config.nSplits}-fold purged CV · TypeScript implementation</span>
          </div>
        </footer>
      )}
    </div>
  );
}

// ─── Sub-Components ────────────────────────────────────────────

function KpiCard({ icon, title, value, subtitle, gradient, iconBg, className }: {
  icon: React.ReactNode; title: string; value: string; subtitle: string;
  gradient: string; iconBg: string; className?: string;
}) {
  return (
    <Card className={`border-slate-200/60 dark:border-slate-800/60 shadow-sm bg-white dark:bg-slate-900/50 ${className || ""}`}>
      <CardContent className="p-3.5">
        <div className="flex items-center gap-2 mb-1">
          <div className={`w-7 h-7 rounded-lg ${iconBg} flex items-center justify-center`}>
            <span className={`bg-gradient-to-br ${gradient} bg-clip-text text-transparent`}>
              {icon}
            </span>
          </div>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">{title}</span>
        </div>
        <p className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">{value}</p>
        <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">{subtitle}</p>
      </CardContent>
    </Card>
  );
}

function MiniMetric({ label, value, unit = "", positive }: {
  label: string; value: string; unit?: string; positive?: boolean;
}) {
  return (
    <Card className="border-slate-200/60 dark:border-slate-800/60 bg-white dark:bg-slate-900/50">
      <CardContent className="p-2.5">
        <p className="text-[10px] text-slate-400 dark:text-slate-500 uppercase tracking-wider font-medium">{label}</p>
        <p className={`text-base font-bold mt-0.5 ${positive !== undefined ? (positive ? "text-emerald-600" : "text-red-500") : "text-slate-900 dark:text-white"}`}>
          {value}{unit}
        </p>
      </CardContent>
    </Card>
  );
}

function MetricCompare({ label, model, persistence, unit = "", lowerBetter = true }: {
  label: string; model: number; persistence: number; unit: string; lowerBetter?: boolean;
}) {
  const improved = lowerBetter ? model < persistence : model > persistence;
  const pctChange = lowerBetter
    ? ((persistence - model) / Math.max(persistence, 0.001)) * 100
    : ((model - persistence) / Math.max(persistence, 0.001)) * 100;

  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">{fmt(persistence)}{unit}</span>
        <ChevronRight className="w-3 h-3 text-slate-300 dark:text-slate-600" />
        <span className="text-xs font-bold text-slate-800 dark:text-slate-100 font-mono">{fmt(model)}{unit}</span>
        {improved ? (
          <Badge variant="secondary" className="text-[9px] px-1.5 py-0 bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 font-mono h-4">
            -{fmt(pctChange, 1)}%
          </Badge>
        ) : (
          <Badge variant="secondary" className="text-[9px] px-1.5 py-0 bg-red-50 dark:bg-red-950/50 text-red-500 font-mono h-4">
            +{fmt(Math.abs(pctChange), 1)}%
          </Badge>
        )}
      </div>
    </div>
  );
}

function ChartCard({ title, subtitle, icon, className, children }: {
  title: string; subtitle?: string; icon?: React.ReactNode; className?: string; children: React.ReactNode;
}) {
  return (
    <Card className={`border-slate-200/60 dark:border-slate-800/60 shadow-sm bg-white dark:bg-slate-900/50 ${className || ""}`}>
      <CardHeader className="pb-1 pt-3.5 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold flex items-center gap-1.5">
            {icon}{title}
          </CardTitle>
        </div>
        {subtitle && <CardDescription className="text-[11px] mt-0.5">{subtitle}</CardDescription>}
      </CardHeader>
      <CardContent className="px-4 pb-4">{children}</CardContent>
    </Card>
  );
}

function TabBtn({ icon, children, ...props }: React.ComponentProps<typeof TabsTrigger> & { icon: React.ReactNode }) {
  return (
    <TabsTrigger
      {...props}
      className="data-[state=active]:bg-emerald-600 data-[state=active]:text-white data-[state=active]:shadow-sm text-[11px] font-medium px-3 py-1.5 text-slate-600 dark:text-slate-400 data-[state=active]:text-white"
    >
      <span className="flex items-center gap-1.5">{icon}{children}</span>
    </TabsTrigger>
  );
}