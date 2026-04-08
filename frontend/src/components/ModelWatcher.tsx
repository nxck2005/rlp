import React, { useEffect, useRef, useState } from 'react';
import { Activity, Target, BarChart3, AlertCircle, Eye, ChevronLeft, ChevronRight } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

interface ModelWatcherProps {
  modelId: string;
  name: string;
}

interface TelemetryData {
  action: number;
  reward: number;
  done: boolean;
  reset: boolean;
  agent_view?: number[][][]; // 7x7x3
  full_grid?: number[][][]; // WxHx3
  agent_pos?: [number, number];
  agent_dir?: number;
  final_reward?: number;
  steps?: number;
}

interface StepData {
  frame: string;
  agent_view: number[][][];
  full_grid: number[][][];
  agent_pos: [number, number];
  agent_dir: number;
  action: number;
  reward: number;
  done: boolean;
}

interface EpisodeRecord {
  id: number;
  steps: StepData[];
  finalReward: number;
  totalSteps: number;
}

const ACTION_LABELS = [
  'Turn Left',
  'Turn Right',
  'Move Forward',
  'Pick Up',
  'Drop',
  'Toggle',
  'Done'
];

const COLORS = ['#111111', '#333333', '#555555', '#777777', '#999999', '#AAAAAA', '#CCCCCC'];

// Map MiniGrid object IDs to Colors (Monochrome/Minimalist)
const OBJECT_COLOR_MAP: Record<number, string> = {
  0: 'transparent', // Empty
  1: 'var(--fg)',    // Wall
  2: 'transparent', // Floor
  3: 'var(--sub)',   // Door
  4: '#eab308',      // Key (Keep yellow for contrast)
  5: 'var(--fg)',    // Ball
  6: 'var(--fg)',    // Box
  7: '#22c55e',      // Goal (Keep green for contrast)
  8: '#ef4444',      // Lava (Keep red for danger)
};

const AgentEye: React.FC<{ grid?: number[][][] }> = ({ grid }) => {
  if (!grid || !Array.isArray(grid) || grid.length < 7) {
    return <div className="aspect-square bg-accent-bg border border-border rounded animate-pulse" />;
  }
  
  const flatCells = [];
  try {
    for (let y = 0; y < 7; y++) {
      for (let x = 0; x < 7; x++) {
        flatCells.push(grid[x][y] || [0, 0, 0]);
      }
    }
  } catch (e) { return <div className="aspect-square border border-border" />; }

  return (
    <div className="grid grid-cols-7 gap-px aspect-square bg-border p-px border border-border">
      {flatCells.map((cell, i) => {
          const objectId = cell[0];
          const state = cell[2];
          let bgColor = OBJECT_COLOR_MAP[objectId] || 'var(--bg)';
          if (objectId === 3 && state === 1) bgColor = '#22c55e'; // Open door
          
          return (
            <div 
              key={i} 
              className="w-full h-full bg-bg flex items-center justify-center"
              style={{ backgroundColor: bgColor === 'transparent' ? 'var(--bg)' : bgColor }}
            />
          );
      })}
    </div>
  );
};

const SpatialOverlay: React.FC<{ 
  path: [number, number][], 
  heatmap: Record<string, number>, 
  gridSize: [number, number],
  visible: { path: boolean, heatmap: boolean }
}> = ({ path, heatmap, gridSize, visible }) => {
  if (!gridSize[0]) return null;
  const [W, H] = gridSize;
  
  // Find max value in heatmap for normalization
  const maxHeat = Math.max(...Object.values(heatmap), 1);

  return (
    <div className="absolute inset-0 pointer-events-none grid" style={{ 
      gridTemplateColumns: `repeat(${W}, 1fr)`,
      gridTemplateRows: `repeat(${H}, 1fr)`,
      padding: '16px' // Match the padding of the container
    }}>
      {/* Heatmap Layer */}
      {visible.heatmap && Array.from({ length: W * H }).map((_, i) => {
        const x = i % W;
        const y = Math.floor(i / W);
        const count = heatmap[`${x},${y}`] || 0;
        const opacity = (count / maxHeat) * 0.6;
        return (
          <div 
            key={`heat-${i}`} 
            style={{ backgroundColor: `rgba(255, 165, 0, ${opacity})` }}
            className="w-full h-full border border-white/5"
          />
        );
      })}

      {/* Path / Breadcrumb Layer (SVG Overlay) */}
      {visible.path && (
        <svg 
          className="absolute inset-0 w-full h-full" 
          viewBox={`0 0 ${W} ${H}`} 
          preserveAspectRatio="none"
          style={{ padding: '16px' }}
        >
          {path.length > 1 && (
            <polyline
              points={path.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(' ')}
              fill="none"
              stroke="white"
              strokeWidth="0.05"
              strokeLinejoin="round"
              strokeDasharray="0.1 0.1"
              className="opacity-50"
            />
          )}
          {path.length > 0 && (
            <circle 
              cx={path[path.length-1][0] + 0.5} 
              cy={path[path.length-1][1] + 0.5} 
              r="0.1" 
              fill="white" 
            />
          )}
        </svg>
      )}
    </div>
  );
};

const ModelWatcher: React.FC<ModelWatcherProps> = ({ modelId, name }) => {
  const [frame, setFrame] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionHistory, setActionHistory] = useState<number[]>(new Array(7).fill(0));
  const [successCount, setSuccessCount] = useState(0);
  const [episodeCount, setEpisodeCount] = useState(0);
  
  const [isLive, setIsLive] = useState(true);
  const [history, setHistory] = useState<EpisodeRecord[]>([]);
  const [currentEpisodeIndex, setCurrentEpisodeIndex] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentRecordingSteps, setCurrentRecordingSteps] = useState<StepData[]>([]);

  // Spatial Analysis State
  const [path, setPath] = useState<[number, number][]>([]);
  const [heatmap, setHeatmap] = useState<Record<string, number>>({});
  const [showPath, setShowPath] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [gridSize, setGridSize] = useState<[number, number]>([0, 0]);

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const url = `ws://${window.location.hostname}:8000/ws/${modelId}`;
    const socket = new WebSocket(url);
    ws.current = socket;

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) { setError(data.error); return; }

        const step: StepData = {
          frame: data.frame,
          agent_view: data.agent_view,
          full_grid: data.full_grid,
          agent_pos: data.agent_pos,
          agent_dir: data.agent_dir,
          action: data.action,
          reward: data.reward,
          done: data.done
        };

        if (gridSize[0] === 0 && data.full_grid) {
          setGridSize([data.full_grid.length, data.full_grid[0].length]);
        }

        setIsLive(currentIsLive => {
          if (currentIsLive) {
            setFrame(data.frame);
            setTelemetry({
              action: data.action, reward: data.reward, done: data.done,
              reset: data.reset, agent_view: data.agent_view,
              full_grid: data.full_grid, agent_pos: data.agent_pos,
              agent_dir: data.agent_dir,
              final_reward: data.final_reward, steps: data.steps
            });
            
            // Update live spatial data
            setPath(prev => [...prev, data.agent_pos]);
            const posKey = `${data.agent_pos[0]},${data.agent_pos[1]}`;
            setHeatmap(prev => ({ ...prev, [posKey]: (prev[posKey] || 0) + 1 }));
          }
          return currentIsLive;
        });

        setCurrentRecordingSteps(prev => {
          const nextSteps = [...prev, step];
          if (data.reset) {
            setHistory(prevH => [{
              id: episodeCount + 1, steps: nextSteps,
              finalReward: data.final_reward, totalSteps: data.steps
            }, ...prevH].slice(0, 10));
            setEpisodeCount(c => c + 1);
            if (data.final_reward > 0) setSuccessCount(s => s + 1);
            setPath([]); // Reset path for new episode
            return [];
          }
          return nextSteps;
        });

        setActionHistory(prev => {
          const next = [...prev];
          next[data.action] += 1;
          return next;
        });
      } catch (err) { console.error(err); }
    };

    socket.onerror = () => setError("Endpoint Unavailable");
    return () => socket.close();
  }, [modelId, episodeCount]);

  useEffect(() => {
    if (!isLive && history[currentEpisodeIndex]) {
      const episode = history[currentEpisodeIndex];
      const step = episode.steps[currentStepIndex];
      if (step) {
        setFrame(step.frame);
        setTelemetry({
          action: step.action, reward: step.reward, done: step.done,
          reset: false, agent_view: step.agent_view,
          full_grid: step.full_grid, agent_pos: step.agent_pos,
          agent_dir: step.agent_dir
        });
        setGridSize([step.full_grid.length, step.full_grid[0].length]);
        // Show path up to current step in replay
        setPath(episode.steps.slice(0, currentStepIndex + 1).map(s => s.agent_pos));
      }
    }
  }, [isLive, currentEpisodeIndex, currentStepIndex, history]);

  const successRate = episodeCount > 0 ? ((successCount / episodeCount) * 100).toFixed(1) : "0.0";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 h-full">
      {/* Visual Stream (Main) */}
      <div className="lg:col-span-7 flex flex-col gap-8">
        <div className="flex justify-between items-end border-b border-border pb-4">
          <div className="flex bg-accent-bg p-1 rounded">
            <button 
              onClick={() => { setIsLive(true); setPath([]); }}
              className={`px-4 py-1 text-[10px] font-bold tracking-widest transition-all ${isLive ? 'bg-fg text-bg' : 'text-sub hover:text-fg'}`}
            >LIVE</button>
            <button 
              onClick={() => setIsLive(false)}
              className={`px-4 py-1 text-[10px] font-bold tracking-widest transition-all ${!isLive ? 'bg-fg text-bg' : 'text-sub hover:text-fg'}`}
            >REPLAY</button>
          </div>
          
          <div className="flex gap-4 items-center">
            <button 
              onClick={() => setShowPath(!showPath)}
              className={`text-[10px] font-bold tracking-widest px-3 py-1 border transition-all ${showPath ? 'bg-fg text-bg border-fg' : 'text-sub border-border hover:border-sub'}`}
            >PATH</button>
            <button 
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`text-[10px] font-bold tracking-widest px-3 py-1 border transition-all ${showHeatmap ? 'bg-fg text-bg border-fg' : 'text-sub border-border hover:border-sub'}`}
            >HEATMAP</button>
          </div>

          <div className="text-right">
            <span className="text-[10px] text-sub uppercase tracking-widest block mb-1">Convergence</span>
            <span className="text-xl font-bold font-heading">{successRate}%</span>
          </div>
        </div>

        <div className="relative aspect-square border border-border bg-bg p-4 group">
          {frame ? (
            <>
              <img src={`data:image/jpeg;base64,${frame}`} className="w-full h-full object-contain pixelated transition-opacity duration-500" alt="Stream" />
              <SpatialOverlay 
                path={path} 
                heatmap={heatmap} 
                gridSize={gridSize} 
                visible={{ path: showPath, heatmap: showHeatmap }} 
              />
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-sub uppercase tracking-widest animate-pulse">Initializing Latent Stream...</div>
          )}
          {!isLive && <div className="absolute top-8 left-8 text-[10px] bg-fg text-bg px-2 py-1 font-bold tracking-tighter">REPLAY_BUFFER_ACTIVE</div>}
        </div>

        {!isLive && history.length > 0 && (
          <div className="space-y-6 animate-in slide-in-from-top-2 duration-500">
            <input 
              type="range" min="0" max={(history[currentEpisodeIndex]?.steps.length || 1) - 1}
              value={currentStepIndex} onChange={(e) => setCurrentStepIndex(Number(e.target.value))}
              className="w-full h-px bg-border appearance-none cursor-none accent-fg"
            />
            <div className="flex justify-between items-center text-[10px] text-sub uppercase tracking-widest">
              <button onClick={() => setCurrentStepIndex(p => Math.max(0, p-1))} className="hover:text-fg flex items-center gap-1"><ChevronLeft size={12}/> Prev</button>
              <div className="flex items-baseline gap-4">
                <select 
                  className="bg-bg border-none outline-none font-bold text-fg cursor-none"
                  value={currentEpisodeIndex} onChange={e => {setCurrentEpisodeIndex(Number(e.target.value)); setCurrentStepIndex(0);}}
                >
                  {history.map((ep, i) => <option key={i} value={i}>Episode {ep.id}</option>)}
                </select>
                <span>Step {currentStepIndex + 1} / {history[currentEpisodeIndex]?.totalSteps}</span>
              </div>
              <button onClick={() => setCurrentStepIndex(p => Math.min((history[currentEpisodeIndex]?.steps.length || 1)-1, p+1))} className="hover:text-fg flex items-center gap-1">Next <ChevronRight size={12}/></button>
            </div>
          </div>
        )}
      </div>

      {/* Telemetry (Side) */}
      <div className="lg:col-span-5 flex flex-col gap-12">
        <section>
          <h4 className="text-[10px] uppercase tracking-[0.2em] text-sub mb-6 border-b border-border pb-2 flex items-center gap-2">
            <Eye size={12}/> Partial Observation (7x7)
          </h4>
          <div className="max-w-[200px]">
            <AgentEye grid={telemetry?.agent_view} />
          </div>
        </section>

        <section className="flex-grow">
          <h4 className="text-[10px] uppercase tracking-[0.2em] text-sub mb-6 border-b border-border pb-2 flex items-center gap-2">
            <BarChart3 size={12}/> Policy Distribution
          </h4>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ACTION_LABELS.map((l, i) => ({ name: l, value: actionHistory[i] }))} layout="vertical">
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: 'var(--sub)', fontSize: 9 }} width={80} />
                <Tooltip cursor={{fill: 'var(--accent-bg)'}} contentStyle={{ backgroundColor: 'var(--bg)', border: '1px solid var(--border)', fontSize: '10px' }} />
                <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                  {ACTION_LABELS.map((_, i) => <Cell key={i} fill={i === telemetry?.action ? 'var(--fg)' : 'var(--border)'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section>
          <h4 className="text-[10px] uppercase tracking-[0.2em] text-sub mb-6 border-b border-border pb-2 flex items-center gap-2">
            <Target size={12}/> Event History
          </h4>
          <div className="space-y-1 max-h-[200px] overflow-y-auto pr-4">
            {history.map((res, i) => (
              <div 
                key={i} onClick={() => { setIsLive(false); setCurrentEpisodeIndex(i); setCurrentStepIndex(0); }}
                className="group flex justify-between items-center py-2 border-b border-border/50 cursor-none hover:bg-accent-bg transition-colors px-2"
              >
                <span className="text-[10px] font-bold tracking-tighter">[{res.finalReward > 0 ? 'SUCC' : 'FAIL'}] EP_{res.id}</span>
                <span className="text-[10px] text-sub font-mono">{res.totalSteps} steps</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default ModelWatcher;
