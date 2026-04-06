import React, { useEffect, useRef, useState } from 'react';
import { Activity, Target, BarChart3, AlertCircle, Eye } from 'lucide-react';
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
  final_reward?: number;
  steps?: number;
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

const COLORS = ['#8b5cf6', '#ec4899', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6366f1'];

// Map MiniGrid object IDs to colors
const OBJECT_COLOR_MAP: Record<number, string> = {
  0: '#1e293b', // Empty (Slate-800)
  1: '#94a3b8', // Wall (Light grey)
  2: '#1e293b', // Floor (Same as empty)
  3: '#475569', // Door
  4: '#fde047', // Key
  5: '#3b82f6', // Ball
  6: '#a855f7', // Box
  7: '#22c55e', // Goal (Green)
  8: '#ef4444', // Lava (Red)
};

const AgentEye: React.FC<{ grid?: number[][][] }> = ({ grid }) => {
  if (!grid || !Array.isArray(grid) || grid.length < 7) {
    return <div className="aspect-square bg-slate-900 rounded-lg animate-pulse" />;
  }
  
  // Transpose the grid if necessary (MiniGrid obs is often X,Y,C)
  const flatCells = [];
  try {
    for (let y = 0; y < 7; y++) {
      for (let x = 0; x < 7; x++) {
        if (grid[x] && grid[x][y]) {
          flatCells.push(grid[x][y]);
        } else {
          flatCells.push([0, 0, 0]); // Default fallback
        }
      }
    }
  } catch (e) {
    console.error("Error processing grid:", e);
    return <div className="aspect-square bg-slate-900 rounded-lg border border-red-500/30 flex items-center justify-center text-red-500 text-xs text-center p-4">Invalid Grid Data</div>;
  }

  return (
    <div className="grid grid-cols-7 gap-1 aspect-square bg-slate-900 p-2 rounded-lg border border-slate-700">
      {flatCells.map((cell, i) => {
          const objectId = cell[0] || 0;
          const state = cell[2] || 0;
          
          let bgColor = OBJECT_COLOR_MAP[objectId] || '#1e293b';
          if (objectId === 3) {
            if (state === 0) bgColor = '#ef4444'; 
            if (state === 1) bgColor = '#22c55e'; 
            if (state === 2) bgColor = '#eab308'; 
          }
          
          return (
            <div 
              key={i} 
              className="w-full h-full rounded-sm border border-black/10 transition-all duration-300"
              style={{ backgroundColor: bgColor }}
            />
          );
      })}
    </div>
  );
};

interface StepData {
  frame: string;
  agent_view: number[][][];
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

const ModelWatcher: React.FC<ModelWatcherProps> = ({ modelId, name }) => {
  const [frame, setFrame] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionHistory, setActionHistory] = useState<number[]>(new Array(7).fill(0));
  const [successCount, setSuccessCount] = useState(0);
  const [episodeCount, setEpisodeCount] = useState(0);
  
  // Analysis Mode State
  const [isLive, setIsLive] = useState(true);
  const [history, setHistory] = useState<EpisodeRecord[]>([]);
  const [currentEpisodeIndex, setCurrentEpisodeIndex] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [currentRecordingSteps, setCurrentRecordingSteps] = useState<StepData[]>([]);

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const url = `ws://${window.location.hostname}:8000/ws/${modelId}`;
    console.log(`[WebSocket] Connecting to ${url}`);
    
    const socket = new WebSocket(url);
    ws.current = socket;

    socket.onopen = () => {
      console.log(`[WebSocket] Connected`);
      setError(null);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          return;
        }

        const step: StepData = {
          frame: data.frame,
          agent_view: data.agent_view,
          action: data.action,
          reward: data.reward,
          done: data.done
        };

        // Update live view ONLY if we are in live mode
        // We use a functional update to avoid dependency on isLive state
        setIsLive(currentIsLive => {
          if (currentIsLive) {
            setFrame(data.frame);
            setTelemetry({
              action: data.action,
              reward: data.reward,
              done: data.done,
              reset: data.reset,
              agent_view: data.agent_view,
              final_reward: data.final_reward,
              steps: data.steps
            });
          }
          return currentIsLive;
        });

        // Record steps into the current episode buffer
        setCurrentRecordingSteps(prev => {
          const nextSteps = [...prev, step];
          
          // If the episode ended, commit the buffer to history
          if (data.reset) {
            setHistory(prevHistory => {
              const newRecord: EpisodeRecord = {
                id: episodeCount + 1,
                steps: nextSteps,
                finalReward: data.final_reward,
                totalSteps: data.steps
              };
              return [newRecord, ...prevHistory].slice(0, 10);
            });
            setEpisodeCount(c => c + 1);
            if (data.final_reward > 0) setSuccessCount(s => s + 1);
            return []; // Clear the buffer for the next episode
          }
          
          return nextSteps;
        });

        // Update global stats
        setActionHistory(prev => {
          const next = [...prev];
          next[data.action] += 1;
          return next;
        });

      } catch (err) {
        console.error("[WebSocket] Parse error", err);
      }
    };

    socket.onerror = () => setError("Connection Error");
    
    return () => {
      console.log("[WebSocket] Cleaning up");
      socket.close();
    };
  }, [modelId]); // ONLY reconnect if the modelId changes!

  // Handle Scrubbing
  useEffect(() => {
    if (!isLive && history[currentEpisodeIndex]) {
      const episode = history[currentEpisodeIndex];
      const step = episode.steps[currentStepIndex];
      if (step) {
        setFrame(step.frame);
        setTelemetry({
          action: step.action,
          reward: step.reward,
          done: step.done,
          reset: false,
          agent_view: step.agent_view
        });
      }
    }
  }, [isLive, currentEpisodeIndex, currentStepIndex, history]);

  const chartData = ACTION_LABELS.map((label, index) => ({
    name: label,
    value: actionHistory[index]
  }));

  const successRate = episodeCount > 0 ? ((successCount / episodeCount) * 100).toFixed(1) : "0.0";

  return (
    <div className="bg-slate-800 rounded-xl p-6 shadow-2xl border border-slate-700 h-full flex flex-col">
      {/* Header with Mode Toggle */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="text-purple-400" />
            {name}
          </h2>
          <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-700">
            <button 
              onClick={() => setIsLive(true)}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${isLive ? 'bg-red-500 text-white shadow-lg shadow-red-500/20' : 'text-slate-500 hover:text-slate-300'}`}
            >
              LIVE
            </button>
            <button 
              onClick={() => setIsLive(false)}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${!isLive ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/20' : 'text-slate-500 hover:text-slate-300'}`}
            >
              ANALYSIS
            </button>
          </div>
        </div>
        <div className="flex gap-4">
          <div className="bg-slate-900 px-3 py-1 rounded-lg border border-slate-700 flex flex-col items-center">
            <span className="text-[10px] text-slate-400 uppercase font-bold">Success Rate</span>
            <span className="text-green-400 font-mono text-lg">{successRate}%</span>
          </div>
        </div>
      </div>

      {error ? (
        <div className="flex-grow flex items-center justify-center">
          <div className="bg-red-900/20 border border-red-500/50 p-4 rounded-lg flex items-center gap-3 text-red-200">
            <AlertCircle />
            <p>{error}</p>
          </div>
        </div>
      ) : (
        <div className="flex-grow grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden">
          {/* Main View */}
          <div className="flex flex-col gap-4 h-full w-full max-w-[500px] mx-auto">
            <div className="relative aspect-square w-full min-h-[300px] bg-black rounded-lg overflow-hidden border border-slate-600 flex items-center justify-center shadow-inner">
              {frame ? (
                <img 
                  src={`data:image/jpeg;base64,${frame}`} 
                  alt="Agent View" 
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <div className="animate-pulse text-slate-500">Connecting to stream...</div>
              )}
              {!isLive && (
                <div className="absolute top-4 left-4 bg-blue-600 px-2 py-1 rounded text-[10px] font-bold text-white uppercase tracking-widest shadow-lg">
                  Replay Mode
                </div>
              )}
            </div>
            
            {/* Scrubber Controls */}
            {!isLive && (
              <div className="bg-slate-900 p-4 rounded-xl border border-blue-500/30 space-y-4 shadow-lg shadow-blue-500/5">
                <div className="flex justify-between items-center">
                  <select 
                    className="bg-slate-800 text-xs text-white border border-slate-700 rounded px-2 py-1 outline-none"
                    value={currentEpisodeIndex}
                    onChange={(e) => {
                      setCurrentEpisodeIndex(Number(e.target.value));
                      setCurrentStepIndex(0);
                    }}
                  >
                    {history.map((ep, i) => (
                      <option key={ep.id} value={i}>
                        Episode {ep.id} ({ep.finalReward > 0 ? 'Success' : 'Fail'})
                      </option>
                    ))}
                  </select>
                  <span className="text-[10px] text-slate-500 uppercase font-mono">
                    Step {currentStepIndex + 1} / {history[currentEpisodeIndex]?.totalSteps || 0}
                  </span>
                </div>
                <input 
                  type="range"
                  min="0"
                  max={(history[currentEpisodeIndex]?.steps.length || 1) - 1}
                  value={currentStepIndex}
                  onChange={(e) => setCurrentStepIndex(Number(e.target.value))}
                  className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-center gap-2">
                   <button 
                    onClick={() => setCurrentStepIndex(prev => Math.max(0, prev - 1))}
                    className="p-2 bg-slate-800 rounded-lg text-slate-400 hover:text-white border border-slate-700"
                   >
                     Backward
                   </button>
                   <button 
                    onClick={() => setCurrentStepIndex(prev => Math.min((history[currentEpisodeIndex]?.steps.length || 1) - 1, prev + 1))}
                    className="p-2 bg-slate-800 rounded-lg text-slate-400 hover:text-white border border-slate-700"
                   >
                     Forward
                   </button>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                <span className="block text-xs text-slate-500 mb-1">Last Action</span>
                <span className="text-white font-medium">{telemetry ? ACTION_LABELS[telemetry.action] : '-'}</span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-700">
                <span className="block text-xs text-slate-500 mb-1">Step Reward</span>
                <span className={`font-mono ${telemetry && telemetry.reward > 0 ? 'text-green-400' : 'text-slate-400'}`}>
                  {telemetry ? telemetry.reward.toFixed(3) : '0.000'}
                </span>
              </div>
            </div>
          </div>

          {/* Stats View */}
          <div className="flex flex-col gap-4 h-full">
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-700">
              <h3 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                <Eye size={16} />
                Agent-Eye View (7x7)
              </h3>
              <AgentEye grid={telemetry?.agent_view} />
            </div>

            <div className="bg-slate-900 p-4 rounded-lg border border-slate-700 flex-grow">
              <h3 className="text-sm font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
                <Target size={16} />
                Recent Results
              </h3>
              <div className="space-y-2 overflow-y-auto max-h-[250px]">
                {history.map((result, idx) => (
                  <div 
                    key={idx} 
                    onClick={() => {
                      setIsLive(false);
                      setCurrentEpisodeIndex(idx);
                      setCurrentStepIndex(0);
                    }}
                    className={`p-2 rounded border text-xs flex justify-between items-center cursor-pointer transition-colors ${result.finalReward > 0 ? 'bg-green-900/10 border-green-500/30 text-green-300 hover:bg-green-900/20' : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'}`}
                  >
                    <span className="font-bold">EP {result.id}: {result.finalReward > 0 ? 'SUCCESS' : 'FAILURE'}</span>
                    <span>{result.totalSteps} steps | R: {result.finalReward.toFixed(2)}</span>
                  </div>
                ))}
                {history.length === 0 && (
                   <p className="text-slate-500 text-sm italic">Waiting for first episode...</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelWatcher;
