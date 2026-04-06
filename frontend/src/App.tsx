import { useState } from 'react'
import { Layout, Play, History, Brain, Zap, Settings2 } from 'lucide-react'
import ModelWatcher from './components/ModelWatcher'

const MODELS = [
  { id: 'dqn', name: 'DQN Baseline (5x5)', icon: Zap, color: 'text-yellow-400' },
  { id: 'ppo', name: 'PPO Symbolic (5x5)', icon: Brain, color: 'text-blue-400' },
  { id: 'rppo_baseline', name: 'RPPO Baseline (8x8)', icon: History, color: 'text-purple-400' },
  { id: 'rppo_curriculum', name: 'RPPO Curriculum (8x8)', icon: Play, color: 'text-green-400' },
]

function App() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-purple-600 p-2 rounded-lg">
              <Layout className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">RL Visualizer</h1>
              <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">Experimental Dashboard</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
             <button className="p-2 text-slate-400 hover:text-white transition-colors">
               <Settings2 size={20} />
             </button>
             <div className="h-6 w-px bg-slate-800"></div>
             <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs font-bold text-green-500 uppercase">System Online</span>
             </div>
          </div>
        </div>
      </header>

      <main className="flex-grow max-w-7xl mx-auto w-full p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <aside className="lg:col-span-1 space-y-6">
          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Select Environment</h3>
            <div className="space-y-2">
              {MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl border transition-all duration-200 group ${
                    selectedModel === model.id
                      ? 'bg-purple-600/10 border-purple-500 text-white'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700 hover:bg-slate-800/50'
                  }`}
                >
                  <model.icon size={20} className={selectedModel === model.id ? 'text-purple-400' : 'text-slate-500 group-hover:text-slate-300'} />
                  <span className="font-medium text-sm">{model.name}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl">
             <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Instructions</h4>
             <p className="text-xs text-slate-500 leading-relaxed">
               Select a model to begin the live inference stream. The dashboard will display real-time telemetry, action distributions, and agent-eye view data.
             </p>
          </section>
        </aside>

        {/* Content Area */}
        <div className="lg:col-span-3 min-h-[600px]">
          {selectedModel ? (
            <ModelWatcher 
              key={selectedModel}
              modelId={selectedModel} 
              name={MODELS.find(m => m.id === selectedModel)?.name || ''} 
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center bg-slate-900/30 border-2 border-dashed border-slate-800 rounded-2xl text-slate-500">
               <div className="p-4 bg-slate-900 rounded-full mb-4">
                 <Zap size={48} className="opacity-20" />
               </div>
               <p className="text-lg font-medium">No Model Selected</p>
               <p className="text-sm">Choose an experiment from the sidebar to start visualization</p>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-6 px-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center text-[10px] text-slate-500 uppercase font-bold tracking-[0.2em]">
          <span>© 2026 Reinforcement Learning Project</span>
          <div className="flex gap-6">
            <a href="#" className="hover:text-slate-300 transition-colors">Documentation</a>
            <a href="#" className="hover:text-slate-300 transition-colors">GitHub</a>
            <a href="#" className="hover:text-slate-300 transition-colors">Research Paper</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
